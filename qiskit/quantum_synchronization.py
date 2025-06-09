import torch
import torch.nn as nn
from typing import List

# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit
from qiskit.circuit.parameter import Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

class QuantumSynchronizationLayer(nn.Module):
    """
    Computes a synchronization vector by measuring the correlation between pairs of quantum memory slots.
    """
    def __init__(self, q_memory_models: nn.ModuleList, q_trace_processors: nn.ModuleList):
        """
        Args:
            q_memory_models: The list of TorchConnector-wrapped QNNs from the HybridCTM.
            q_trace_processors: The list of classical MLPs that process state traces.
        """
        super().__init__()
        self.q_memory_models = q_memory_models
        self.q_trace_processors = q_trace_processors
        self.num_slots = len(q_memory_models)

        # Create pairwise correlation circuits
        self._create_correlation_circuits()

    def _create_correlation_circuits(self):
        """
        For each pair of memory slots, create a circuit that entangles them
        and a corresponding TorchConnector to execute it.
        """
        self.correlation_connectors = nn.ModuleList()
        
        for i in range(self.num_slots):
            for j in range(i + 1, self.num_slots):
                # Get the original circuits and parameters from the two slots
                qnn_i = self.q_memory_models[i].neural_network
                qnn_j = self.q_memory_models[j].neural_network
                
                num_qubits_i = qnn_i.circuit.num_qubits
                num_qubits_j = qnn_j.circuit.num_qubits

                # Create a new circuit to measure correlation
                corr_circuit = QuantumCircuit(num_qubits_i + num_qubits_j, name=f'sync_{i}-{j}')
                
                # Apply the original circuits for each slot
                corr_circuit.compose(qnn_i.circuit, qubits=range(num_qubits_i), inplace=True)
                corr_circuit.compose(qnn_j.circuit, qubits=range(num_qubits_i, num_qubits_i + num_qubits_j), inplace=True)

                # Add CNOTs to entangle the first qubit of each slot
                corr_circuit.cx(0, num_qubits_i)
                
                # Input parameters are the combined inputs of both original circuits
                input_params = qnn_i.input_params + qnn_j.input_params
                
                # Weight parameters are the combined weights of both original circuits
                weight_params = qnn_i.weight_params + qnn_j.weight_params

                # Create a QNN for this correlation circuit
                # The output will be the probability distribution of the combined system
                corr_qnn = SamplerQNN(
                    circuit=corr_circuit,
                    input_params=input_params,
                    weight_params=weight_params
                )
                
                # Wrap in a TorchConnector
                # Initialize with random weights - these will be optimized during training
                initial_weights = torch.randn(corr_qnn.num_weights)
                corr_connector = TorchConnector(corr_qnn, initial_weights=initial_weights)
                self.correlation_connectors.append(corr_connector)

    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_trace: The history of hidden states, shape (batch_size, hidden_size, memory_length).
        
        Returns:
            A synchronization vector of shape (batch_size, num_pairs).
        """
        batch_size = state_trace.size(0)
        flat_trace = state_trace.view(batch_size, -1)
        
        sync_values_all_items = []
        connector_idx = 0
        for i in range(self.num_slots):
            for j in range(i + 1, self.num_slots):
                corr_connector = self.correlation_connectors[connector_idx]
                proc_i = self.q_trace_processors[i]
                proc_j = self.q_trace_processors[j]
                
                # Process the trace for each model in the pair
                params_i = proc_i(flat_trace)
                params_j = proc_j(flat_trace)
                
                # Combine parameters for the correlation circuit
                combined_params = torch.cat([params_i, params_j], dim=1)
                
                # Execute the correlation circuit
                q_output_probs = corr_connector(combined_params)
                
                # Calculate a synchronization metric.
                # Here, we assume 1 qubit per slot for simplicity to get P(01) + P(10).
                # A real implementation needs to handle arbitrary qubit counts.
                num_qubits_i = self.q_memory_models[i].neural_network.circuit.num_qubits
                num_qubits_j = self.q_memory_models[j].neural_network.circuit.num_qubits
                if num_qubits_i == 1 and num_qubits_j == 1: # If 2 total qubits
                    # Probabilities for |00>, |01>, |10>, |11>
                    sync_metric = q_output_probs[:, 1] + q_output_probs[:, 2]
                else:
                    sync_metric = torch.mean(q_output_probs, dim=1) # Fallback for other sizes
                
                sync_values_all_items.append(sync_metric.unsqueeze(1))
                connector_idx += 1
                
        # Concatenate results from all pairs
        sync_vector = torch.cat(sync_values_all_items, dim=1)
        return sync_vector 