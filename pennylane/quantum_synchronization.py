import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from typing import List

from quantum_ctm.pennylane.quantum_memory_cell import memory_circuit_template

class QuantumSynchronizationLayer(nn.Module):
    """
    Computes a synchronization vector by measuring the correlation between pairs of quantum memory slots using PennyLane.
    """
    def __init__(self, q_memory_models: nn.ModuleList, q_trace_processors: nn.ModuleList):
        super().__init__()
        self.q_memory_models = q_memory_models
        self.q_trace_processors = q_trace_processors
        self.num_slots = len(q_memory_models)
        self._create_correlation_circuits()

    def _create_correlation_circuits(self):
        self.correlation_connectors = nn.ModuleList()
        
        for i in range(self.num_slots):
            for j in range(i + 1, self.num_slots):
                q_model_i = self.q_memory_models[i]
                q_model_j = self.q_memory_models[j]
                
                num_qubits_i = len(q_model_i.qnode.device.wires)
                num_qubits_j = len(q_model_j.qnode.device.wires)
                total_qubits = num_qubits_i + num_qubits_j
                
                dev = qml.device("default.qubit", wires=total_qubits)

                @qml.qnode(dev, interface='torch')
                def correlation_circuit(inputs, weights):
                    # Split inputs and weights for each model
                    inputs_i, inputs_j = torch.split(inputs, [q_model_i.num_circuit_inputs, q_model_j.num_circuit_inputs], dim=1)
                    weights_i, weights_j = torch.split(weights, [q_model_i.num_weights, q_model_j.num_weights])
                    
                    # Apply the first model's circuit logic
                    memory_circuit_template(inputs_i, weights_i)
                    
                    # Apply the second model's circuit on different wires using qml.map_wires
                    qml.map_wires(memory_circuit_template, {k: k + num_qubits_i for k in range(num_qubits_j)})(inputs_j, weights_j)

                    # Add entangling CNOT
                    qml.CNOT(wires=[0, num_qubits_i])
                    
                    return qml.probs(wires=range(total_qubits))

                # Define the weight shapes for the TorchLayer
                weight_shapes = {"weights": q_model_i.num_weights + q_model_j.num_weights}
                corr_layer = qml.qnn.TorchLayer(correlation_circuit, weight_shapes)
                self.correlation_connectors.append(corr_layer)

    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        batch_size = state_trace.size(0)
        flat_trace = state_trace.view(batch_size, -1)
        
        sync_values_all_items = []
        connector_idx = 0
        for i in range(self.num_slots):
            for j in range(i + 1, self.num_slots):
                corr_connector = self.correlation_connectors[connector_idx]
                proc_i = self.q_trace_processors[i]
                proc_j = self.q_trace_processors[j]
                
                params_i = proc_i(flat_trace)
                params_j = proc_j(flat_trace)
                
                combined_params = torch.cat([params_i, params_j], dim=1)
                
                q_output_probs = corr_connector(combined_params)
                
                num_qubits_i = len(self.q_memory_models[i].qnode.device.wires)
                num_qubits_j = len(self.q_memory_models[j].qnode.device.wires)
                if num_qubits_i == 1 and num_qubits_j == 1:
                    sync_metric = q_output_probs[:, 1] + q_output_probs[:, 2]
                else:
                    sync_metric = torch.mean(q_output_probs, dim=1)
                
                sync_values_all_items.append(sync_metric.unsqueeze(1))
                connector_idx += 1
                
        sync_vector = torch.cat(sync_values_all_items, dim=1)
        return sync_vector 