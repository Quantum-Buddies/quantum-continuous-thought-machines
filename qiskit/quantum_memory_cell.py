import torch
import torch.nn as nn
from typing import Tuple

# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumNeuronLevelModel:
    """
    This class constructs a Qiskit SamplerQNN that acts as a Quantum Neuron-Level Model.
    It is no longer an nn.Module itself, but a factory for creating a QNN
    that can be wrapped by TorchConnector.
    """
    def __init__(self, num_qubits, memory_length, hidden_size, slot_index=0):
        self.num_qubits = num_qubits
        self.memory_length = memory_length
        self.hidden_size = hidden_size
        self.slot_index = slot_index
        # The number of parameters for the feature map circuit
        self.num_circuit_inputs = 2 * num_qubits
        
        # Define the feature map and ansatz for our QNN
        self._feature_map, self._ansatz = self._create_circuit()
        
        # Define the full circuit and its parameters
        self.circuit = self._feature_map.compose(self._ansatz)
        self.input_params = list(self._feature_map.parameters)
        self.weight_params = list(self._ansatz.parameters)

    def _create_circuit(self) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Creates the feature map and ansatz circuits."""
        
        # Feature map encodes the classical features into the quantum state.
        feature_map = QuantumCircuit(self.num_qubits, name=f"FeatureMap_s{self.slot_index}")
        feature_params = [Parameter(f'x_s{self.slot_index}_{i}') for i in range(self.num_circuit_inputs)]
        
        # A simple encoding where each qubit gets two rotation gates parameterized by input features.
        for i in range(self.num_qubits):
            feature_map.ry(feature_params[2*i], i)
            feature_map.rz(feature_params[2*i + 1], i)

        # Ansatz represents the trainable weights of the QNLM
        ansatz = QuantumCircuit(self.num_qubits, name=f"Ansatz_s{self.slot_index}")
        num_weights = self.num_qubits * 2 # Example: 2 weights per qubit
        weight_params = [Parameter(f'w_s{self.slot_index}_{i}') for i in range(num_weights)]
        for i in range(self.num_qubits):
            ansatz.ry(weight_params[2*i], i)
            ansatz.rz(weight_params[2*i + 1], i)
        
        return feature_map, ansatz

    def create_qnn(self) -> SamplerQNN:
        """Instantiates and returns the SamplerQNN."""
        
        qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
        )
        return qnn 