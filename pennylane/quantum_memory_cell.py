import torch
import torch.nn as nn
import pennylane as qml

# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def memory_circuit_template(inputs, weights):
    """Template for the quantum circuit of a single memory cell (gates only)."""
    num_qubits = weights.shape[0] // 2
    # The circuit expects wires to be passed implicitly by the device.
    # It assumes wires are named 0, 1, 2, ...
    for i in range(num_qubits):
        qml.RY(inputs[:, 2*i], wires=i)
        qml.RZ(inputs[:, 2*i+1], wires=i)
    for i in range(num_qubits):
        qml.RY(weights[2*i], wires=i)
        qml.RZ(weights[2*i+1], wires=i)

# Define the QNode *outside* the class to ensure it can be pickled
def memory_circuit(inputs, weights):
    """The quantum circuit for a single memory cell."""
    memory_circuit_template(inputs, weights)
    num_qubits = weights.shape[0] // 2
    return qml.probs(wires=range(num_qubits))

class QuantumNeuronLevelModel(qml.qnn.TorchLayer):
    """
    A PennyLane TorchLayer that acts as a Quantum Neuron-Level Model.
    This can be used directly as an nn.Module.
    """
    def __init__(self, num_qubits, memory_length, hidden_size, slot_index=0):
        
        self.num_qubits = num_qubits
        self.num_circuit_inputs = 2 * num_qubits
        self.num_weights = num_qubits * 2
        self.slot_index = slot_index
        
        dev = qml.device("default.qubit", wires=self.num_qubits)
        
        # Define weight shapes for the TorchLayer
        weight_shapes = {"weights": self.num_weights}
        
        # Initialize the TorchLayer
        super().__init__(qml.QNode(memory_circuit, dev, interface='torch'), weight_shapes)

    def forward(self, inputs):
        # The forward pass is handled by the parent TorchLayer class
        return super().forward(inputs) 