import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import importlib
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

# This part will be handled by the dynamic import now

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HybridCTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_mem_slots,
                 num_qubits_per_slot, backend_mode: str = 'pennylane',
                 memory_length: int = 8, use_attention: bool = True,
                 backbone_type: str = 'resnet18', task_type: str = 'classification',
                 action_size: int = None):
        super(HybridCTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_mem_slots = num_mem_slots
        self.memory_length = memory_length
        self.backend_mode = backend_mode
        self.use_attention = use_attention
        self.backbone_type = backbone_type
        self.task_type = task_type
        self.action_size = action_size

        # --- Dynamically import backend-specific modules ---
        try:
            q_memory_module = importlib.import_module(f"quantum_ctm.{self.backend_mode}.quantum_memory_cell")
            QuantumNeuronLevelModel = q_memory_module.QuantumNeuronLevelModel
            if self.use_attention:
                q_sync_module = importlib.import_module(f"quantum_ctm.{self.backend_mode}.quantum_synchronization")
                QuantumSynchronizationLayer = q_sync_module.QuantumSynchronizationLayer
        except ImportError as e:
            raise ImportError(f"Could not import modules for backend '{self.backend_mode}'. Error: {e}")

        # --- Backbone for Image Processing ---
        self._create_backbone()

        # Classical components
        self.query_generator = nn.Linear(hidden_size, hidden_size)

        # --- Quantum Memory ---
        self.q_memory_models = nn.ModuleList()
        self.q_output_mappers = nn.ModuleList()
        self.q_trace_processors = nn.ModuleList()

        for i in range(num_mem_slots):
            qnlm_factory = QuantumNeuronLevelModel(
                num_qubits=num_qubits_per_slot,
                memory_length=memory_length,
                hidden_size=hidden_size,
                slot_index=i
            )

            if self.backend_mode == 'pennylane':
                torch_qnn = qnlm_factory
            else:
                from qiskit_machine_learning.connectors import TorchConnector
                qnn = qnlm_factory.create_qnn()
                initial_weights = torch.randn(qnn.num_weights)
                torch_qnn = TorchConnector(qnn, initial_weights=initial_weights)

            self.q_memory_models.append(torch_qnn)

            qnn_output_size = 2**num_qubits_per_slot
            self.q_output_mappers.append(nn.Linear(qnn_output_size, hidden_size))

            trace_input_size = hidden_size * memory_length
            qnn_input_size = qnlm_factory.num_circuit_inputs
            self.q_trace_processors.append(
                nn.Sequential(
                    nn.Linear(trace_input_size, trace_input_size // 4),
                    nn.ReLU(),
                    nn.Linear(trace_input_size // 4, qnn_input_size)
                )
            )

        # --- Synchronization and Attention ---
        if self.use_attention:
            self.sync_layer = QuantumSynchronizationLayer(self.q_memory_models, self.q_trace_processors)
            num_sync_pairs = len(self.sync_layer.correlation_connectors)
            self.d_action = num_sync_pairs // 2
            self.d_out = num_sync_pairs - self.d_action

            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
            self.q_proj = nn.Linear(self.d_action, hidden_size)

            self.certainty_head = nn.Sequential(nn.Linear(self.d_out, 1), nn.Sigmoid())
            
            if self.task_type == 'rl':
                self.actor = layer_init(nn.Linear(self.d_out, self.action_size), std=0.01)
                self.critic = layer_init(nn.Linear(self.d_out, 1), std=1)
            else:
                self.output_projector = nn.Linear(self.d_out, output_dim)
        else:
            # Fallback for non-attention mode
            if self.task_type == 'rl':
                self.actor = layer_init(nn.Linear(hidden_size, self.action_size), std=0.01)
                self.critic = layer_init(nn.Linear(hidden_size, 1), std=1)
            else:
                self.output_projector = nn.Linear(hidden_size, output_dim)


        self.state_updater = nn.Sequential(
            nn.Linear(hidden_size * (2 if self.use_attention else 1), hidden_size),
            nn.ReLU()
        )

        self.current_hidden_state = None

    def _create_backbone(self):
        if self.backbone_type == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Remove the final FC layer
            # Project ResNet's output features to our hidden_size
            self.input_layer = nn.Linear(resnet.fc.in_features, self.hidden_size)
        elif self.backbone_type == 'linear':
            self.backbone = nn.Identity()
            self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")


    def forward(self, x_batch, iterations: int = 10):
        batch_size = x_batch.size(0)

        # 1. Process input with backbone
        features = self.backbone(x_batch).flatten(1)
        processed_input = self.input_layer(features)

        kv = processed_input.unsqueeze(1)
        recurrent_hidden_state = torch.relu(processed_input)
        state_trace = torch.zeros(batch_size, self.hidden_size, self.memory_length, device=x_batch.device)

        predictions_over_time = []
        certainties_over_time = []

        for i in range(iterations):
            flat_trace = state_trace.view(batch_size, -1)

            # --- Quantum Memory Access ---
            # Simplified: For now, we assume all slots are accessed in each step.
            # A more sophisticated mechanism could select slots.
            memory_output_components = []
            for slot_idx in range(self.num_mem_slots):
                q_model = self.q_memory_models[slot_idx]
                q_mapper = self.q_output_mappers[slot_idx]
                q_trace_proc = self.q_trace_processors[slot_idx]

                slot_params = q_trace_proc(flat_trace)
                slot_q_output_probs = q_model(slot_params)
                slot_memory_output = q_mapper(slot_q_output_probs)
                memory_output_components.append(slot_memory_output)

            # Average the outputs from all memory slots
            memory_output_batch = torch.mean(torch.stack(memory_output_components), dim=0)

            if self.use_attention:
                sync_vector = self.sync_layer(state_trace)
                sync_action = sync_vector[:, :self.d_action]
                sync_out = sync_vector[:, self.d_action:]

                query = self.q_proj(sync_action).unsqueeze(1)
                attn_output, _ = self.attention(query, kv, kv)
                attn_output = attn_output.squeeze(1)

                combined_representation = torch.cat((recurrent_hidden_state, attn_output), dim=1)
                
                if self.task_type == 'rl':
                    # In RL, the output is handled by actor/critic heads
                    pass
                else:
                    step_output = self.output_projector(sync_out)
                    predictions_over_time.append(step_output.unsqueeze(-1))

                step_certainty = self.certainty_head(sync_out)
            else:
                combined_representation = recurrent_hidden_state
                if self.task_type == 'rl':
                    pass
                else:
                    step_output = self.output_projector(recurrent_hidden_state)
                    predictions_over_time.append(step_output.unsqueeze(-1))
                # Fake certainty if not using attention
                step_certainty = torch.ones(batch_size, 1, device=x_batch.device)

            recurrent_hidden_state = self.state_updater(combined_representation)
            state_trace = torch.cat((state_trace[..., 1:], recurrent_hidden_state.unsqueeze(-1)), dim=-1)

            certainties_over_time.append(step_certainty)

        # Final processing depends on the task
        if self.task_type == 'rl':
            # For RL, we return the final hidden state to be used by actor/critic
            return recurrent_hidden_state, sync_out if self.use_attention else None
        else:
            all_predictions = torch.cat(predictions_over_time, dim=-1)
            all_certainties = torch.cat(certainties_over_time, dim=1)
            return all_predictions, all_certainties

    def get_value(self, x, iterations: int = 10):
        hidden_state, sync_out = self.forward(x, iterations=iterations)
        return self.critic(sync_out if self.use_attention else hidden_state)

    def get_action_and_value(self, x, action=None, iterations: int = 10):
        hidden_state, sync_out = self.forward(x, iterations=iterations)
        input_for_heads = sync_out if self.use_attention else hidden_state
        
        logits = self.actor(input_for_heads)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), self.critic(input_for_heads) 