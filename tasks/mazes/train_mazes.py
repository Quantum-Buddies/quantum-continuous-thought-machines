import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from quantum_ctm.hybrid_ctm import HybridCTM
from quantum_ctm.data.custom_datasets import MazeImageFolder
from quantum_ctm.utils.losses import maze_loss

# --- Maze Specific Components ---

class DummyMazeDataset(Dataset):
    """
    A dummy dataset that generates random maze data on the fly.
    This avoids the need for a separate data generation step.
    """
    def __init__(self, num_samples=1000, maze_size=10, path_length=15, num_actions=4):
        self.num_samples = num_samples
        self.maze_size = maze_size
        self.path_length = path_length
        self.num_actions = num_actions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a dummy maze image (e.g., random noise)
        maze_image = torch.randn(1, self.maze_size, self.maze_size)
        # Generate a dummy path (sequence of actions)
        path = torch.randint(0, self.num_actions, (self.path_length,))
        return maze_image, path

def plot_maze(maze, path=None, title=""):
    """Plots a maze with an optional path."""
    fig, ax = plt.subplots(1)
    ax.imshow(maze.permute(1, 2, 0))
    if path is not None:
        # This is a simplified path plotting, assuming path is a list of (y,x) coordinates
        # A more sophisticated plotting would be needed to draw lines based on actions
        pass
    ax.set_title(title)
    plt.show()

# --- Main Training Script ---

def main():
    parser = argparse.ArgumentParser(description="Quantum CTM for Maze Solving")
    parser.add_argument('--backend', type=str, default='pennylane', choices=['pennylane', 'qiskit'], help='Quantum backend to use.')
    parser.add_argument('--data_root', type=str, default='data/mazes/medium', help='Root directory of the maze dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the CTM.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of recurrent iterations in the CTM.')
    parser.add_argument('--maze_route_length', type=int, default=50, help='Maximum length of the maze solution path.')
    parser.add_argument('--num_mem_slots', type=int, default=4, help='Number of quantum memory slots.')
    parser.add_argument('--num_qubits_per_slot', type=int, default=2, help='Number of qubits per memory slot.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    args = parser.parse_args()

    # --- Dataset ---
    # NOTE: You need to download the maze dataset from the original CTM repository
    # and place it in the `data/mazes/medium` directory.
    # https://github.com/SakanaAI/continuous-thought-machines
    if not os.path.exists(args.data_root):
        print(f"Dataset not found at {args.data_root}.")
        print("Creating a dummy dataset for demonstration purposes.")
        dummy_dir = os.path.join(args.data_root, 'train/dummy')
        os.makedirs(dummy_dir, exist_ok=True)
        # Create a dummy image that MazeImageFolder can process
        dummy_maze = np.zeros((39, 39, 3), dtype=np.uint8)
        dummy_maze[0, 0] = [255, 0, 0] # Start point (red)
        dummy_maze[1, 0] = [0, 0, 255] # Path (blue)
        dummy_maze[1, 1] = [0, 255, 0] # End point (green)
        dummy_image = Image.fromarray(dummy_maze)
        dummy_image.save(os.path.join(dummy_dir, 'dummy_maze.png'))


    train_dataset = MazeImageFolder(root=os.path.join(args.data_root, 'train'), maze_route_length=args.maze_route_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Model ---
    # The output dimension is 5 for the 5 possible actions (U, D, L, R, Wait)
    output_dim = args.maze_route_length * 5
    model = HybridCTM(
        input_size=3 * 39 * 39, # Placeholder, backbone will override
        output_dim=output_dim,
        hidden_size=args.hidden_size,
        num_mem_slots=args.num_mem_slots,
        num_qubits_per_slot=args.num_qubits_per_slot,
        backend_mode=args.backend,
        use_attention=True,
        backbone_type='resnet18' # Specify the backbone
    ).to(args.device)

    # --- Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting training on {args.device}...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(args.device).float()
            targets = torch.from_numpy(np.stack(targets)).to(args.device)

            optimizer.zero_grad()

            # The model returns predictions for all iterations
            predictions, certainties = model(images, iterations=args.iterations)
            
            # Reshape predictions to match the loss function's expectation
            # (B, route_length, num_actions, iterations)
            predictions = predictions.view(
                images.size(0), 
                args.maze_route_length, 
                5, 
                args.iterations
            )
            
            loss, _, _ = maze_loss(predictions, certainties, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    main() 