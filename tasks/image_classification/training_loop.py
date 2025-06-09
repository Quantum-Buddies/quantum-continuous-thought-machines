import torch
import torch.nn as nn
import torch.optim as optim
python quantum_ctm/tasks/image_classification/training_loop.py --epochs 1 --batch_size 32 --backend pennylanefrom torch.utils.data import DataLoader
import numpy as np
import time
import argparse
from typing import Dict, Optional, Callable
from torchvision import datasets, transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from quantum_ctm.hybrid_ctm import HybridCTM
from quantum_ctm.utils.losses import image_classification_loss

class QuantumCTMTrainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: str = "cpu", iterations: int = 5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.iterations = iterations
        self.training_history = []
        print(f"[Trainer] Initialized for standard PyTorch training with iterations={iterations} on device {device}.")

    def train_epoch(self, dataloader: DataLoader, loss_fn: Callable, verbose: bool = True) -> Dict:
        self.model.train()
        epoch_start_time = time.time()
        batch_losses = []
        total_batches = len(dataloader)
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            batch_start_time = time.time()
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, certainties = self.model(x_batch, iterations=self.iterations)
            
            loss, _ = loss_fn(predictions, certainties, y_batch)
            
            loss.backward()
            self.optimizer.step()
            
            batch_losses.append(loss.item())
            batch_time = time.time() - batch_start_time
            
            if verbose and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
                print(f"  Batch {batch_idx + 1}/{total_batches}: Loss={loss.item():.4f}, BatchTime={batch_time:.2f}s")
        
        epoch_time = time.time() - epoch_start_time
        return {'avg_loss': np.mean(batch_losses), 'epoch_time': epoch_time}

    def evaluate(self, dataloader: DataLoader, loss_fn: Callable) -> Dict:
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                predictions, certainties = self.model(x_batch, iterations=self.iterations)
                loss, _ = loss_fn(predictions, certainties, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)
        return {'avg_loss': total_loss / total_samples}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, num_epochs: int = 10,
              save_path: Optional[str] = None, validation_freq: int = 1):
        print("\n[Trainer] Starting training...")
        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_metrics = self.train_epoch(train_loader, image_classification_loss)
            print(f"  Epoch {epoch} summary: Avg Loss={train_metrics['avg_loss']:.4f}, Time={train_metrics['epoch_time']:.2f}s")

            if val_loader and (epoch % validation_freq == 0):
                val_metrics = self.evaluate(val_loader, image_classification_loss)
                print(f"  Validation: Avg Loss={val_metrics['avg_loss']:.4f}")
                if val_metrics['avg_loss'] < best_val_loss:
                    best_val_loss = val_metrics['avg_loss']
                    if save_path:
                        self.save_model(save_path)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum CTM Trainer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_qubits_per_slot", type=int, default=2)
    parser.add_argument("--num_mem_slots", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--memory_length", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="hybrid_ctm_mnist.pth")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--backend", type=str, default="pennylane", choices=["qiskit", "pennylane"])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu'
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
    ])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = HybridCTM(
        input_size=3 * 32 * 32, # Placeholder, backbone handles actual size
        hidden_size=args.hidden_size,
        output_dim=10,
        num_mem_slots=args.num_mem_slots,
        num_qubits_per_slot=args.num_qubits_per_slot,
        memory_length=args.memory_length,
        backend_mode=args.backend,
        backbone_type='resnet18',
        task_type='classification',
        use_attention=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = QuantumCTMTrainer(
        model=model,
        optimizer=optimizer,
        iterations=args.iterations,
        device=device
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=args.save_path
    )

if __name__ == '__main__':
    main() 