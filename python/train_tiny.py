"""
Tiny Linear Model for FEDSTR Phase 1
=====================================

This implements an ultra-simple linear model that can be trained on MNIST
and has parameters small enough to fit in NOSTR events (<100 KB).

Model: 784 -> 128 -> 10 (total ~100k params = ~400KB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import io
import hashlib
import base64
from typing import Tuple, Dict, List


class TinyLinearNet(nn.Module):
    """Ultra-simple linear network for testing"""
    
    def __init__(self):
        super(TinyLinearNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def train_model(
    dataset_name: str,
    start_idx: int,
    end_idx: int,
    epochs: int,
    batch_size: int,
    initial_params: str = None
) -> Dict:
    """
    Train tiny model on a subset of the dataset.
    
    Args:
        dataset_name: "mnist" or "fashion-mnist"
        start_idx: Starting index for data subset
        end_idx: Ending index for data subset
        epochs: Number of training epochs
        batch_size: Batch size for training
        initial_params: Base64-encoded initial parameters (for rounds 2+)
    
    Returns:
        Dictionary with trained model and metrics
    """
    print(f"\n{'='*60}")
    print(f"FEDSTR Tiny Model Training")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Data range: {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    if dataset_name == "mnist":
        full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    elif dataset_name == "fashion-mnist":
        full_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create subset
    subset_indices = list(range(start_idx, min(end_idx, len(full_dataset))))
    train_dataset = Subset(full_dataset, subset_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Actual training samples: {len(train_dataset)}")
    
    # Model initialization
    model = TinyLinearNet().to(device)
    
    # Load initial parameters if provided (for federated rounds 2+)
    if initial_params:
        print("Loading initial parameters from previous round...")
        params_bytes = base64.b64decode(initial_params)
        buffer = io.BytesIO(params_bytes)
        state_dict = torch.load(buffer, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Parameters loaded\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training metrics
    loss_history: List[float] = []
    accuracy_history: List[float] = []
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # Epoch metrics
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = 100. * correct / total
        
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    print("\n✓ Training completed!")
    
    # Serialize model
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    
    # Calculate hash
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    
    # Encode to base64
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    
    print(f"\nModel Statistics:")
    print(f"  Size: {len(model_bytes) / 1024:.2f} KB")
    print(f"  Hash: {model_hash[:16]}...")
    print(f"  Final Loss: {loss_history[-1]:.4f}")
    print(f"  Final Accuracy: {accuracy_history[-1]:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        "model_base64": model_base64,
        "model_hash": model_hash,
        "size_bytes": len(model_bytes),
        "final_loss": loss_history[-1],
        "final_accuracy": accuracy_history[-1],
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
        "epochs_completed": epochs,
        "training_samples": len(train_dataset)
    }


def load_and_evaluate(model_base64: str, dataset_name: str = "mnist") -> Dict:
    """
    Load a trained model and evaluate it on the test set.
    Useful for verification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TinyLinearNet().to(device)
    model_bytes = base64.b64decode(model_base64)
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name == "mnist":
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    else:
        test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Evaluation
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_dataset)
    test_accuracy = 100. * correct / total
    
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }


if __name__ == "__main__":
    # Quick test
    print("Testing tiny model training...")
    result = train_model(
        dataset_name="mnist",
        start_idx=0,
        end_idx=10000,
        epochs=3,
        batch_size=64
    )
    
    print(f"\nModel can be embedded: {len(result['model_base64']) < 1000000}")  # <1MB for NOSTR
    
    # Evaluate
    eval_result = load_and_evaluate(result['model_base64'])
    print(f"\nTest Set Performance:")
    print(f"  Loss: {eval_result['test_loss']:.4f}")
    print(f"  Accuracy: {eval_result['test_accuracy']:.2f}%")
