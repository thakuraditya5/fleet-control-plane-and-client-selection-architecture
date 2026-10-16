"""
Lightweight PyTorch Convolutional Neural Network and Data Utilities.
Represents edge vision and road/lane perception tasks on EV on-board ECUs.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_MNIST_CACHE = {}


class EdgeCNN(nn.Module):
    """
    Lightweight CNN tailored for resource-constrained edge ECUs.
    Features compact receptive fields and minimal parameter overhead.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Extracts model weights as a list of NumPy arrays for Flower."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    """Loads weights from a list of NumPy arrays into the PyTorch model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


def get_model_size_bytes(net: nn.Module) -> int:
    """Calculates model size in bytes based on 32-bit floating point weights."""
    total_elements = sum(p.numel() for p in net.parameters())
    return total_elements * 4  # 4 bytes per float32


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Trains the model locally on client edge device."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.to(device)
    net.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluates model performance on validation or test dataset."""
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def partition_dataset(
    num_clients: int = 20,
    batch_size: int = 32,
    data_dir: str = "./data",
    seed: int = 42,
    usage_profiles: Optional[List[str]] = None,
    rho: float = 0.0,
    max_client_examples: Optional[int] = None,
    test_sample_size: int = 2000,
    dirichlet_alpha: float = 0.8,
    favored_class_weight: float = 4.5,
    non_favored_class_weight: float = 0.5,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Partitions MNIST across clients. At rho=0, labels are independent of usage
    profile; at rho=1, most labels follow the client's heavy/light class group.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    cache_key = os.path.abspath(data_dir)
    if cache_key not in _MNIST_CACHE:
        _MNIST_CACHE[cache_key] = (
            datasets.MNIST(data_dir, train=True, download=True, transform=transform),
            datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        )
    train_dataset, test_dataset = _MNIST_CACHE[cache_key]

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be between 0 and 1")
    if usage_profiles is None:
        usage_profiles = ["light-use"] * num_clients
    if len(usage_profiles) != num_clients:
        raise ValueError("usage_profiles must contain one profile per client")

    rng = np.random.default_rng(seed)
    targets = np.array(train_dataset.targets)
    num_classes = 10
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_k = np.where(targets == c)[0]
        rng.shuffle(idx_k)
        independent = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        profile_weights = np.array([
            favored_class_weight if (profile == "heavy-use") == (c >= 5) else non_favored_class_weight
            for profile in usage_profiles
        ], dtype=float)
        profile_weights /= profile_weights.sum()
        proportions = (1.0 - rho) * independent + rho * profile_weights
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_splits = np.split(idx_k, split_points)

        for client_id, split in enumerate(idx_splits):
            client_indices[client_id].extend(split.tolist())

    client_loaders = []
    for indices in client_indices:
        rng.shuffle(indices)
        if max_client_examples is not None and len(indices) > max_client_examples:
            indices = rng.choice(indices, size=max_client_examples, replace=False).tolist()
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
        client_loaders.append(loader)

    test_rng = np.random.default_rng(seed + 10_000)
    test_indices = np.arange(len(test_dataset))
    if test_sample_size and test_sample_size < len(test_indices):
        test_indices = test_rng.choice(test_indices, size=test_sample_size, replace=False)
    test_subset = Subset(test_dataset, test_indices.tolist())
    test_loader = DataLoader(test_subset, batch_size=512, shuffle=False)

    return client_loaders, test_loader
