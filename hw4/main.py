import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet import ResNet18  # Assuming ResNet18 is implemented in resnet.py

def calculate_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Deep Learning Experiments")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='GPU IDs for training')
    args = parser.parse_args()

    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DataLoader with CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler
    )

    # Model initialization
    model = ResNet18().to(device)

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)

    # Optimizer and Loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"\nBatch Size: {args.batch_size}, Epochs: {args.epochs}, GPU IDs: {args.gpu_ids}")

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        total_time, compute_time, communication_time, train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        print(f"Epoch {epoch + 1}/{args.epochs}: Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%, Total Time={total_time:.2f}s, Compute Time={compute_time:.2f}s, Communication Time={communication_time:.2f}s")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    total_compute_time = 0.0
    total_communication_time = 0.0
    start_epoch_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Start timing for compute
        start_compute_time = time.time()

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        compute_time = time.time() - start_compute_time
        total_compute_time += compute_time

        # Accumulate statistics
        total_loss += loss.item() * inputs.size(0)
        total_correct += calculate_accuracy(outputs, targets)
        total_samples += inputs.size(0)

        # Simulate communication timing (if distributed or parallel)
        start_communication_time = time.time()
        if isinstance(model, nn.DataParallel):
            print("Synchronizing")
            torch.cuda.synchronize()
        communication_time = time.time() - start_communication_time
        total_communication_time += communication_time

    total_epoch_time = time.time() - start_epoch_time
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100

    return total_epoch_time, total_compute_time, total_communication_time, avg_loss, avg_acc

if __name__ == "__main__":
    main()
