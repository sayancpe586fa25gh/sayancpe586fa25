import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader, random_split
from sayancpe586fa25 import deepl

import subprocess
def get_best_gpu(strategy="utilization"):
    """
    Select best GPU by 'utilization' or 'memory'.
    """
    if strategy == "memory":
    # Use PyTorch directly for free memory
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i) # (free, total)
            free_mem.append(props[0])
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
            )
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))

    # Pick strategy: "utilization" or "memory"
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: {device_id}")

# -------------------------------------------------
# Compute normalization statistics
# -------------------------------------------------
# -------------------------------------------------
# Compute normalization statistics safely
# -------------------------------------------------
def compute_normalization(train_dataset):
    """
    Compute mean and std from the training dataset only,
    then normalize the full dataset (train + validation)
    """
    # Select training rows only
    X_train = train_dataset.dataset.X[train_dataset.indices]

    # Compute mean and std along features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8  # prevent division by zero

    # Normalize full dataset using training statistics
    train_dataset.dataset.X = (train_dataset.dataset.X - mean) / std

    return mean, std

# -------------------------------------------------
# Training function
# -------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    total_loss = 0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = loss_fn(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------
# Validation
# -------------------------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score

def validate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str,
                        default="/data/CPE_487-587/ACCDataset")

    args = parser.parse_args()

    # -------------------------------------------------
    # Select GPU
    # -------------------------------------------------
    device_id = get_best_gpu(strategy="memory")
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Selected GPU: {device_id}")
    

    print("Using device:", device)

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    dataset = deepl.ACCDataset(args.data_path, k=50)

    # -------------------------------------------------
    # Normalize dataset
    # -------------------------------------------------

    #mean, std = compute_normalization(dataset)

    # -------------------------------------------------
    # Train / Validation split
    # -------------------------------------------------

    

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    mean, std = compute_normalization(train_dataset)
    
    # -------------------------------------------------
    # DataLoaders
    # -------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = deepl.ACCNet().to(device)

    # -------------------------------------------------
    # Loss + Optimizer
    # -------------------------------------------------

    loss_fn = deepl.FocalTverskyLoss(alpha=0.3, gamma=0.7)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=5
    )
    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    train_losses = []
    val_accuracies = []
    print("ACC ON:", (dataset.Y == 1).sum().item())
    print("ACC OFF:", (dataset.Y == 0).sum().item())
    for epoch in range(args.epochs):

        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        metrics = validate(model, val_loader, device)
        train_losses.append(train_loss)
        val_accuracies.append(metrics['accuracy'])
        scheduler.step(train_loss)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

    # -------------------------------------------------
    # Plot training curves
    # -------------------------------------------------

    epochs = np.arange(args.epochs)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()

    plt.savefig("training_curves.png")

    print("Saved training_curves.png")

    # -------------------------------------------------
    # Save ONNX model
    # -------------------------------------------------

    dummy_input = torch.randn(1, 11).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        "acc_model.onnx",
        export_params=True,
        do_constant_folding=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"]
    )

    print("Saved acc_model.onnx")

    # -------------------------------------------------
    # Save normalization coefficients
    # -------------------------------------------------

    norm_data = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }

    with open("normalization.json", "w") as f:
        json.dump(norm_data, f)

    print("Saved normalization.json")


# -------------------------------------------------
if __name__ == "__main__":
    main()
