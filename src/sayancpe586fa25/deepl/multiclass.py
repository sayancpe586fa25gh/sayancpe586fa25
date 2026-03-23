import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import os



class SimpleNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SimpleNN, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)   # logits
        return x


class ClassTrainer:
    def __init__(
        self,
        X_train,
        Y_train,
        model,
        eta=0.001,
        epochs=10000,
        loss_fn=None,
        optimizer_cls=optim.Adam
    ):
        # (j) device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (a) and (b)
        self.X_train = X_train.to(self.device)
        self.Y_train = Y_train.to(self.device)

        # (i) model
        self.model = model.to(self.device)

        # (c) learning rate
        self.eta = eta

        # (d) epochs
        self.epochs = epochs

        # (e) loss function
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

        # (f) optimizer
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.eta)

        # (g) loss vector
        self.loss_vector = torch.zeros(self.epochs)

        # (h) accuracy vector
        self.accuracy_vector = torch.zeros(self.epochs)

    # --------------------------------------------------
    # (a) Train
    # --------------------------------------------------
    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            logits = self.model(self.X_train)
            loss = self.loss_fn(logits, self.Y_train)

            loss.backward()
            self.optimizer.step()

            # Store loss
            self.loss_vector[epoch] = loss.item()

            # Compute accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == self.Y_train).float().mean()
                self.accuracy_vector[epoch] = acc.item()

    # --------------------------------------------------
    # (b) Test
    # --------------------------------------------------
    def test(self, X_test, Y_test):
        self.model.eval()

        X_test = X_test.to(self.device)
        Y_test = Y_test.to(self.device)

        with torch.no_grad():
            logits = self.model(X_test)
            loss = self.loss_fn(logits, Y_test)

            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == Y_test).float().mean().item()
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "predictions": preds.detach().cpu(),
            "labels": Y_test.detach().cpu()
        }

    # --------------------------------------------------
    # (c) Predict
    # --------------------------------------------------
    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)

        return preds.detach().cpu(), logits.detach().cpu()

    # --------------------------------------------------
    # (d) Save model in ONNX format
    # --------------------------------------------------
    def save(self, file_name="trained_model.onnx"):
        dummy_input = torch.randn(
            1, self.X_train.shape[1], device=self.device
        )

        torch.onnx.export(
            self.model,
            dummy_input,
            file_name,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )

    # --------------------------------------------------
    # (e) Evaluation
    # --------------------------------------------------
    def evaluation(self, X_test, Y_test):
        # Training curves
        epochs = np.arange(self.epochs)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.loss_vector.numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accuracy_vector.numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")

        plt.tight_layout()
        plt.show()

        # Training metrics
        train_preds, _ = self.predict(self.X_train)
        train_labels = self.Y_train.cpu()

        # Test metrics
        test_results = self.test(X_test, Y_test)
        test_preds = test_results["predictions"]
        test_labels = test_results["labels"]

        # Confusion matrix (test)
        cm = confusion_matrix(test_labels, test_preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Test Confusion Matrix")
        plt.show()

        # Metrics
        metrics = {
            "Train Accuracy": accuracy_score(train_labels, train_preds),
            "Train Precision": precision_score(train_labels, train_preds),
            "Train Recall": recall_score(train_labels, train_preds),
            "Train F1": f1_score(train_labels, train_preds),
            "Test Accuracy": accuracy_score(test_labels, test_preds),
            "Test Precision": precision_score(test_labels, test_preds),
            "Test Recall": recall_score(test_labels, test_preds),
            "Test F1": f1_score(test_labels, test_preds),
        }

        return metrics


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class ImageNetCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Convolutional Blocks
        self.block1 = ConvLayer(3, 64)
        self.block2 = ConvLayer(64, 128)
        self.block3 = ConvLayer(128, 256)
        self.block4 = ConvLayer(256, 512)
        self.block5 = ConvLayer(512, 512)

        # Global Average Pool
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Global average pooling
        x = self.global_pool(x)      # (B, 512, 1, 1)
        x = torch.flatten(x, 1)      # (B, 512)

        # Classifier
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)              # logits (no softmax)

        return x

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

class CNNTrainer:
    def __init__(
        self,
        train_loader,
        model,
        val_loader=None,
        epochs=10000,
        loss_fn=None,
        ):
        # (j) device

        #device_id = get_best_gpu(strategy="utilization")
        device_id = get_best_gpu(strategy="memory")
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f"Selected GPU: {device_id}")

        # (a) and (b)
        self.train_loader = train_loader
        self.val_loader = val_loader
        # (i) model
        self.model = model.to(self.device)

        # (c) learning rate
        # (d) epochs
        self.epochs = epochs

        # (e) loss function
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        # (f) optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        # (g) loss vector
        self.loss_vector = torch.zeros(self.epochs)

        # (h) accuracy vector
        self.accuracy_vector = torch.zeros(self.epochs)

    # --------------------------------------------------
    # (a) Train
    # --------------------------------------------------
    def train(self):

        self.model.train()

        for epoch in range(self.epochs):

            epoch_loss = 0
            correct = 0
            total = 0

            for batch in self.train_loader:

                inputs = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)

                loss = self.loss_fn(logits, labels)

                loss.backward()

                self.optimizer.step()
                batch_size = inputs.size(0)
                epoch_loss += loss.item() * batch_size

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()

                total += batch_size

            self.scheduler.step()
            epoch_loss = epoch_loss / total
            epoch_acc = correct / total

            self.loss_vector[epoch] = epoch_loss
            self.accuracy_vector[epoch] = epoch_acc

            print(
                f"Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )
    

    # --------------------------------------------------
    # (b) Test
    # --------------------------------------------------
    def test(self, loader):

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for batch in loader:

                inputs = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(inputs)

                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        accuracy = correct / total

        return {
            "loss": total_loss / len(loader),
            "accuracy": accuracy,
            "predictions": torch.cat(all_preds),
            "labels": torch.cat(all_labels)
        }

    # --------------------------------------------------
    # (c) Predict
    # --------------------------------------------------
    def predict(self, loader):

        self.model.eval()

        all_preds = []
        all_logits = []

        with torch.no_grad():

            for batch in loader:

                inputs = batch["pixel_values"].to(self.device)

                logits = self.model(inputs)

                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_logits.append(logits.cpu())

        return torch.cat(all_preds), torch.cat(all_logits)

    # --------------------------------------------------
    # (d) Save model in ONNX format
    # --------------------------------------------------
    def save(self, file_name="trained_CNN.onnx"):

        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            file_name,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )

        print(f"Model saved to {file_name}")

    # --------------------------------------------------
    # (e) Evaluation
    # --------------------------------------------------
    def evaluation(self):

        epochs = np.arange(self.epochs)

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(epochs, self.loss_vector.numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.subplot(1,2,2)
        plt.plot(epochs, self.accuracy_vector.numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")

        plt.tight_layout()
        plt.savefig('CNN_Training.png', bbox_inches='tight')
        plt.show()

        # Train metrics
        train_results = self.test(self.train_loader)

        # Validation metrics
        if self.val_loader is not None:
            val_results = self.test(self.val_loader)

            cm = confusion_matrix(
                val_results["labels"],
                val_results["predictions"]
            )

            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.title("Validation Confusion Matrix")
            plt.savefig('CNN_Validation_CM.png', bbox_inches='tight')
            plt.show()

        metrics = {
            "Train Accuracy": train_results["accuracy"],
            "Train Loss": train_results["loss"]
        }

        if self.val_loader is not None:
            metrics.update({
                "Val Accuracy": val_results["accuracy"],
                "Val Loss": val_results["loss"]
            })

        return metrics
