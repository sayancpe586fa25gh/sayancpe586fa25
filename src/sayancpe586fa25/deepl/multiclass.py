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


