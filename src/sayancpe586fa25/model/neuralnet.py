import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort

class TorchNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TorchNet, self).__init__()
        self.scaler = StandardScaler()

        # Define network layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim),
            nn.Sigmoid()
        )

    def standardize_data(self, X):
        return self.scaler.fit_transform(X)

    def forward(self, x):
        return self.model(x)

    def train_model(self, X_train, y_train, epochs=1000, lr=0.01):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.show()

    def save_onnx(self, filename):
        dummy_input = torch.randn(1, 8)
        torch.onnx.export(self, dummy_input, filename, input_names=['input'], output_names=['output'])

    def predict_from_onnx(self, filename, sample, scaler=None):
        session = ort.InferenceSession(filename)
        input_name = session.get_inputs()[0].name
        if scaler:
            sample = scaler.transform(sample)
        pred = session.run(None, {input_name: sample.astype(np.float32)})[0]
        return (pred > 0.5).astype(int)

