import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import torch.nn as nn

class FourierFeatureLayer(nn.Module):

    def forward(self, x):
        # x shape: (batch, seq_len)

        # Compute FFT
        fft = torch.fft.rfft(x, dim=1)

        # Separate real and imaginary parts
        real = fft.real
        imag = fft.imag

        # Concatenate
        features = torch.cat([real, imag], dim=1)

        return features
class VelocityDerivativeLayer(nn.Module):

    def forward(self, x):

        dv = x[:,1:] - x[:,:-1]

        dv = torch.nn.functional.pad(dv,(1,0))

        return torch.cat([x,dv],dim=1)


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        return self.relu(out)

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

class FocalTverskyLoss(nn.Module):

    def __init__(self, alpha=0.4, beta=0.6, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )   
        loss = (1 - tversky) ** self.gamma

        return loss

class ACCDataset(Dataset):

    def __init__(self, data_path, k=10):

        self.k = k
        X_all = []
        y_all = []

        speed_files = glob.glob(f"{data_path}/*decoded_wheel_speed_fl.csv")

        for sf in speed_files:

            ts = sf.split("/")[-1].split("_Messages_")[0]
            acc_file = f"{data_path}/{ts}_Messages_decoded_acc_status.csv"

            speed = pd.read_csv(sf)[["Time","Message"]]
            acc = pd.read_csv(acc_file)[["Time","Message"]]
            
            # convert units
            speed["Message"] = speed["Message"] * (1000/3600)

            speed = speed.rename(columns={"Message":"speed"})
            acc = acc.rename(columns={"Message":"acc"})
            
            # binarize
            acc["acc"] = (acc["acc"] == 6).astype(int)

            speed = speed.set_index("Time")
            acc = acc.set_index("Time")
            # remove duplicate timestamps in acc
            acc = acc[~acc.index.duplicated(keep="last")]
            acc = acc.reindex(speed.index, method="ffill")

            df = pd.concat([speed, acc], axis=1)

            for i in range(k+1):
                df[f"v_t-{i}"] = df["speed"].shift(i)

            df = df.dropna()

            features = df[[f"v_t-{i}" for i in range(k+1)]].values
            labels = df["acc"].values

            X_all.append(features)
            y_all.append(labels)

        self.X = torch.tensor(np.vstack(X_all), dtype=torch.float32)
        self.Y = torch.tensor(np.concatenate(y_all), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(self.bn(self.conv(x)))
class ACCNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.derivative = VelocityDerivativeLayer()

        self.net = nn.Sequential(

            TCNBlock(1,32,1),
            TCNBlock(32,32,2),
            TCNBlock(32,32,4),
            TCNBlock(32,32,8),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(32,32),
            nn.ReLU(),

            nn.Linear(32,1)
        )

    def forward(self,x):

        x = self.derivative(x)      # (batch,22)

        x = x.unsqueeze(1)          # (batch,1,22)

        x = self.net(x)

        x = x.squeeze(-1)

        return self.classifier(x).squeeze(-1)
"""
class ACCNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.derivative = VelocityDerivativeLayer()

        self.input_conv = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.resblock1 = ResidualBlock(32)
        self.resblock2 = ResidualBlock(32)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,1)
        )

    def forward(self,x):

        # x = (batch,11)

        x = self.derivative(x)     # (batch,22)

        x = x.unsqueeze(1)         # (batch,1,22)

        x = self.input_conv(x)     # (batch,32,22)

        x = self.resblock1(x)
        x = self.resblock2(x)

        x = self.pool(x)           # (batch,32,1)

        x = x.squeeze(-1)          # (batch,32)

        x = self.classifier(x)

        return x.squeeze(-1)

class ACCNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.derivative = VelocityDerivativeLayer()
        self.fourier = FourierFeatureLayer()

        self.net = nn.Sequential(

            nn.Linear(24,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)
        )

    def forward(self,x):
        x = self.derivative(x)      # (batch,22)
        x = self.fourier(x)         # (batch,24)
        return self.net(x).squeeze(-1)
"""
