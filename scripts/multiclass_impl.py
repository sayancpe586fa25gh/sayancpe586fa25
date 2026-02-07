from sayancpe586fa25 import deepl
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/Android_Malware.csv")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--keyword", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_csv(args.data, low_memory=False)
df.columns = df.columns.str.strip()

drop_cols = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp"
]

df = df.drop(columns=drop_cols, errors="ignore")

X = df.drop(columns=["Label"])
y = df["Label"]
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

num_features = X_train.shape[1]
num_classes = len(np.unique(y_encoded))

model = deepl.SimpleNN(in_features=num_features, num_classes=num_classes)

trainer = deepl.ClassTrainer(
    X_train=X_train,
    Y_train=y_train,
    model=model,
    eta=args.lr,
    epochs=args.epochs,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer_cls=optim.Adam
)

trainer.train()

train_preds, _ = trainer.predict(X_train)
test_preds, _ = trainer.predict(X_test)

train_preds = train_preds.numpy()
test_preds = test_preds.numpy()

y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

metrics = {
    "train_accuracy": accuracy_score(y_train_np, train_preds),
    "train_precision": precision_score(y_train_np, train_preds, average="macro"),
    "train_recall": recall_score(y_train_np, train_preds, average="macro"),
    "train_f1": f1_score(y_train_np, train_preds, average="macro"),
    "test_accuracy": accuracy_score(y_test_np, test_preds),
    "test_precision": precision_score(y_test_np, test_preds, average="macro"),
    "test_recall": recall_score(y_test_np, test_preds, average="macro"),
    "test_f1": f1_score(y_test_np, test_preds, average="macro"),
}

timestamp = time.strftime("%Y%m%d%H%M%S")
csv_name = f"{args.keyword}_metrics_{timestamp}.csv"
csv_path = os.path.join(args.output_dir, csv_name)

pd.DataFrame([metrics]).to_csv(csv_path, index=False)

print(f"Saved metrics to {csv_path}")
