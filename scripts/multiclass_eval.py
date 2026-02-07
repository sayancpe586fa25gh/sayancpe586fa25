import argparse
import glob
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--keyword", type=str, required=True)
parser.add_argument("--input_dir", type=str, default="results")
parser.add_argument("--output_dir", type=str, default="results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

pattern = os.path.join(args.input_dir, f"*{args.keyword}*_metrics_*.csv")
files = glob.glob(pattern)

if len(files) == 0:
    raise RuntimeError("No matching CSV files found.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

plot_data = []

for split in ["train", "test"]:
    for metric in metrics:
        col = f"{split}_{metric}"
        for value in df[col]:
            plot_data.append({
                "Split": split.capitalize(),
                "Metric": metric.upper(),
                "Value": value
            })

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_df, x="Metric", y="Value", hue="Split")
plt.title("Model Performance Distribution")
plt.ylabel("Score")
plt.ylim(0, 1)

timestamp = time.strftime("%Y%m%d%H%M%S")
plot_name = f"{args.keyword}_boxplot_{timestamp}.pdf"
plot_path = os.path.join(args.output_dir, plot_name)

plt.savefig(plot_path)
plt.close()

print(f"Saved boxplot to {plot_path}")

