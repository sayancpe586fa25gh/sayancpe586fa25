import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from datasets import load_from_disk
from sayancpe586fa25 import deepl



# -------------------------------------------------
# Command line arguments
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.005)

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    args = parse_args()

    # ---------------------------------------------
    # Load dataset
    # ---------------------------------------------
    """
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        cache_dir="/data/CPE_487-587/imagenet-1k"
    )
    """
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    class_names = train_dataset.features["label"].names
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")

    # ---------------------------------------------
    # Select subset
    # ---------------------------------------------
    train_size = int(len(train_dataset) * args.train_ratio)
    val_size = int(len(val_dataset) * args.val_ratio)

    train_dataset = train_dataset.select(range(train_size))
    #val_dataset = val_dataset.select(range(val_size))
    # ---------------------------------------------
    # Build train class set (SAFE)
    # ---------------------------------------------
    train_classes = set(
    train_dataset[i]["label"] for i in range(len(train_dataset))
    )
    # ---------------------------------------------
    # Build validation subset (IMPORTANT FIX)
    # ---------------------------------------------
    val_indices = []
    
    for i in range(len(val_dataset)):
    
        label = val_dataset[i]["label"]
    
        if label in train_classes:
            val_indices.append(i)
    
        if len(val_indices) >= val_size:
            break
    
    val_dataset = val_dataset.select(val_indices)
    val_classes = set(val_dataset[i]["label"] for i in range(len(val_dataset)))
    
    print(f"Validation samples after filtering: {len(val_dataset)}")
    print(f"Unique train classes: {len(train_classes)}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # ---------------------------------------------
    # Save example training image
    # ---------------------------------------------
    example = train_dataset[0]
    image = example["image"]
    label_id = example["label"]

    full_label = class_names[label_id]
    primary_name = full_label.split(",")[0].strip()

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title(f"Train Image\nID {label_id}: {primary_name}")
    plt.axis("off")
    plt.savefig("train_sample_cnn.png")
    plt.close()

    # ---------------------------------------------
    # Save example validation image
    # ---------------------------------------------
    example = val_dataset[0]
    image = example["image"]
    label_id = example["label"]

    full_label = class_names[label_id]
    primary_name = full_label.split(",")[0].strip()

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title(f"Validation Image\nID {label_id}: {primary_name}")
    plt.axis("off")
    plt.savefig("val_sample.png")
    plt.close()

    # ---------------------------------------------
    # Image transforms
    # ---------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # ---------------------------------------------
    # Preprocessing functions
    # ---------------------------------------------
    def preprocess_train(examples):
        images = [train_transform(img.convert("RGB")) for img in examples["image"]]

        return {
            "pixel_values": images,
            "labels": examples["label"]
        }

    def preprocess_val(examples):
        images = [val_transform(img.convert("RGB")) for img in examples["image"]]

        return {
            "pixel_values": images,
            "labels": examples["label"]
        }

    #train_classes = set(train_dataset["label"])
    #train_classes = set(train_dataset[i]["label"] for i in range(len(train_dataset)))
    #val_dataset = val_dataset.filter(lambda x: x["label"] in train_classes, batched=False)
    train_dataset = train_dataset.with_transform(preprocess_train)
    #val_dataset = val_dataset.select(range(val_size))
    val_dataset = val_dataset.with_transform(preprocess_val)

    # ---------------------------------------------
    # Collate function
    # ---------------------------------------------
    def collate_fn(batch):

        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    # ---------------------------------------------
    # DataLoaders
    # ---------------------------------------------
    #train_classes = set(train_dataset["label"])
    #train_classes = set(train_dataset[i]["labels"] for i in range(len(train_dataset)))
    #val_dataset = val_dataset.filter(lambda x: x["label"] in train_classes)
    #val_dataset = val_dataset.select(range(val_size))
    #val_classes = set(val_dataset[i]["label"] for i in range(len(val_dataset)))

    print("Train classes:", len(train_classes))
    print("Val classes:", len(val_classes))
    print("Overlap:", len(train_classes & val_classes))
    train_loader = DataLoader(
        train_dataset,
        batch_size=192,
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=192,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    

    # ---------------------------------------------
    # Instantiate model
    # ---------------------------------------------
    model = deepl.ImageNetCNN(num_classes=num_classes)

    # ---------------------------------------------
    # Instantiate trainer
    # ---------------------------------------------
    trainer = deepl.CNNTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=args.epochs
    )
    trainer.train()
    # ---------------------------------------------
    # Training loop
    # ---------------------------------------------
    trainer.save(file_name="imagenet_model_3.onnx")
    metrics = trainer.evaluation()
    print(metrics)

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    main()
