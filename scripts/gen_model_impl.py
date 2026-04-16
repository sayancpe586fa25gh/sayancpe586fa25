import argparse
import torch
import zipfile
import io
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

from sayancpe586fa25.deepl.gen_model import (
    VAE, GAN, DiffusionModel, GenModelTrainer
)

class CelebAZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        # IMPORTANT: open ZIP ONCE
        self.zf = zipfile.ZipFile(zip_path, 'r')

        self.image_names = sorted([
            n for n in self.zf.namelist()
            if n.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # SAFE read (no re-opening ZIP)
        with self.zf.open(self.image_names[idx]) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


def get_model(name):
    if name == "vae":
        return VAE()
    elif name == "gan":
        return GAN()
    elif name == "diffusion":
        return DiffusionModel()
    else:
        raise ValueError("Unknown model")

import subprocess

def get_best_gpu(strategy="memory"):
    if not torch.cuda.is_available():
        return None

    if strategy == "memory":
        free_mem = []
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            free_mem.append(free)
        return int(np.argmax(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        utils = [int(x) for x in result.stdout.strip().split("\n")]
        return int(np.argmin(utils))

    return 0

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True,
                        choices=["vae", "gan", "diffusion"])

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--onnx_dir", type=str, default="./gen_models_v2")

    parser.add_argument("--zip_path", type=str, default="/data/CPE_487-587/img_align_celeba.zip")

    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--data_fraction", type=float, default=1.0)

    args = parser.parse_args()

    # -------------------------
    # DEVICE
    # -------------------------
    gpu_id = get_best_gpu("memory")

    device = (
        torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available() and gpu_id is not None
        else torch.device("cpu")
    )

    print(f"Using device: {device}")

    # -------------------------
    # TRANSFORMS (CELEBA SAFE)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # -------------------------
    # DATASET
    # -------------------------
    dataset = CelebAZipDataset(args.zip_path, transform)

    if args.data_fraction < 1.0:
        idx = np.random.permutation(len(dataset))
        dataset = Subset(dataset, idx[:int(len(idx) * args.data_fraction)])

    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # -------------------------
    # MODEL
    # -------------------------
    model = get_model(args.model).to(device)

    trainer = GenModelTrainer(
        model=model,
        dataloader=train_loader,
        device=device,
        save_every=args.save_every,
        onnx_dir=args.onnx_dir
    )

    # -------------------------
    # OPTIMIZERS (IMPORTANT)
    # -------------------------
    if args.model == "vae":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        trainer.train(
            epochs=args.epochs,
            model_type="vae",
            optimizer=optimizer
        )

    elif args.model == "gan":
        opt_G = torch.optim.Adam(
            model.generator.parameters(),
            lr=2e-4, betas=(0.5, 0.999)
        )

        opt_D = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=2e-4, betas=(0.5, 0.999)
        )

        trainer.train(
            epochs=args.epochs,
            model_type="gan",
            opt_G=opt_G,
            opt_D=opt_D
        )

    elif args.model == "diffusion":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        trainer.train(
            epochs=args.epochs,
            model_type="diffusion",
            optimizer=optimizer
        )


if __name__ == "__main__":
    main()
