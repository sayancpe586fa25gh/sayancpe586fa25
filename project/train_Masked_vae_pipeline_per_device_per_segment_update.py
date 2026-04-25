# ===============================
# === Pipeline ==================
# ===============================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from vae import MaskedVAE, MaskedVAETrainer

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

class SRAMSegmentPerIndexDataset(Dataset):
    def __init__(self, file, row, col, split="train", segment_size=128, multi_cycle=False, n_cycles_sample=3):
        """
        Fixed (row, col) segment → iterate over cycles only
        """
        self.file = file
        self.row = row
        self.col = col
        self.segment_size = segment_size
        self.split = split
        self.multi_cycle = multi_cycle
        self.n_cycles_sample = n_cycles_sample

        self.split_idx = {"train": (0,71), "val": (71,86), "test": (86,101)}

        self.data = np.load(file, mmap_mode="r")
        self.start, self.end = self.split_idx[split]

        self.cycles = list(range(self.start, self.end))

    def __len__(self):
        return len(self.cycles)

    def __getitem__(self, idx):
        cycle = self.cycles[idx]

        if self.multi_cycle and self.split == "train":
            cycles = np.random.choice(self.cycles, size=self.n_cycles_sample, replace=False)
            segment = np.mean(self.data[cycles, self.row, self.col:self.col+self.segment_size], axis=0)
        else:
            segment = self.data[cycle, self.row, self.col:self.col+self.segment_size]

        return torch.from_numpy(segment.astype(np.float32))

def main(data_root):

    # ---- collect files ----
    all_files = []
    for fam in os.listdir(data_root):
        fam_path = os.path.join(data_root, fam)
        if not os.path.isdir(fam_path):
            continue
        for f in os.listdir(fam_path):
            if f.endswith(".npy"):
                all_files.append(os.path.join(fam_path, f))

    print(f"Total SRAM devices: {len(all_files)}")

    device_id = get_best_gpu(strategy="memory")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # 🔁 PER DEVICE
    # =========================================================
    for each_file in all_files:
        print(f"\n=== Device: {each_file} ===")

        data = np.load(each_file, mmap_mode="r")
        rows, cols = data.shape[1], data.shape[2]
        segment_size = 256
        missing_len = 128
        # =====================================================
        # 🔁 PER SEGMENT
        # =====================================================
        for row in range(rows):
            for col in range(0, cols - segment_size + 1, segment_size):

                print(f"\n--- Segment (row={row}, col={col}) ---")

                # ✅ MASKED MODEL
                model = MaskedVAE(input_dim=segment_size*2, latent_dim=64).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                trainer = MaskedVAETrainer(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    beta=0.1,
                    missing_len=missing_len
                )

                # ---- datasets ----
                train_ds = SRAMSegmentPerIndexDataset(
                    each_file, row, col,
                    split="train",
                    segment_size=segment_size,
                    multi_cycle=True
                )

                val_ds = SRAMSegmentPerIndexDataset(
                    each_file, row, col,
                    split="val",
                    segment_size=segment_size
                )

                test_ds = SRAMSegmentPerIndexDataset(
                    each_file, row, col,
                    split="test",
                    segment_size=segment_size
                )

                # ---- loaders ----
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
                test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

                # ---- train ----
                trainer.fit(train_loader, val_loader, epochs=25)

                # ---- test (includes HD + accuracy already) ----
                test_loss = trainer.test(test_loader)
                print(f"Test loss: {test_loss:.4f}")

                # =====================================================
                # 🔥 EXTRA: Explicit batch-wise masked HD check
                # =====================================================
                with torch.no_grad():
                    for i, x in enumerate(test_loader):
                        x = x.to(device)

                        # ✅ apply SAME masking logic
                        x_masked, mask = trainer.mask_segment(x)

                        x_hat, _, _ = model(x_masked, mask)

                        real = (x > 0.5)
                        pred = (x_hat > 0.5)
                        missing = (mask == 0)

                        hd = ((real != pred) & missing).sum(dim=1).float()
                        total_bits = missing.sum(dim=1).float()

                        acc = 1 - (hd / total_bits)

                        print(f"Batch {i} | HD: {hd.mean().item():.2f} | Acc: {acc.mean().item():.4f}")

                        if i >= 10:
                            break

            # 🔴 keep for debugging, remove later
            #break
        break
# -------------------------------
# === Run =======================
# -------------------------------
if __name__ == "__main__":
    main("converted_npy")

