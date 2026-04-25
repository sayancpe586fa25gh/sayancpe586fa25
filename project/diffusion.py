import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class DiffusionScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T = T
        self.device = device

        # ---- schedules ----
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # ---- useful precomputations (IMPORTANT for stability) ----
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def to(self, device):
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

        self.sqrt_alpha = self.sqrt_alpha.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)

def forward_diffusion(x0, t, scheduler):
    """
    x0: (B, D)
    t: (B,)
    """
    noise = torch.randn_like(x0)

    sqrt_alpha_bar = scheduler.sqrt_alpha_bar[t].unsqueeze(1)
    sqrt_one_minus = scheduler.sqrt_one_minus_alpha_bar[t].unsqueeze(1)

    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    return xt, noise

class DiffusionMLP(nn.Module):
    def __init__(self, input_dim=256, time_dim=128):
        super().__init__()

        self.time_embed = nn.Embedding(1000, time_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x = torch.cat([x, t_emb], dim=1)
        return self.net(x)

def diffusion_loss(model, x0, scheduler):
    B = x0.size(0)

    t = torch.randint(0, scheduler.T, (B,), device=x0.device)

    xt, noise = forward_diffusion(x0, t, scheduler)

    noise_pred = model(xt, t)

    return F.mse_loss(noise_pred, noise)

class DiffusionTrainer:
    def __init__(self, model, optimizer, scheduler, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    # -------------------
    # Train
    # -------------------
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for x in tqdm(dataloader, desc="Train"):
            x = x.to(self.device).float()

            loss = diffusion_loss(self.model, x, self.scheduler)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # -------------------
    # Validation
    # -------------------
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()
                loss = diffusion_loss(self.model, x, self.scheduler)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    # -------------------
    # Sampling
    # -------------------
    def sample(self, num_samples, dim):
        self.model.eval()

        x = torch.randn(num_samples, dim).to(self.device)

        for t in reversed(range(self.scheduler.T)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x, t_tensor)

            alpha = self.scheduler.alpha[t]
            alpha_bar = self.scheduler.alpha_bar[t]
            beta = self.scheduler.beta[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
            )

            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)

        return torch.sigmoid(x)

    # -------------------
    # Test (HD metric)
    # -------------------
    def test(self, dataloader, segment_size):
        self.model.eval()

        total_hd = 0
        total_bits = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()
                B = x.size(0)

                x_hat = self.sample(B, segment_size)

                real = (x > 0.5).int()
                pred = (x_hat > 0.5).int()

                hd = (real != pred).sum(dim=1).float()

                total_hd += hd.sum().item()
                total_bits += B * segment_size

        acc = 1 - total_hd / total_bits

        print(f"Total HD: {total_hd}")
        print(f"Bit Accuracy: {acc:.4f}")
        print(f"Avg HD per segment: {total_hd / (total_bits / segment_size):.2f}")

        return acc

    # -------------------
    # Fit
    # -------------------
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

def mask_segment(x, missing_len=64):
    B, D = x.shape
    mask = torch.ones_like(x)

    start = torch.randint(0, D - missing_len, (B,), device=x.device)

    for i in range(B):
        mask[i, start[i]:start[i] + missing_len] = 0.0

    x_masked = x * mask
    return x_masked, mask

class MaskedDiffusionMLP(nn.Module):
    def __init__(self, input_dim=256, time_dim=128):
        super().__init__()

        self.input_dim = input_dim

        self.time_embed = nn.Embedding(1000, time_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + time_dim, 512),  # x + mask + t
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)  # predict noise
        )

    def forward(self, xt, mask, t):
        t_emb = self.time_embed(t)
        x = torch.cat([xt, mask, t_emb], dim=1)
        return self.net(x)

def masked_diffusion_loss(model, x0, scheduler, missing_len):
    B = x0.size(0)

    # ---- create mask ----
    x_masked, mask = mask_segment(x0, missing_len)

    # ---- sample timestep ----
    t = torch.randint(0, scheduler.T, (B,), device=x0.device)

    # 🔥 IMPORTANT: diffuse ORIGINAL x0
    xt, noise = forward_diffusion(x0, t, scheduler)

    # ---- predict noise ----
    noise_pred = model(xt, mask, t)

    # ---- compute loss ONLY on missing region ----
    missing = (mask == 0)

    denom = missing.sum()
    if denom == 0:
        return ((noise_pred - noise) ** 2).mean()

    loss = ((noise_pred - noise) ** 2 * missing).sum() / denom

    return loss

def masked_diffusion_sample(model, scheduler, x_masked, mask, device):
    model.eval()

    x = torch.randn_like(x_masked).to(device)

    for t in reversed(range(scheduler.T)):
        t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        noise_pred = model(x, mask, t_tensor)

        alpha = scheduler.alpha[t]
        alpha_bar = scheduler.alpha_bar[t]
        beta = scheduler.beta[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
        )

        if t > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)

    # 🔥 CRITICAL: fuse known + generated
    x_final = x * (1 - mask) + x_masked

    return torch.sigmoid(x_final)

class MaskedDiffusionTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device="cuda",
        missing_len=64,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.missing_len = missing_len

    # -----------------------------
    # Train
    # -----------------------------
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for x in tqdm(dataloader, desc="Diffusion Train"):
            x = x.to(self.device).float()

            loss = masked_diffusion_loss(
                self.model,
                x,
                self.scheduler,
                self.missing_len
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # -----------------------------
    # Validate
    # -----------------------------
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()

                loss = masked_diffusion_loss(
                    self.model,
                    x,
                    self.scheduler,
                    self.missing_len
                )

                total_loss += loss.item()

        return total_loss / len(dataloader)

    # -----------------------------
    # Test (HD + Accuracy)
    # -----------------------------
    def test(self, dataloader):
        self.model.eval()

        total_loss = 0
        total_hd = 0
        total_bits = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()

                # ---- mask ----
                x_masked, mask = mask_segment(x, self.missing_len)

                # ---- generate ----
                x_hat = masked_diffusion_sample(
                    self.model,
                    self.scheduler,
                    x_masked,
                    mask,
                    self.device
                )

                # ---- loss ----
                loss = masked_diffusion_loss(
                    self.model,
                    x,
                    self.scheduler,
                    self.missing_len
                )
                total_loss += loss.item()

                # ---- HD ----
                real = (x > 0.5)
                pred = (x_hat > 0.5)
                missing = (mask == 0)

                hd = ((real != pred) & missing).sum().item()
                total_hd += hd
                total_bits += missing.sum().item()

        acc = 1 - total_hd / total_bits if total_bits > 0 else 0

        print(f"Missing Bits: {total_bits}")
        print(f"Total HD: {total_hd}")
        print(f"Bit Accuracy: {acc:.4f}")

        return total_loss / len(dataloader)

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )
