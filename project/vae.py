import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=32, hidden_dims=[512, 256]):
        super().__init__()

        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # ----- Decoder -----
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        logits = self.decoder(z)
        return torch.sigmoid(logits)  # Bernoulli output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, n, device):
        z = torch.randn(n, self.fc_mu.out_features).to(device)
        samples = self.decode(z)
        return (samples > 0.5).float()  # binarized

import torch
import torch.nn.functional as F
from tqdm import tqdm


class VAETrainer:
    def __init__(
        self,
        model,
        optimizer,
        device="cuda",
        beta=1.0,
        kl_anneal_epochs=10,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.kl_anneal_epochs = kl_anneal_epochs

    def loss_fn(self, x, x_hat, mu, logvar, epoch):
        # Reconstruction loss (binary)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # KL annealing
        kl_weight = min(1.0, epoch / self.kl_anneal_epochs)

        total_loss = recon_loss + self.beta * kl_weight * kl

        return total_loss, recon_loss, kl

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        for x in tqdm(dataloader, desc="Train"):
            x = x.to(self.device).float()

            self.optimizer.zero_grad()
            x_hat, mu, logvar = self.model(x)

            loss, _, _ = self.loss_fn(x, x_hat, mu, logvar, epoch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()
                x_hat, mu, logvar = self.model(x)

                loss, _, _ = self.loss_fn(x, x_hat, mu, logvar, epoch)
                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    def test(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()
                x_hat, mu, logvar = self.model(x)

                loss, _, _ = self.loss_fn(x, x_hat, mu, logvar, epoch=1)
                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )



class MaskedVAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64, output_dim=256):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # since bits in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_masked, mask):
        # concatenate input
        x_in = torch.cat([x_masked, mask], dim=1)  # (B, 512)

        mu, logvar = self.encode(x_in)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar


class MaskedVAETrainer:
    def __init__(
        self,
        model,
        optimizer,
        device="cuda",
        beta=0.1,                    # ✅ FIXED KL strength
        kl_anneal_epochs=10,
        missing_len=64,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.kl_anneal_epochs = kl_anneal_epochs
        self.missing_len = missing_len

    # -----------------------------
    # ✅ VECTORIZE MASK
    # -----------------------------
    def mask_segment(self, x):
        B, D = x.shape

        start = torch.randint(0, D - self.missing_len, (B,), device=x.device)

        idx = torch.arange(D, device=x.device).unsqueeze(0)   # (1, D)
        start = start.unsqueeze(1)                            # (B, 1)

        mask = ((idx < start) | (idx >= start + self.missing_len)).float()

        masked_x = x * mask
        return masked_x, mask

    # -----------------------------
    # ✅ FIXED LOSS (KL scaling)
    # -----------------------------
    def loss_fn(self, x, x_hat, mask, mu, logvar, epoch):

        # reconstruction (only missing)
        recon = F.binary_cross_entropy(x_hat, x, reduction="none")
        missing = (1 - mask)

        denom = missing.sum()
        if denom == 0:
            recon_loss = recon.mean()
        else:
            recon_loss = (recon * missing).sum() / denom

        # ✅ KL properly scaled
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # annealing
        kl_weight = min(1.0, epoch / self.kl_anneal_epochs)

        total_loss = recon_loss + self.beta * kl_weight * kl

        return total_loss, recon_loss, kl

    # -----------------------------
    # TRAIN
    # -----------------------------
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        for x in tqdm(dataloader, desc="Train"):
            x = x.to(self.device).float()

            x_masked, mask = self.mask_segment(x)

            x_hat, mu, logvar = self.model(x_masked, mask)

            loss, _, _ = self.loss_fn(x, x_hat, mask, mu, logvar, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # ✅ FIX normalization
        return total_loss / len(dataloader.dataset)

    # -----------------------------
    # VALIDATE
    # -----------------------------
    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()

                x_masked, mask = self.mask_segment(x)
                x_hat, mu, logvar = self.model(x_masked, mask)

                loss, _, _ = self.loss_fn(x, x_hat, mask, mu, logvar, epoch)
                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    # -----------------------------
    # TEST (HD + Accuracy)
    # -----------------------------
    def test(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_hd = 0
        total_bits = 0

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()

                x_masked, mask = self.mask_segment(x)
                x_hat, mu, logvar = self.model(x_masked, mask)

                loss, _, _ = self.loss_fn(x, x_hat, mask, mu, logvar, epoch=1)
                total_loss += loss.item()

                real = (x > 0.5)
                pred = (x_hat > 0.5)

                missing = (mask == 0)

                hd = ((real != pred) & missing).sum().item()
                total_hd += hd
                total_bits += missing.sum().item()

        acc = 1 - total_hd / total_bits if total_bits > 0 else 0

        print(f"Test HD: {total_hd}")
        print(f"Missing Bits: {total_bits}")
        print(f"Bit Accuracy: {acc:.4f}")
        print(f"Avg HD per segment: {total_hd / len(dataloader.dataset):.2f}")

        return total_loss / len(dataloader.dataset)

    # -----------------------------
    # FIT
    # -----------------------------
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
