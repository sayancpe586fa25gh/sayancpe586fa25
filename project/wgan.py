import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=64, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()   # output in [0,1]
        )

    def forward(self, z):
        return self.net(z)

class WGANCritic(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)   # no sigmoid
        )

    def forward(self, x):
        return self.net(x).view(-1)

class WGANTrainer:
    def __init__(
        self,
        generator,
        critic,
        g_optimizer,
        d_optimizer,
        latent_dim=64,
        device="cuda",
        lambda_gp=10,
        critic_steps=5
    ):
        self.G = generator.to(device)
        self.D = critic.to(device)

        self.g_opt = g_optimizer
        self.d_opt = d_optimizer

        self.latent_dim = latent_dim
        self.device = device

        self.lambda_gp = lambda_gp
        self.critic_steps = critic_steps

    def gradient_penalty(self, real, fake):
        B = real.size(0)

        alpha = torch.rand(B, 1, device=self.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        d_interpolated = self.D(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gp
    def train_epoch(self, dataloader):
        self.G.train()
        self.D.train()

        total_g_loss = 0
        total_d_loss = 0

        for real in tqdm(dataloader, desc="WGAN Train"):
            real = real.to(self.device).float()
            B = real.size(0)

            # =========================
            # Train Critic multiple times
            # =========================
            for _ in range(self.critic_steps):
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake = self.G(z).detach()

                d_real = self.D(real)
                d_fake = self.D(fake)

                gp = self.gradient_penalty(real, fake)

                d_loss = -(d_real.mean() - d_fake.mean()) + self.lambda_gp * gp

                self.d_opt.zero_grad()
                d_loss.backward()
                self.d_opt.step()

                total_d_loss += d_loss.item()

            # =========================
            # Train Generator
            # =========================
            z = torch.randn(B, self.latent_dim, device=self.device)
            fake = self.G(z)

            g_loss = -self.D(fake).mean()

            self.g_opt.zero_grad()
            g_loss.backward()
            self.g_opt.step()

            total_g_loss += g_loss.item()

        return total_g_loss / len(dataloader), total_d_loss / len(dataloader)
    def validate(self, dataloader):
        self.G.eval()
        self.D.eval()

        total_d = 0

        with torch.no_grad():
            for real in dataloader:
                real = real.to(self.device).float()
                B = real.size(0)

                z = torch.randn(B, self.latent_dim, device=self.device)
                fake = self.G(z)

                d_real = self.D(real)
                d_fake = self.D(fake)

                total_d += (d_real.mean() - d_fake.mean()).item()

        return total_d / len(dataloader)
    def sample(self, n_samples):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.G(z)
        return samples

class MaskedWGANGenerator(nn.Module):
    def __init__(self, input_dim=256, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # output bits
        )

    def forward(self, x_masked, mask):
        x = torch.cat([x_masked, mask], dim=1)  # (B, 256)
        return self.net(x)

class MaskedWGANCritic(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # no sigmoid
        )

    def forward(self, x, mask):
        x_in = torch.cat([x, mask], dim=1)  # (B, 256)
        return self.net(x_in).view(-1)

class MaskedWGANTrainer:
    def __init__(
        self,
        input_dim,
        output_dim,
        generator,
        critic,
        g_optimizer,
        d_optimizer,
        device="cuda",
        lambda_gp=10,
        critic_steps=5,
        missing_len=64
    ):
        self.input_dim = input_dim
        self.G = generator.to(device)
        self.D = critic.to(device)
        self.output_dim = output_dim
        self.g_opt = g_optimizer
        self.d_opt = d_optimizer

        self.device = device
        self.lambda_gp = lambda_gp
        self.critic_steps = critic_steps
        self.missing_len = missing_len

    def mask_segment(self, x):
        B, D = x.shape
        mask = torch.ones_like(x)

        start = torch.randint(0, D - self.missing_len, (B,), device=x.device)

        for i in range(B):
            mask[i, start[i]:start[i]+self.missing_len] = 0.0

        x_masked = x * mask

        return x_masked, mask

    def gradient_penalty(self, real, fake, mask):
        B = real.size(0)

        alpha = torch.rand(B, 1, device=self.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        d_interpolated = self.D(interpolated, mask)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gp

    def train_epoch(self, dataloader):
        self.G.train()
        self.D.train()

        total_g_loss = 0
        total_d_loss = 0

        for real in tqdm(dataloader, desc="Masked WGAN Train"):
            real = real.to(self.device).float()
            B = real.size(0)

            # 🔷 create mask
            x_masked, mask = self.mask_segment(real)

            # =========================
            # Train Critic
            # =========================
            for _ in range(self.critic_steps):
                fake = self.G(x_masked, mask).detach()

                d_real = self.D(real, mask)
                d_fake = self.D(fake, mask)

                gp = self.gradient_penalty(real, fake, mask)

                d_loss = -(d_real.mean() - d_fake.mean()) + self.lambda_gp * gp

                self.d_opt.zero_grad()
                d_loss.backward()
                self.d_opt.step()

                total_d_loss += d_loss.item()

            # =========================
            # Train Generator
            # =========================
            fake = self.G(x_masked, mask)

            g_loss = -self.D(fake, mask).mean()

            self.g_opt.zero_grad()
            g_loss.backward()
            self.g_opt.step()

            total_g_loss += g_loss.item()

        return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

    def test(self, dataloader):
        self.G.eval()
        total_hd = 0
        total_bits = 0

        with torch.no_grad():
            for real in dataloader:
                real = real.to(self.device).float()

                x_masked, mask = self.mask_segment(real)

                fake = self.G(x_masked, mask)

                real_bits = (real > 0.5).int()
                fake_bits = (fake > 0.5).int()

                missing = (mask == 0)

                hd = ((real_bits != fake_bits) & missing).sum().item()
                total_hd += hd
                total_bits += missing.sum().item()

        acc = 1 - total_hd / total_bits if total_bits > 0 else 0

        print(f"Missing Bits: {total_bits}")
        print(f"Total HD: {total_hd}")
        print(f"Bit Accuracy: {acc:.4f}")

        return acc
    def sample(self, x_masked, mask):
        """
        Generate samples from the generator given masked input.
        """
        self.G.eval()
        with torch.no_grad():
            fake = self.G(x_masked, mask)
        return fake

    def validate(self, dataloader):
        self.G.eval()
        total_hd = 0
        total_bits = 0
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device).float()
                x_masked, mask = self.mask_segment(x)
                fake = self.G(x_masked, mask)

                real_bits = (x > 0.5).int()
                fake_bits = (fake > 0.5).int()
                missing = (mask == 0)

                hd = ((real_bits != fake_bits) & missing).sum().item()
                total_hd += hd
                total_bits += missing.sum().item()

        acc = 1 - total_hd / total_bits if total_bits > 0 else 0
        return acc
