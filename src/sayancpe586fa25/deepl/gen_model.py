import os
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        # ======================
        # Encoder (64x64 -> 16x16)
        # ======================
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Flatten()
        )

        self.enc_out_dim = 128 * 8 * 8

        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # ======================
        # Decoder
        # ======================
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 64x64
            nn.Tanh()
        )

    # ======================
    # Reparameterization
    # ======================
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ======================
    # Forward (TRAINING)
    # ======================
    def forward(self, x):
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_dec(z).view(-1, 128, 8, 8)
        recon = self.decoder(h_dec)

        return recon, mu, logvar

    # ======================
    # Loss (STABLE VERSION)
    # ======================
    def loss_fn(self, recon, x, mu, logvar):
        recon_loss = F.l1_loss(recon, x, reduction='mean')

        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + 0.1 * kl  # KL weight stabilizer

    # ======================
    # ONNX SAFE INFERENCE
    # ======================
    def encode_decode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_dec(z).view(-1, 128, 8, 8)
        return self.decoder(h_dec)

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            # z: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),         # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),         # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),          # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),            # 64x64
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # 64x64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class GAN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.generator = DCGANGenerator(latent_dim)
        self.discriminator = DCGANDiscriminator()


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)

        # time embedding (VERY IMPORTANT)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x, t):
        """
        x: noisy image (B,3,64,64)
        t: timestep (B,)
        """

        # normalize timestep
        t = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t).view(-1, 64, 1, 1)

        h = F.relu(self.conv1(x))
        h = h + t_emb  # inject time info

        h = F.relu(self.conv2(h))
        out = self.conv3(h)

        return out

class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()

        self.model = SimpleUNet()
        self.timesteps = timesteps

        # -----------------------
        # Noise schedule (DDPM)
        # -----------------------
        beta = torch.linspace(1e-4, 0.02, timesteps)

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # register buffers (so ONNX + CUDA safe)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    # trainer-compatible forward
    def forward(self, x, t):
        return self.model(x, t)

    # -----------------------
    # forward diffusion
    # -----------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

class DiffusionWrapper(nn.Module):
    def __init__(self, diffusion_model, timestep=0):
        super().__init__()
        self.model = diffusion_model
        self.timestep = timestep

    def forward(self, x, t):
        return self.model(x, t)

class GenModelTrainer:
    def __init__(self, model, dataloader, device,
                 save_every=5, onnx_dir="./onnx_models"):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.save_every = save_every
        self.onnx_dir = onnx_dir

        os.makedirs(onnx_dir, exist_ok=True)

    # =========================================================
    # MAIN TRAIN LOOP
    # =========================================================
    def train(self, epochs, model_type, optimizer=None,
              opt_G=None, opt_D=None):

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch in self.dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                batch = batch.to(self.device)

                # =========================
                # VAE
                # =========================
                if model_type == "vae":
                    recon, mu, logvar = self.model(batch)
                    loss = self.model.loss_fn(recon, batch, mu, logvar)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # =========================
                # GAN (DCGAN)
                # =========================
                elif model_type == "gan":
                    d_loss, g_loss = self._train_gan(batch, opt_G, opt_D)
                    loss = d_loss + g_loss

                # =========================
                # DIFFUSION (DDPM)
                # =========================
                elif model_type == "diffusion":
                    loss = self._train_diffusion(batch, optimizer)

                else:
                    raise ValueError("Unknown model type")

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

            # =========================
            # ONNX EXPORT
            # =========================
            if (epoch + 1) % self.save_every == 0:
                self.export_onnx((epoch+1), model_type)

    # =========================================================
    # GAN TRAIN STEP (DCGAN)
    # =========================================================
    def _train_gan(self, real, opt_G, opt_D):

        G = self.model.generator
        D = self.model.discriminator

        bs = real.size(0)

        real = real.to(self.device)

        real_labels = torch.ones(bs, 1, device=self.device)
        fake_labels = torch.zeros(bs, 1, device=self.device)

        # -------------------
        # Train Discriminator
        # -------------------
        z = torch.randn(bs, self.model.latent_dim, device=self.device)
        fake = G(z)

        d_real = D(real)
        d_fake = D(fake.detach())

        d_loss = (
            F.binary_cross_entropy(d_real, real_labels) +
            F.binary_cross_entropy(d_fake, fake_labels)
        )

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # -------------------
        # Train Generator
        # -------------------
        z = torch.randn(bs, self.model.latent_dim, device=self.device)
        fake = G(z)

        g_loss = F.binary_cross_entropy(D(fake), real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        return d_loss, g_loss

    # =========================================================
    # DIFFUSION TRAIN STEP (DDPM)
    # =========================================================
    def _train_diffusion(self, x0, optimizer):

        model = self.model

        t = torch.randint(
            0, model.timesteps,
            (x0.size(0),),
            device=self.device
        )

        noise = torch.randn_like(x0)

        alpha_bar = model.alpha_bar[t].view(-1, 1, 1, 1)
        noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        pred = model(noisy, t)

        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # =========================================================
    # ONNX EXPORT (SAFE FOR ALL MODELS)
    # =========================================================
    def export_onnx(self, epoch, model_type):

        path = os.path.join(
            self.onnx_dir,
            f"{model_type}_epoch_{epoch}.onnx"
        )

        # -------------------------
        # VAE
        # -------------------------
        if model_type == "vae":
            model_to_export = self.model
            dummy = torch.randn(1, 3, 64, 64).to(self.device)
            input_names = ["input"]
        # -------------------------
        # GAN (generator only)
        # -------------------------
        elif model_type == "gan":
            model_to_export = self.model.generator
            dummy = torch.randn(1, self.model.latent_dim).to(self.device)
            input_names = ["input"]
        # -------------------------
        # DIFFUSION
        # -------------------------
        elif model_type == "diffusion":
            model_to_export = DiffusionWrapper(self.model)
            dummy = torch.randn(1, 3, 64, 64).to(self.device)
            t_dummy = torch.zeros(1, dtype=torch.long).to(self.device)

            dummy = (dummy, t_dummy)
            input_names = ["input", "t"]
        else:
            raise ValueError("Unknown model type")

        torch.onnx.export(
            model_to_export,
            dummy,
            path,
            input_names=input_names,
            output_names=["output"],
            opset_version=11
        )

        print(f"[ONNX SAVED] {path}")
