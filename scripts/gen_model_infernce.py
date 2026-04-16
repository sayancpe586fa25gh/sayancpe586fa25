import argparse
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import cv2

from image_metrics import *
def diffusion_sampler(sess, timesteps=1000):
    """
    ONNX-based DDPM sampler
    (reduced steps for speed)
    """

    x = np.random.randn(1, 3, 64, 64).astype(np.float32)

    for t in reversed(range(timesteps)):

        t_arr = np.array([t], dtype=np.int64)

        pred_noise = sess.run(None, {"input": x, "t": t_arr})[0]

        # simple stable schedule
        beta = np.float32(0.02 * (t / timesteps))
        alpha = np.float32(1.0 - beta)

        x = (x - beta * pred_noise) / np.sqrt(alpha)

        # prevent explosion
        x = np.clip(x, -3, 3)

    return x
def generate_samples(onnx_path, model_type):
    sess = ort.InferenceSession(onnx_path)

    samples = []

    for _ in range(25):

        if model_type == "gan":
            z = np.random.randn(1, 128).astype(np.float32)
            out = sess.run(None, {"input": z})[0]

        elif model_type == "vae":
            x = np.random.randn(1, 3, 64, 64).astype(np.float32)
            out = sess.run(None, {"input": x})[0]

        elif model_type == "diffusion":
            out = diffusion_sampler(sess)   # 🔥 IMPORTANT

        samples.append(out[0])

    return samples


def save_samples_grid(samples, model_name, save_path):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))

    for i, ax in enumerate(axes.flatten()):
        img = samples[i]

        # Denormalize [-1,1] → [0,255]
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)

        ax.imshow(img)
        ax.axis("off")

    plt.suptitle(f"{model_name} Samples", fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()

    print(f"Saved {model_name} samples to {save_path}")

def evaluate(samples):
    scores = {k: [] for k in
              ["laplacian", "tenengrad", "freq", "std", "glcm"]}

    for img in samples:
        img = ((img + 1) * 127.5).clip(0,255).astype(np.uint8)
        img = img.transpose(1,2,0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scores["laplacian"].append(laplacian_variance(gray))
        scores["tenengrad"].append(tenengrad(gray))
        scores["freq"].append(high_freq_energy(gray))
        scores["std"].append(local_std(gray))
        scores["glcm"].append(glcm_contrast(gray))

    return scores



def plot_box(all_scores):
    metrics = ["laplacian", "tenengrad", "freq", "std", "glcm"]
    models = list(all_scores.keys())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, metric in enumerate(metrics):
        data = []

        for model in models:
            data.append(all_scores[model][metric])

        axes[i].boxplot(data, labels=models)
        axes[i].set_title(metric)
        axes[i].tick_params(axis='x', rotation=30)

    plt.suptitle("Sharpness Metrics Comparison Across Models", fontsize=16)
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae_path")
    parser.add_argument("--gan_path")
    parser.add_argument("--diffusion_path")

    args = parser.parse_args()

    all_scores = {}

    if args.vae_path:
        samples = generate_samples(args.vae_path, "vae")
        save_samples_grid(samples, "VAE", "vae_samples.png")
        all_scores["VAE"] = evaluate(samples)
    
    if args.gan_path:
        samples = generate_samples(args.gan_path, "gan")
        save_samples_grid(samples, "GAN", "gan_samples.png")
        all_scores["GAN"] = evaluate(samples)
    
    if args.diffusion_path:
        samples = generate_samples(args.diffusion_path, "diffusion")
        save_samples_grid(samples, "Diffusion", "diffusion_samples.png")
        all_scores["Diffusion"] = evaluate(samples)
        
    plot_box(all_scores)


if __name__ == "__main__":
    main()
