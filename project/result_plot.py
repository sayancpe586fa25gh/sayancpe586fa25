"""
Parse HD: values from a VAE run log, then save:
  - <op>.png : two subplots — (1) PDF/KDE density plot, (2) box plot — at 600 dpi

Usage:
    python plot_hd.py --log run_log_vae/test_5.log --op test5_results
"""

import argparse
import re
import sys
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HD log parser + PDF/PNG reporter")
    p.add_argument("--log", required=True, help="Path to the .log file")
    p.add_argument("--op",  required=True, help="Output file key (no extension)")
    p.add_argument("--key",  required=True, help="Plot Title key (Model names)")
    return p.parse_args()


# ── Log parsing ────────────────────────────────────────────────────────────────

def extract_hd_values(log_path: str) -> list[float]:
    """Return all numeric values after 'HD:' in the log file."""
    values = []
    #pattern = re.compile(r"| HD:\s*([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
    try:
        with open(log_path, "r") as f:
            for line in f:
                values.append(float(line))
                #for match in pattern.finditer(line):
                #    values.append(float(match.group(1)))
    except FileNotFoundError:
        sys.exit(f"[ERROR] Log file not found: {log_path}")
    if not values:
        sys.exit("[ERROR] No 'HD:' values found in the log file.")
    return values


# ── Summary stats ──────────────────────────────────────────────────────────────

def compute_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "Count":  len(a),
        "Min":    float(np.min(a)),
        "Max":    float(np.max(a)),
        "Mean":   float(np.mean(a)),
        "Median": float(np.median(a)),
        "Std":    float(np.std(a)),
        "Q1":     float(np.percentile(a, 25)),
        "Q3":     float(np.percentile(a, 75)),
        "IQR":    float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


# ── Combined PNG: PDF (density) + Box plot ─────────────────────────────────────

def save_combined_png(values: list[float], stats: dict, out_path: str, title: str):
    """Two subplots side-by-side: (1) probability density function, (2) box plot."""
    arr = np.array(values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # ── Subplot 1: Probability Density Function (KDE + histogram) ────────────
    n_bins = max(10, int(np.sqrt(len(arr))))
    ax1.hist(arr, bins=n_bins, density=True,
             color="#B5D4F4", edgecolor="#185FA5", linewidth=0.6,
             alpha=0.7, label="Histogram (density)")

    if len(arr) >= 4:
        kde = gaussian_kde(arr, bw_method="scott")
        x_range = np.linspace(arr.min() - 0.5 * arr.std(),
                               arr.max() + 0.5 * arr.std(), 400)
        ax1.plot(x_range, kde(x_range), color="#E24B4A", linewidth=2.0, label="KDE")

    # mark mean and median
    ax1.axvline(stats["Mean"],   color="#185FA5", linewidth=1.2,
                linestyle="--", label=f"Mean={stats['Mean']:.4f}")
    ax1.axvline(stats["Median"], color="#3B6D11", linewidth=1.2,
                linestyle=":",  label=f"Median={stats['Median']:.4f}")

    ax1.set_title("Probability Density Function", fontsize=11)
    ax1.set_xlabel("HD value", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.6)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    # ── Subplot 2: Box plot ───────────────────────────────────────────────────
    bp = ax2.boxplot(
        arr,
        vert=True,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="#E24B4A", linewidth=2),
        boxprops=dict(facecolor="#B5D4F4", color="#185FA5", linewidth=1.2),
        whiskerprops=dict(color="#185FA5", linewidth=1.2, linestyle="--"),
        capprops=dict(color="#185FA5", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="#EF9F27", markersize=5,
                        markeredgecolor="#BA7517", alpha=0.7),
    )
    # jittered individual points
    rng = np.random.default_rng(42)
    x_jitter = rng.uniform(0.82, 1.18, len(arr))
    ax2.scatter(x_jitter, arr, alpha=0.35, s=18, color="#185FA5", zorder=3)

    # annotate key stats on the box plot
    for label, val, ha in [
        (f"Q3={stats['Q3']:.4f}",     stats["Q3"],     "left"),
        (f"Med={stats['Median']:.4f}", stats["Median"], "left"),
        (f"Q1={stats['Q1']:.4f}",     stats["Q1"],     "left"),
    ]:
        ax2.annotate(label, xy=(1.28, val), fontsize=7.5,
                     color="#444441", va="center")

    ax2.set_title("Box Plot", fontsize=11)
    ax2.set_ylabel("HD value", fontsize=10)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["HD"])
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.5, 1.7)

    # shared footer with key stats
    footer = (f"n={stats['Count']}   mean={stats['Mean']:.4f}   "
              f"std={stats['Std']:.4f}   min={stats['Min']:.4f}   max={stats['Max']:.4f}")
    fig.text(0.5, -0.02, footer, ha="center", fontsize=8.5, color="#5F5E5A")

    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    #plt.close(fig)
    plt.show()
    print(f"[OK] Plot saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"[..] Parsing log: {args.log}")
    values = extract_hd_values(args.log)
    stats  = compute_stats(values)

    print(f"[..] Found {len(values)} HD values  (mean={stats['Mean']:.4f})")

    out_path = f"{args.op}.png"
    title = f"{args.key} HD Values — Distribution Analysis"
    save_combined_png(values, stats, out_path, title)

    print(f"\nDone!  PNG → {out_path}")


if __name__ == "__main__":
    main()
