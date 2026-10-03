"""
Pseudo-Absorbance Demo
=======================
Takes raw sensor data (or the plan's simulated example data),
computes pseudo-absorbance per channel, and plots both side by side:

    LEFT:  raw B channel vs concentration (curved)
    RIGHT: A_B = log10(B_blank / B_sample) vs concentration (straight)

This is the visual proof for Slide 17 (model pipeline — preprocessing gap).

The point:
    One line of code — np.log10(blank/sample) — transforms a curved
    relationship into a straight one. After this, Ridge regression
    handles the data without needing polynomial terms or Random Forest
    to learn the curve shape. This is what every spectrophotometer does.

Usage:
    python pseudo_absorbance_demo.py              (uses built-in example data)
    python pseudo_absorbance_demo.py data.csv      (uses your CSV file)

CSV format expected:
    concentration_mg_L, R, G, B, Clear
    0, 8200, 9150, 10800, 32100
    20, 7850, 8600, 10100, 30500
    ...
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # works without display server
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os


# =========================================================
# PLAN'S SIMULATED EXAMPLE DATA
# From the research plan's reference table (marked "Example")
# =========================================================
EXAMPLE_DATA = {
    "concentration": [0, 20, 40, 60, 80, 100, 120, 150, 180, 220, 260, 300],
    "R":     [8200, 7850, 7500, 7150, 6800, 6410, 6050, 5580, 5120, 4600, 4100, 4160],
    "G":     [9150, 8700, 8250, 7800, 7200, 6590, 6100, 5500, 4900, 4350, 3850, 3860],
    "B":     [10800, 10200, 9650, 9100, 8450, 7720, 7100, 6350, 5650, 5000, 4500, 4720],
    "Clear": [32100, 30500, 28900, 27200, 25400, 25600, 22800, 20800, 18900, 17200, 15700, 19400],
}


def load_csv(filepath):
    """Load a CSV with columns: concentration_mg_L, R, G, B, Clear"""
    import csv
    data = {"concentration": [], "R": [], "G": [], "B": [], "Clear": []}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["concentration"].append(float(row["concentration_mg_L"]))
            data["R"].append(float(row["R"]))
            data["G"].append(float(row["G"]))
            data["B"].append(float(row["B"]))
            data["Clear"].append(float(row["Clear"]))
    return data


def compute_pseudo_absorbance(data, channel="B"):
    """Compute A = log10(blank / sample) for one channel."""
    values = np.array(data[channel], dtype=float)
    blank = values[0]  # first entry assumed to be 0 mg/L

    # avoid log(0) or log(negative)
    ratios = blank / np.maximum(values[1:], 1)
    A = np.log10(ratios)

    return A


def plot_comparison(data, channel="B", save_path="pseudo_absorbance_comparison.png"):
    """
    Plot raw channel vs concentration AND pseudo-absorbance vs concentration
    side by side.
    """
    conc = np.array(data["concentration"])
    raw = np.array(data[channel], dtype=float)

    # Pseudo-absorbance (skip blank for A calculation)
    conc_no_blank = conc[1:]
    A = compute_pseudo_absorbance(data, channel)

    # Linear fits
    slope_raw, intercept_raw, r_raw, _, _ = stats.linregress(conc, raw)
    slope_A, intercept_A, r_A, _, _ = stats.linregress(conc_no_blank, A)
    r2_raw = r_raw ** 2
    r2_A = r_A ** 2

    # Dark theme matching the presentation style
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0a0f1a")

    # LEFT: raw channel
    ax1.set_facecolor("#0f1525")
    ax1.scatter(conc, raw, color="#FF6B6B", s=60, zorder=5, edgecolors="white", linewidth=0.5)
    fit_x = np.linspace(conc.min(), conc.max(), 100)
    ax1.plot(fit_x, slope_raw * fit_x + intercept_raw, "--", color="#FF6B6B", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Concentration (mg/L)", fontsize=11, color="white")
    ax1.set_ylabel(f"Raw {channel} channel (counts)", fontsize=11, color="white")
    ax1.set_title(f"Raw {channel} vs Concentration", fontsize=13, fontweight="bold", color="white")
    ax1.text(0.95, 0.95, f"R² = {r2_raw:.3f}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=12, color="#FF6B6B", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#FF6B6B", alpha=0.8))
    ax1.text(0.5, 0.05, "model has to learn this curve",
             transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=10, color="#FF6B6B", style="italic")

    # RIGHT: pseudo-absorbance
    ax2.set_facecolor("#0f1525")
    ax2.scatter(conc_no_blank, A, color="#4ADE80", s=60, zorder=5, edgecolors="white", linewidth=0.5)
    fit_x2 = np.linspace(conc_no_blank.min(), conc_no_blank.max(), 100)
    ax2.plot(fit_x2, slope_A * fit_x2 + intercept_A, "--", color="#4ADE80", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Concentration (mg/L)", fontsize=11, color="white")
    ax2.set_ylabel(f"A_{channel} = log₁₀(blank / sample)", fontsize=11, color="white")
    ax2.set_title(f"Pseudo-Absorbance A_{channel} vs Concentration", fontsize=13, fontweight="bold", color="white")
    ax2.text(0.95, 0.95, f"R² = {r2_A:.3f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=12, color="#4ADE80", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#4ADE80", alpha=0.8))
    ax2.text(0.5, 0.05, "Ridge regression handles this directly",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=10, color="#4ADE80", style="italic")

    # Arrow between plots
    fig.text(0.50, 0.50, "→", fontsize=30, ha="center", va="center",
             color="#FFD700", fontweight="bold")
    fig.text(0.50, 0.42, "A = log₁₀(blank/sample)", fontsize=9, ha="center", va="center",
             color="#FFD700")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("One line of code. Curved becomes straight.",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # Print summary
    print()
    print(f"  Raw {channel} linear fit:         R² = {r2_raw:.4f}")
    print(f"  Pseudo-absorbance A_{channel} fit: R² = {r2_A:.4f}")
    print(f"  Improvement:                  +{(r2_A - r2_raw):.4f}")
    if r2_A > r2_raw:
        print(f"  → Log transform improved linearity.")
    else:
        print(f"  → Log transform did not improve linearity for this data.")
        print(f"     (This can happen if the raw data is already near-linear")
        print(f"      or if the concentration range is narrow.)")


def plot_all_channels(data, save_path="all_channels_absorbance.png"):
    """Plot pseudo-absorbance for R, G, B, Clear to compare sensitivity."""
    conc = np.array(data["concentration"])
    conc_no_blank = conc[1:]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("#0a0f1a")
    ax.set_facecolor("#0f1525")

    colors = {"R": "#FF6B6B", "G": "#4ADE80", "B": "#60A5FA", "Clear": "#A78BFA"}
    r2_values = {}

    for ch in ["R", "G", "B", "Clear"]:
        A = compute_pseudo_absorbance(data, ch)
        slope, intercept, r, _, _ = stats.linregress(conc_no_blank, A)
        r2 = r ** 2
        r2_values[ch] = r2

        ax.scatter(conc_no_blank, A, color=colors[ch], s=40, alpha=0.7, label=f"A_{ch} (R²={r2:.3f})")
        fit_x = np.linspace(conc_no_blank.min(), conc_no_blank.max(), 100)
        ax.plot(fit_x, slope * fit_x + intercept, "--", color=colors[ch], alpha=0.5)

    ax.set_xlabel("Concentration (mg/L)", fontsize=11, color="white")
    ax.set_ylabel("Pseudo-absorbance", fontsize=11, color="white")
    ax.set_title("Pseudo-absorbance per channel — which is most linear?",
                 fontsize=13, fontweight="bold", color="white")
    ax.legend(loc="upper left", fontsize=9)

    # annotate the winner
    best_ch = max(r2_values, key=r2_values.get)
    ax.text(0.95, 0.05, f"Most linear: A_{best_ch} (R² = {r2_values[best_ch]:.4f})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color=colors[best_ch], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor=colors[best_ch], alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    print(f"\nChannel R² ranking:")
    for ch, r2 in sorted(r2_values.items(), key=lambda x: -x[1]):
        marker = " ← best" if ch == best_ch else ""
        print(f"  A_{ch:>5s}: R² = {r2:.4f}{marker}")
    plt.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Loading data from: {sys.argv[1]}")
        data = load_csv(sys.argv[1])
    else:
        print("Using the plan's simulated example data.")
        print("(Pass a CSV path as argument to use real data.)")
        print()
        data = EXAMPLE_DATA

    # Main comparison plot (for slide 17)
    plot_comparison(data, channel="B", save_path="pseudo_absorbance_comparison.png")

    # All channels comparison (bonus — helps decide G vs B)
    plot_all_channels(data, save_path="all_channels_absorbance.png")

    print()
    print("NOTE: If using simulated data, these R² values are illustrative.")
    print("      Run again with real Phase 1 data for actual results.")
