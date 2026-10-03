"""
Residual Analysis
==================
Takes model predictions and actual values, plots residuals,
and checks whether the residual pattern is random (model is sufficient)
or systematic (something is missing — usually the log transform).

Mentioned in: Slide 18 (model pipeline — evaluation gap)

How to read the output:
    - Random scatter around zero → Ridge captured the pattern. Stop.
    - U-shaped or curved residuals → systematic error remaining.
      Check: was the log transform applied? If yes, try polynomial terms.
    - Funnel shape (wider at one end) → variance changes with concentration.
      Consider weighted regression or log-transforming the target.

Usage:
    python residual_analysis.py                    (runs demo with simulated data)
    python residual_analysis.py predictions.csv    (uses your data)

CSV format expected:
    actual_mg_L, predicted_mg_L
    20, 22.3
    40, 38.1
    ...
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os


def analyze_residuals(y_true, y_pred, labels=None):
    """
    Compute residual statistics.

    Returns a dict with mean, std, skewness, normality test, and
    whether a quadratic pattern is detected.
    """
    residuals = np.array(y_pred) - np.array(y_true)

    # basic stats
    mean_r = np.mean(residuals)
    std_r = np.std(residuals, ddof=1)
    skew_r = float(stats.skew(residuals))
    n = len(residuals)

    # test for normality (Shapiro-Wilk, only meaningful for n >= 8)
    if n >= 8:
        stat_sw, p_sw = stats.shapiro(residuals)
        normality = "normal (p={:.3f})".format(p_sw) if p_sw > 0.05 else "non-normal (p={:.3f})".format(p_sw)
    else:
        normality = "too few points to test"

    # test for quadratic trend in residuals
    if n >= 5:
        x = np.array(y_true)
        coeffs = np.polyfit(x, residuals, 2)
        # if the quadratic coefficient is significant relative to residual spread
        quad_magnitude = abs(coeffs[0]) * (x.max() - x.min()) ** 2
        has_curve = quad_magnitude > std_r * 0.5
    else:
        has_curve = False

    return {
        "n": n,
        "mean": mean_r,
        "std": std_r,
        "skewness": skew_r,
        "max_absolute": np.max(np.abs(residuals)),
        "normality": normality,
        "has_quadratic_pattern": has_curve,
        "residuals": residuals,
    }


def plot_residuals(y_true, y_pred, title="Residual Analysis", save_path="residual_analysis.png"):
    """
    Generate a two-panel residual plot:
        LEFT:  residuals vs predicted concentration
        RIGHT: histogram of residuals
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_pred - y_true

    stats_result = analyze_residuals(y_true, y_pred)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0a0f1a")

    # LEFT: residuals vs predicted
    ax1.set_facecolor("#0f1525")
    ax1.axhline(y=0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")

    # shade ± 1 RMSE zone
    rmse = np.sqrt(np.mean(residuals ** 2))
    ax1.axhspan(-rmse, rmse, alpha=0.08, color="#4ADE80")
    ax1.axhline(y=rmse, color="#4ADE80", linewidth=0.5, alpha=0.3, linestyle=":")
    ax1.axhline(y=-rmse, color="#4ADE80", linewidth=0.5, alpha=0.3, linestyle=":")

    color = "#FF6B6B" if stats_result["has_quadratic_pattern"] else "#4ADE80"
    ax1.scatter(y_pred, residuals, color=color, s=60, zorder=5,
                edgecolors="white", linewidth=0.5, alpha=0.8)

    # fit quadratic trend line if pattern detected
    if stats_result["has_quadratic_pattern"]:
        sort_idx = np.argsort(y_pred)
        coeffs = np.polyfit(y_pred, residuals, 2)
        trend_x = np.linspace(y_pred.min(), y_pred.max(), 100)
        trend_y = np.polyval(coeffs, trend_x)
        ax1.plot(trend_x, trend_y, "--", color="#FF6B6B", linewidth=1.5, alpha=0.7)
        ax1.text(0.5, 0.05, "⚠️ systematic pattern detected",
                 transform=ax1.transAxes, ha="center", va="bottom",
                 fontsize=10, color="#FF6B6B", style="italic")
    else:
        ax1.text(0.5, 0.05, "✅ residuals look random — model is capturing the pattern",
                 transform=ax1.transAxes, ha="center", va="bottom",
                 fontsize=10, color="#4ADE80", style="italic")

    ax1.set_xlabel("Predicted concentration (mg/L)", fontsize=11, color="white")
    ax1.set_ylabel("Residual: predicted − actual (mg/L)", fontsize=11, color="white")
    ax1.set_title("Residuals vs Predicted", fontsize=13, fontweight="bold", color="white")

    # annotation box with stats
    stats_text = (f"Mean: {stats_result['mean']:.2f} mg/L\n"
                  f"Std: {stats_result['std']:.2f} mg/L\n"
                  f"RMSE: {rmse:.2f} mg/L\n"
                  f"Skew: {stats_result['skewness']:.2f}")
    ax1.text(0.02, 0.95, stats_text,
             transform=ax1.transAxes, ha="left", va="top",
             fontsize=9, color="white", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                       edgecolor="gray", alpha=0.8))

    # RIGHT: histogram
    ax2.set_facecolor("#0f1525")
    ax2.hist(residuals, bins=max(8, len(residuals) // 3), color=color, alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax2.axvline(x=0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")
    ax2.set_xlabel("Residual (mg/L)", fontsize=11, color="white")
    ax2.set_ylabel("Count", fontsize=11, color="white")
    ax2.set_title("Residual Distribution", fontsize=13, fontweight="bold", color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(title, fontsize=15, fontweight="bold", color="white", y=0.98)

    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    return stats_result


def print_verdict(stats_result):
    """Print a plain-language interpretation."""
    print()
    print("=" * 55)
    print("  RESIDUAL ANALYSIS VERDICT")
    print("=" * 55)
    print(f"  n = {stats_result['n']} data points")
    print(f"  Mean residual:     {stats_result['mean']:>8.3f} mg/L")
    print(f"  Std residual:      {stats_result['std']:>8.3f} mg/L")
    print(f"  Max |residual|:    {stats_result['max_absolute']:>8.3f} mg/L")
    print(f"  Skewness:          {stats_result['skewness']:>8.3f}")
    print(f"  Normality:         {stats_result['normality']}")
    print(f"  Quadratic pattern: {'YES ⚠️' if stats_result['has_quadratic_pattern'] else 'no ✅'}")
    print()

    if stats_result["has_quadratic_pattern"]:
        print("  VERDICT: Residuals show a systematic curve.")
        print("  Possible causes:")
        print("    1. Log transform (pseudo-absorbance) not applied")
        print("    2. Non-linearity at high concentrations (Beer-Lambert ceiling)")
        print("    3. Missing polynomial terms")
        print("  Try: apply log transform first, then re-fit and re-check.")
    else:
        print("  VERDICT: Residuals look random.")
        print("  The model appears to be capturing the underlying pattern.")
        print("  Adding complexity (RF, NN) is unlikely to help much.")
    print("=" * 55)


# =========================================================
# DEMO — simulated examples of "good" and "bad" residuals
# =========================================================
def run_demo():
    """Generate two example plots: one good, one with pattern."""
    np.random.seed(42)
    n = 30
    conc_true = np.sort(np.random.uniform(10, 290, n))

    # Good residuals: random scatter
    pred_good = conc_true + np.random.normal(0, 4, n)
    result_good = plot_residuals(conc_true, pred_good,
                                 title="Example: Random Residuals (Ridge is sufficient)",
                                 save_path="residuals_good.png")
    print_verdict(result_good)

    # Bad residuals: systematic U-shape
    pred_bad = conc_true + 0.0005 * (conc_true - 150) ** 2 - 3 + np.random.normal(0, 2, n)
    result_bad = plot_residuals(conc_true, pred_bad,
                                title="Example: Systematic Pattern (something is missing)",
                                save_path="residuals_problem.png")
    print_verdict(result_bad)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        import csv
        y_true, y_pred = [], []
        with open(sys.argv[1]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true.append(float(row["actual_mg_L"]))
                y_pred.append(float(row["predicted_mg_L"]))
        result = plot_residuals(y_true, y_pred, save_path="residual_analysis.png")
        print_verdict(result)
    else:
        print("No CSV provided. Running demo with simulated data.")
        print()
        run_demo()
        print()
        print("To use with your data:")
        print("  python residual_analysis.py predictions.csv")
        print("  CSV columns: actual_mg_L, predicted_mg_L")
