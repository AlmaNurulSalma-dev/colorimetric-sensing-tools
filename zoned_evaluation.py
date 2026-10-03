"""
Zoned Evaluation
=================
Splits model predictions into concentration zones (low / mid / high)
and reports R², MAE, RMSE, MARE per zone — instead of one global number
that can hide poor performance at specific ranges.

Mentioned in: Slide 18 (model pipeline — evaluation gap)

Why this matters:
    A global RMSE of 5 mg/L might average out a 12.3 mg/L error at low
    concentrations and a 1.8 mg/L error at high concentrations. If the
    application cares about low concentrations (which is common), the
    global number is misleading.

Usage:
    python zoned_evaluation.py                     (runs demo)
    python zoned_evaluation.py predictions.csv     (uses your data)

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


def compute_metrics(y_true, y_pred):
    """Compute R², MAE, RMSE, MARE for a set of predictions."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    n = len(y_true)
    if n < 2:
        return {"n": n, "R2": None, "MAE": None, "RMSE": None, "MARE": None}

    residuals = y_pred - y_true
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))

    # MARE: exclude zeros to avoid division by zero
    nonzero_mask = y_true > 0
    if np.any(nonzero_mask):
        mare = np.mean(np.abs(residuals[nonzero_mask]) / y_true[nonzero_mask]) * 100
    else:
        mare = None

    return {
        "n": n,
        "R2": round(r2, 4) if r2 is not None else None,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MARE": round(mare, 1) if mare is not None else None,
    }


def evaluate_by_zone(y_true, y_pred, zones=None):
    """
    Split predictions into concentration zones and compute metrics per zone.

    Parameters
    ----------
    y_true : array-like
        Actual concentrations
    y_pred : array-like
        Predicted concentrations
    zones : list of tuples, optional
        Zone boundaries as [(low_start, low_end), (mid_start, mid_end), ...]
        Defaults to equal thirds of the data range.

    Returns
    -------
    dict with global metrics and per-zone metrics
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if zones is None:
        # auto-split into three equal zones
        c_min, c_max = y_true.min(), y_true.max()
        third = (c_max - c_min) / 3
        zones = [
            (c_min, c_min + third),
            (c_min + third, c_min + 2 * third),
            (c_min + 2 * third, c_max + 0.1),  # +0.1 to include the max
        ]

    zone_names = ["Low", "Mid", "High"] if len(zones) == 3 else [f"Zone {i+1}" for i in range(len(zones))]

    results = {
        "global": compute_metrics(y_true, y_pred),
        "zones": [],
    }

    for i, (z_start, z_end) in enumerate(zones):
        mask = (y_true >= z_start) & (y_true < z_end)
        if np.any(mask):
            zone_metrics = compute_metrics(y_true[mask], y_pred[mask])
        else:
            zone_metrics = {"n": 0, "R2": None, "MAE": None, "RMSE": None, "MARE": None}

        zone_metrics["name"] = zone_names[i] if i < len(zone_names) else f"Zone {i+1}"
        zone_metrics["range"] = f"{z_start:.0f}–{z_end:.0f} mg/L"
        results["zones"].append(zone_metrics)

    return results


def print_report(results, target_rmse=None):
    """Pretty-print the zoned evaluation report."""
    g = results["global"]

    print("=" * 70)
    print("  ZONED EVALUATION REPORT")
    print("=" * 70)
    print()
    print(f"  GLOBAL METRICS (n = {g['n']})")
    print(f"  R²:   {g['R2']}")
    print(f"  MAE:  {g['MAE']} mg/L")
    print(f"  RMSE: {g['RMSE']} mg/L")
    print(f"  MARE: {g['MARE']}%")
    print()
    print("  PER-ZONE BREAKDOWN")
    print("  " + "-" * 64)
    print(f"  {'Zone':<8s} {'Range':<18s} {'n':>4s} {'R²':>7s} {'MAE':>8s} {'RMSE':>8s} {'MARE':>7s}  Flag")
    print("  " + "-" * 64)

    for z in results["zones"]:
        flag = ""
        if target_rmse and z["RMSE"] is not None and z["RMSE"] > target_rmse:
            flag = f"❌ exceeds target ({target_rmse})"
        elif z["RMSE"] is not None:
            flag = "✅"

        r2_str = f"{z['R2']:.4f}" if z["R2"] is not None else "  n/a"
        mae_str = f"{z['MAE']:.2f}" if z["MAE"] is not None else "  n/a"
        rmse_str = f"{z['RMSE']:.2f}" if z["RMSE"] is not None else "  n/a"
        mare_str = f"{z['MARE']:.1f}%" if z["MARE"] is not None else "  n/a"

        print(f"  {z['name']:<8s} {z['range']:<18s} {z['n']:>4d} {r2_str:>7s} "
              f"{mae_str:>8s} {rmse_str:>8s} {mare_str:>7s}  {flag}")

    print("  " + "-" * 64)
    print()

    # highlight the worst zone
    valid_zones = [z for z in results["zones"] if z["RMSE"] is not None]
    if valid_zones:
        worst = max(valid_zones, key=lambda z: z["RMSE"])
        best = min(valid_zones, key=lambda z: z["RMSE"])
        ratio = worst["RMSE"] / best["RMSE"] if best["RMSE"] > 0 else 0

        if ratio > 3:
            print(f"  ⚠️  {worst['name']} zone RMSE ({worst['RMSE']} mg/L) is {ratio:.1f}× worse")
            print(f"      than {best['name']} zone ({best['RMSE']} mg/L).")
            print(f"      The global RMSE ({g['RMSE']} mg/L) is hiding this gap.")
        elif ratio > 1.5:
            print(f"  Note: {worst['name']} zone is {ratio:.1f}× worse than {best['name']} zone.")
            print(f"        Worth investigating, but not dramatic.")
        else:
            print(f"  ✅ Performance is relatively uniform across zones.")
    print("=" * 70)


def plot_zone_comparison(results, target_rmse=None, save_path="zone_evaluation.png"):
    """Bar chart of RMSE per zone with target line."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0a0f1a")
    ax.set_facecolor("#0f1525")

    zones = results["zones"]
    names = [z["name"] + "\n" + z["range"] for z in zones]
    rmse_values = [z["RMSE"] if z["RMSE"] is not None else 0 for z in zones]

    # color by severity
    colors = []
    for z in zones:
        if z["RMSE"] is None:
            colors.append("#555555")
        elif target_rmse and z["RMSE"] > target_rmse:
            colors.append("#FF6B6B")
        elif target_rmse and z["RMSE"] > target_rmse * 0.7:
            colors.append("#FFD700")
        else:
            colors.append("#4ADE80")

    bars = ax.bar(names, rmse_values, color=colors, edgecolor="white", linewidth=0.5, width=0.5)

    # value labels on bars
    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="white")

    if target_rmse:
        ax.axhline(y=target_rmse, color="white", linewidth=1, linestyle="--", alpha=0.7)
        ax.text(len(zones) - 0.5, target_rmse + 0.3, f"target: {target_rmse} mg/L",
                ha="right", va="bottom", fontsize=9, color="white", alpha=0.7)

    # global RMSE annotation
    ax.axhline(y=results["global"]["RMSE"], color="#A78BFA", linewidth=1,
               linestyle=":", alpha=0.5)
    ax.text(0, results["global"]["RMSE"] + 0.3,
            f"global RMSE: {results['global']['RMSE']} mg/L",
            ha="left", va="bottom", fontsize=9, color="#A78BFA", alpha=0.7)

    ax.set_ylabel("RMSE (mg/L)", fontsize=11, color="white")
    ax.set_title("Model Error by Concentration Zone", fontsize=13,
                 fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# =========================================================
# DEMO
# =========================================================
def run_demo():
    """Simulated demo showing how global RMSE hides zone-level problems."""
    np.random.seed(42)

    # simulate a model that works well at high concentrations
    # but poorly at low concentrations
    n = 60
    y_true = np.sort(np.random.uniform(5, 295, n))

    # error increases at low concentrations
    noise_scale = 15 * np.exp(-y_true / 50) + 1.5
    y_pred = y_true + np.random.normal(0, 1, n) * noise_scale

    zones = [(0, 100), (100, 200), (200, 300)]
    results = evaluate_by_zone(y_true, y_pred, zones=zones)

    print_report(results, target_rmse=8.0)
    plot_zone_comparison(results, target_rmse=8.0, save_path="zone_evaluation.png")


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

        zones_input = [(0, 100), (100, 200), (200, 300)]
        results = evaluate_by_zone(y_true, y_pred, zones=zones_input)
        print_report(results, target_rmse=8.0)
        plot_zone_comparison(results, target_rmse=8.0)
    else:
        print("No CSV provided. Running demo with simulated data.")
        print()
        run_demo()
        print()
        print("To use with your data:")
        print("  python zoned_evaluation.py predictions.csv")
        print("  CSV columns: actual_mg_L, predicted_mg_L")
        print()
        print("To change zones, edit the 'zones' variable in the script:")
        print("  zones = [(0, 50), (50, 150), (150, 300)]")
