"""
Sample Preparation Calculator
==============================
Computes the exact volumes of stock solution, extra ethanol, and water
needed for each target concentration — ensuring all samples end up with
the same ethanol fraction regardless of curcumin concentration.

How to use:
    1. Set your stock concentration, target ethanol %, and final volume
    2. Run the script
    3. Follow the table for each concentration

Mentioned in: Slide 14 (Ethanol ratio fix)
              Slide 17 in original numbering

Why this matters:
    Stock solution is curcumin in pure ethanol. Without adjustment,
    20 mg/L has 2% ethanol and 300 mg/L has 30% ethanol.
    Different ethanol content changes refractive index, keto-enol
    equilibrium, and absorption spectrum — confounding the concentration
    measurement.
"""

import numpy as np


def calculate_preparation(
    target_concentrations_mg_L,
    stock_concentration_mg_L=1000.0,
    target_ethanol_fraction=0.30,
    final_volume_mL=5.00,
):
    """
    Calculate preparation volumes for all target concentrations.

    Parameters
    ----------
    target_concentrations_mg_L : list of float
        Target curcumin concentrations in mg/L
    stock_concentration_mg_L : float
        Stock solution concentration in mg/L (dissolved in pure ethanol)
    target_ethanol_fraction : float
        Target ethanol volume fraction (e.g., 0.30 for 30%)
    final_volume_mL : float
        Final sample volume in mL

    Returns
    -------
    list of dicts with volumes and checks
    """
    results = []

    target_ethanol_mL = target_ethanol_fraction * final_volume_mL

    for c_target in target_concentrations_mg_L:
        # Volume of stock needed for target concentration
        v_stock = (c_target * final_volume_mL) / stock_concentration_mg_L

        # Ethanol already in the stock volume (stock is pure ethanol)
        ethanol_from_stock = v_stock

        # Extra ethanol needed to reach target fraction
        extra_ethanol = target_ethanol_mL - ethanol_from_stock

        # Water to fill the rest
        water = final_volume_mL - v_stock - extra_ethanol

        # Checks
        warnings = []
        if extra_ethanol < 0:
            warnings.append(
                f"Stock alone provides more ethanol ({ethanol_from_stock:.2f} mL) "
                f"than target ({target_ethanol_mL:.2f} mL). "
                f"Increase target ethanol fraction or lower concentration."
            )
        if water < 0:
            warnings.append(
                f"Total liquids exceed final volume. "
                f"Increase final volume or lower concentration."
            )
        if v_stock > final_volume_mL:
            warnings.append(
                f"Stock volume ({v_stock:.2f} mL) exceeds final volume. "
                f"Concentration too high for this stock."
            )

        actual_ethanol_pct = (ethanol_from_stock + max(extra_ethanol, 0)) / final_volume_mL * 100

        results.append({
            "concentration_mg_L": c_target,
            "v_stock_mL": round(v_stock, 3),
            "v_extra_ethanol_mL": round(max(extra_ethanol, 0), 3),
            "v_water_mL": round(max(water, 0), 3),
            "v_total_mL": round(v_stock + max(extra_ethanol, 0) + max(water, 0), 3),
            "ethanol_pct": round(actual_ethanol_pct, 1),
            "warnings": warnings,
        })

    return results


def print_table(results, stock_conc, ethanol_target, final_vol):
    """Pretty-print the preparation table."""
    print("=" * 80)
    print("  SAMPLE PREPARATION TABLE")
    print(f"  Stock: {stock_conc} mg/L in pure ethanol")
    print(f"  Target ethanol: {ethanol_target*100:.0f}% v/v in all samples")
    print(f"  Final volume: {final_vol:.2f} mL per sample")
    print("=" * 80)
    print()
    print(f"  {'Conc':>8s}  {'Stock':>8s}  {'+ EtOH':>8s}  {'+ Water':>8s}  "
          f"{'Total':>8s}  {'EtOH %':>7s}  Status")
    print(f"  {'(mg/L)':>8s}  {'(mL)':>8s}  {'(mL)':>8s}  {'(mL)':>8s}  "
          f"{'(mL)':>8s}  {'':>7s}")
    print("  " + "-" * 72)

    for r in results:
        status = "✅" if not r["warnings"] else "⚠️"
        print(f"  {r['concentration_mg_L']:>8.0f}  "
              f"{r['v_stock_mL']:>8.3f}  "
              f"{r['v_extra_ethanol_mL']:>8.3f}  "
              f"{r['v_water_mL']:>8.3f}  "
              f"{r['v_total_mL']:>8.3f}  "
              f"{r['ethanol_pct']:>6.1f}%  "
              f"{status}")
        for w in r["warnings"]:
            print(f"           ⚠️  {w}")

    print()
    ethanol_values = [r["ethanol_pct"] for r in results if not r["warnings"]]
    if ethanol_values:
        if len(set(ethanol_values)) == 1:
            print(f"  ✅ All samples have identical ethanol content: {ethanol_values[0]}%")
        else:
            print(f"  ⚠️  Ethanol content varies: {min(ethanol_values)}% – {max(ethanol_values)}%")
    print("=" * 80)


def print_without_adjustment(concentrations, stock_conc, final_vol):
    """Show what happens if you DON'T adjust ethanol — for comparison."""
    print()
    print("  FOR COMPARISON — without ethanol adjustment:")
    print("  " + "-" * 50)
    for c in concentrations:
        v_stock = (c * final_vol) / stock_conc
        ethanol_pct = (v_stock / final_vol) * 100
        print(f"  {c:>8.0f} mg/L  →  {ethanol_pct:>5.1f}% ethanol")
    print("  " + "-" * 50)
    print("  These are NOT the same solvent environment.")
    print()


# =========================================================
# EXAMPLE — using the plan's 12 concentrations
# =========================================================
if __name__ == "__main__":

    # The plan's 12 target concentrations
    concentrations = [0, 20, 40, 60, 80, 100, 120, 150, 180, 220, 260, 300]

    stock_conc = 1000.0     # mg/L in pure ethanol
    ethanol_target = 0.30   # 30% v/v
    final_vol = 5.00        # mL

    # Show the problem first
    print_without_adjustment(concentrations, stock_conc, final_vol)

    # Show the fix
    results = calculate_preparation(
        concentrations,
        stock_concentration_mg_L=stock_conc,
        target_ethanol_fraction=ethanol_target,
        final_volume_mL=final_vol,
    )
    print_table(results, stock_conc, ethanol_target, final_vol)

    print()
    print("NOTE: Adjust 'target_ethanol_fraction' if 30% doesn't work for solubility.")
    print("      The blank (0 mg/L) should use the same ethanol fraction — no curcumin,")
    print("      but the same solvent composition as all other samples.")
