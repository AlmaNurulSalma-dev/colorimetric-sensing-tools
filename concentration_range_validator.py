"""
Concentration Range Validator
=============================
Checks whether the planned maximum concentration stays within
the Beer-Lambert linear range for the TCS34725 sensor.

How to use:
    1. Measure a blank (0 mg/L) — record R, G, B, Clear
    2. Measure the highest planned concentration — record R, G, B, Clear
    3. Plug the numbers into this script
    4. Read the verdict

Mentioned in: Slide 8 (Linearity ceiling fix)

Reference:
    Kurata, HardwareX 9 (2021) e00161
    Absorbance reliable up to A ≈ 1.0, usable up to A ≈ 1.8
"""

import numpy as np


def validate_range(blank, sample, max_conc_mg_L):
    """
    Compute pseudo-absorbance for each channel and assess linearity.

    Parameters
    ----------
    blank : dict
        Blank readings, e.g. {"R": 8200, "G": 9150, "B": 10800, "Clear": 32100}
    sample : dict
        Readings at max concentration, same keys as blank
    max_conc_mg_L : float
        The maximum concentration used (mg/L)

    Returns
    -------
    dict with absorbance per channel and verdict per channel
    """
    results = {}

    for ch in ["R", "G", "B", "Clear"]:
        if blank[ch] <= 0 or sample[ch] <= 0:
            results[ch] = {"absorbance": None, "verdict": "INVALID — zero or negative reading"}
            continue

        A = np.log10(blank[ch] / sample[ch])

        if A < 1.0:
            verdict = "OK — within linear range"
            color = "green"
        elif A < 1.5:
            verdict = "CAUTION — usable but approaching non-linear zone"
            color = "yellow"
        else:
            verdict = "TOO HIGH — sensor is in non-linear zone, consider lowering max concentration"
            color = "red"

        results[ch] = {
            "absorbance": round(A, 4),
            "blank_count": blank[ch],
            "sample_count": sample[ch],
            "transmission_pct": round(sample[ch] / blank[ch] * 100, 1),
            "verdict": verdict,
            "color": color,
        }

    return results


def find_recommended_max(blank_B, target_A=1.0, epsilon_estimate=None):
    """
    If absorbance at max concentration exceeds target_A, estimate
    the concentration where A_B = target_A based on Beer-Lambert linearity.

    This is a rough estimate — real validation requires actual measurements
    at intermediate concentrations.
    """
    if epsilon_estimate is None:
        return "Need actual measurements at several concentrations to estimate this."

    # A = epsilon * l * C  =>  C_max = target_A / (epsilon * l)
    # This requires knowing epsilon and path length, which we don't have yet.
    return "Run Phase 1 at reduced concentrations to find the actual linear limit."


def print_report(results, max_conc_mg_L):
    """Pretty-print the validation results."""
    print("=" * 65)
    print(f"  CONCENTRATION RANGE VALIDATION — max: {max_conc_mg_L} mg/L")
    print("=" * 65)
    print()

    for ch in ["R", "G", "B", "Clear"]:
        r = results[ch]
        if r["absorbance"] is None:
            print(f"  {ch:>5s}:  {r['verdict']}")
        else:
            indicator = {"green": "✅", "yellow": "⚠️", "red": "❌"}[r["color"]]
            print(f"  {ch:>5s}:  A = {r['absorbance']:.4f}   "
                  f"({r['transmission_pct']}% transmission)   "
                  f"{indicator} {r['verdict']}")
    print()

    # overall recommendation
    absorbances = [r["absorbance"] for r in results.values() if r["absorbance"] is not None]
    if not absorbances:
        print("  No valid readings to assess.")
        return

    max_A = max(absorbances)
    primary_ch = [ch for ch in results if results[ch].get("absorbance") == max_A][0]

    print("-" * 65)
    if max_A < 1.0:
        print(f"  RESULT: Range 0–{max_conc_mg_L} mg/L is within the linear zone.")
        print(f"  Proceed as planned.")
    elif max_A < 1.5:
        print(f"  RESULT: {primary_ch} channel reaches A = {max_A:.3f} at {max_conc_mg_L} mg/L.")
        print(f"  Usable, but flag concentrations above ~{int(max_conc_mg_L * 1.0/max_A)} mg/L")
        print(f"  as potentially non-linear.")
    else:
        suggested = int(max_conc_mg_L * 1.0 / max_A)
        print(f"  RESULT: {primary_ch} channel reaches A = {max_A:.3f} at {max_conc_mg_L} mg/L.")
        print(f"  This is above the reliable limit (A = 1.0).")
        print(f"  Consider lowering max concentration to ~{suggested} mg/L")
        print(f"  or verifying linearity at intermediate points.")
    print("=" * 65)


# =========================================================
# EXAMPLE — using the plan's simulated reference data
# =========================================================
if __name__ == "__main__":

    # From the research plan's reference table (simulated/example data)
    blank_readings = {"R": 8200, "G": 9150, "B": 10800, "Clear": 32100}

    # Readings at 300 mg/L (simulated)
    sample_300 = {"R": 4160, "G": 3860, "B": 4720, "Clear": 19400}

    results = validate_range(blank_readings, sample_300, max_conc_mg_L=300)
    print_report(results, max_conc_mg_L=300)

    print()
    print("NOTE: These are the plan's SIMULATED example values.")
    print("Replace with actual sensor readings before Phase 1.")
