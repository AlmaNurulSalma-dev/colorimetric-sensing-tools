# Colorimetric Sensing — Pre-experiment Scripts

Python scripts for the environment-compensated multi-channel colorimetric sensing research project (Arduino UNO + TCS34725 + curcumin).

| Script | What it does |
|---|---|
| `concentration_range_validator.py` | Checks if max concentration stays within Beer-Lambert linear range (A < 1.5) |
| `sample_preparation_calculator.py` | Computes volumes (stock + ethanol + water) so all samples have identical ethanol fraction |
| `pseudo_absorbance_demo.py` | Plots raw B (curved) vs pseudo-absorbance (straight) — proves log transform linearizes the curve |
| `residual_analysis.py` | Plots model residuals, detects systematic patterns, prints verdict |
| `zoned_evaluation.py` | Reports R², MAE, RMSE, MARE per concentration zone instead of one global number |
