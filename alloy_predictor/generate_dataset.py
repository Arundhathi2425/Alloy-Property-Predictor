"""
generate_dataset.py
--------------------
Generates a SYNTHETIC steel-alloy composition dataset for the ML property
predictor project.

Why synthetic: getting a large, clean, license-free dataset of real alloy
compositions + measured yield strength is hard without lab access or a paid
database (e.g. Citrination/MatWeb). Instead, this script generates
compositions in realistic wt% ranges for low-alloy steels, and derives yield
strength using physically-motivated solid-solution-strengthening terms
(each element's approximate strengthening contribution, in MPa per wt%,
loosely based on published solid-solution-strengthening coefficients),
plus Gaussian noise to mimic real scatter.

This keeps the ML pipeline (feature engineering, model comparison, SHAP
interpretability) genuinely meaningful to build and explain, while being
transparent that the labels are simulated, not lab-measured.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_SAMPLES = 300

# Composition ranges (wt%), representative of low-alloy / carbon steels
ranges = {
    "C":  (0.05, 1.20),
    "Mn": (0.30, 2.00),
    "Si": (0.10, 1.50),
    "Cr": (0.00, 5.00),
    "Ni": (0.00, 4.00),
    "Mo": (0.00, 1.00),
    "V":  (0.00, 0.30),
}

# Approximate solid-solution / precipitation strengthening coefficients
# (MPa contribution per wt%, illustrative order-of-magnitude values)
coeffs = {
    "C": 650,   # interstitial strengthening dominates; applied via sqrt(C) below
    "Mn": 30,
    "Si": 50,
    "Cr": 40,
    "Ni": 25,
    "Mo": 60,
    "V": 220,   # strong carbide former / grain refiner
}

BASE_YS = 160.0  # MPa, approximate annealed ferrite baseline


def generate():
    data = {el: np.random.uniform(lo, hi, N_SAMPLES) for el, (lo, hi) in ranges.items()}
    df = pd.DataFrame(data)

    ys = (
        BASE_YS
        + coeffs["C"] * np.sqrt(df["C"])          # interstitial term, sub-linear
        + coeffs["Mn"] * df["Mn"]
        + coeffs["Si"] * df["Si"]
        + coeffs["Cr"] * df["Cr"]
        + coeffs["Ni"] * df["Ni"]
        + coeffs["Mo"] * df["Mo"]
        + coeffs["V"] * df["V"]
        + 15 * df["Cr"] * df["Mo"]                # mild synergistic term
    )

    noise = np.random.normal(0, 25, N_SAMPLES)
    df["yield_strength_MPa"] = np.round(ys + noise, 1)

    # Empirical-style hardness-strength correlation (Brinell), plus its own noise
    df["hardness_HB"] = np.round(df["yield_strength_MPa"] / 3.45 + np.random.normal(0, 8, N_SAMPLES), 1)

    df = df.round(3)
    df.to_csv("alloy_composition_data.csv", index=False)
    print(f"Wrote alloy_composition_data.csv with {len(df)} samples.")
    print(df.head())


if __name__ == "__main__":
    generate()
