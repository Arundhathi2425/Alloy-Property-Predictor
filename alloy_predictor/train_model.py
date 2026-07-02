"""
train_model.py
---------------
Trains and compares models to predict alloy yield strength from composition,
then explains the best model with SHAP.

Pipeline:
  1. Load alloy_composition_data.csv (see generate_dataset.py)
  2. Train/compare Linear Regression vs. Random Forest (5-fold CV, R^2 + RMSE)
  3. Fit the winning model on the full training split
  4. SHAP summary plot for feature-level interpretability
  5. Save the trained model + linear-model coefficients (for the browser demo)
  6. Provide a `predict_yield_strength()` helper for CLI use
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import joblib

DATA_FILE = "alloy_composition_data.csv"
FEATURES = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V"]
TARGET = "yield_strength_MPa"


def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET]
    return df, X, y


def compare_models(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
    }

    print("5-fold cross-validation (R^2):")
    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        scores[name] = cv_scores
        print(f"  {name:20s}  mean R^2 = {cv_scores.mean():.3f}  (+/- {cv_scores.std():.3f})")
    return scores


def train_final_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\nRandom Forest test set:  R^2 = {r2:.3f}   RMSE = {rmse:.1f} MPa")

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return rf, lr, X_train, X_test, y_train, y_test


def explain_with_shap(rf, X_train, X_test):
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved SHAP summary plot: shap_summary.png")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    print("\nMean |SHAP value| per feature (impact on predicted yield strength):")
    print(importance.round(1).to_string())
    return importance


def save_artifacts(rf, lr):
    joblib.dump(rf, "random_forest_model.joblib")

    # Save linear-model coefficients as JSON so the browser demo (predictor.html)
    # can reproduce a fast, dependency-free estimate without needing a live backend.
    linear_coeffs = {
        "intercept": round(float(lr.intercept_), 2),
        "coefficients": {f: round(float(c), 2) for f, c in zip(FEATURES, lr.coef_)},
    }
    with open("linear_model_coeffs.json", "w") as f:
        json.dump(linear_coeffs, f, indent=2)
    print("\nSaved random_forest_model.joblib and linear_model_coeffs.json")


def predict_yield_strength(composition: dict, model) -> float:
    """composition: dict with keys C, Mn, Si, Cr, Ni, Mo, V (wt%)."""
    x = pd.DataFrame([{f: composition.get(f, 0.0) for f in FEATURES}])
    return float(model.predict(x)[0])


def main():
    df, X, y = load_data()
    compare_models(X, y)
    rf, lr, X_train, X_test, y_train, y_test = train_final_model(X, y)
    explain_with_shap(rf, X_train, X_test)
    save_artifacts(rf, lr)

    example = {"C": 0.4, "Mn": 1.2, "Si": 0.3, "Cr": 1.0, "Ni": 0.5, "Mo": 0.2, "V": 0.05}
    pred = predict_yield_strength(example, rf)
    print(f"\nExample prediction for composition {example}:")
    print(f"  Predicted yield strength: {pred:.1f} MPa")


if __name__ == "__main__":
    main()
