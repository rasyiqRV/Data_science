# ─────────────────────────────────────────────────────────────────
#  train_model.py
#  Vertix – Train & export the Life Expectancy Random Forest model.
#
#  Mirrors the notebook (uts_rasyiq_LED_23670015_RandomForest).
#  Produces:
#    app/static/models/life_expectancy_model.pkl
#    app/static/models/model_metadata.pkl
#    app/static/img/life_expectancy_distribution.png
#    app/static/img/feature_importance.png
#
#  Run once (and again whenever the dataset changes):
#      python train_model.py
# ─────────────────────────────────────────────────────────────────

import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "Life-Expectancy-Data-Updated.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "app", "static", "models")
IMG_DIR    = os.path.join(BASE_DIR, "app", "static", "img")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ── Feature definition (identical to the notebook) ────────────────
FEATURES = ["GDP_per_capita", "Schooling", "Adult_mortality", "BMI"]
TARGET   = "Life_expectancy"

sns.set_theme(style="whitegrid")


def main():
    # ── 1. Load data ──────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 2. Train Random Forest Regressor ──────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── 3. Evaluate ───────────────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)

    print("=" * 45)
    print("    EVALUATION — RANDOM FOREST (Life Expectancy)")
    print("=" * 45)
    print(f"  R-Squared (R2)        : {r2:.4f}")
    print(f"  Mean Abs Error (MAE)  : {mae:.2f} years")
    print(f"  Root MSE (RMSE)       : {rmse:.2f} years")
    print(f"  Cross-Val R2 (5-fold) : {cv.mean():.4f} +/- {cv.std():.4f}")
    print("=" * 45)

    # ── 4. Feature ranges (for form validation + hints) ───────────
    feature_ranges = {}
    for col in FEATURES:
        feature_ranges[col] = {
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "mean": float(df[col].mean()),
        }

    # ── 5. Save model + metadata ──────────────────────────────────
    importances = {col: float(imp) for col, imp in zip(FEATURES, model.feature_importances_)}

    metadata = {
        "problem_type":         "regression",
        "feature_columns":      FEATURES,
        "numeric_features":     FEATURES,
        "categorical_features": [],
        "target_column":        TARGET,
        "feature_ranges":       feature_ranges,
        "feature_importances":  importances,
        "metrics": {
            "r2":   round(float(r2), 4),
            "mae":  round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "cv_r2_mean": round(float(cv.mean()), 4),
            "cv_r2_std":  round(float(cv.std()), 4),
        },
        "algorithm":      "Random Forest Regressor",
        "n_estimators":   200,
        "max_depth":      10,
        "dataset_rows":   int(df.shape[0]),
        "dataset_cols":   int(df.shape[1]),
        "training_date":  date.today().isoformat(),
        "model_name":     "Life Expectancy Prediction Model",
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "life_expectancy_model.pkl"))
    joblib.dump(metadata, os.path.join(MODEL_DIR, "model_metadata.pkl"))
    print(f"Model + metadata saved to {MODEL_DIR}")

    # ── 6. Charts for the dashboard ───────────────────────────────
    # 6a. Target distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(y, kde=True, color="#6366f1", bins=40)
    plt.title("Distribution of Life Expectancy")
    plt.xlabel("Life Expectancy (years)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "life_expectancy_distribution.png"), dpi=110)
    plt.close()

    # 6b. Feature importance
    feat_df = (
        pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True)
    )
    plt.figure(figsize=(9, 5))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#10b981")
    plt.title("Feature Importance — Random Forest Regressor")
    plt.xlabel("Importance (0–1)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=110)
    plt.close()
    print(f"Charts saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
