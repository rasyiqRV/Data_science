# ─────────────────────────────────────────────────────────────────
#  app/services/model_service.py
#  Vertix – Life Expectancy prediction service.
#
#  Model: Random Forest Regressor trained on the WHO
#  "Life-Expectancy-Data-Updated" dataset.
#
#  Features (numeric):
#    - GDP_per_capita    : Income per capita (USD)
#    - Schooling         : Average years of schooling
#    - Adult_mortality   : Adult mortality rate
#    - BMI               : Body Mass Index
#  Target: Life_expectancy (years)
# ─────────────────────────────────────────────────────────────────

import os
import joblib
import pandas as pd
from flask import current_app

# ── Feature definition (must match training in train_model.py) ────
NUMERIC_FEATURES = ["GDP_per_capita", "Schooling", "Adult_mortality", "BMI"]
CATEGORICAL_FEATURES = []
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-friendly labels used by the UI.
FEATURE_LABELS = {
    "GDP_per_capita":  "GDP per Capita (USD)",
    "Schooling":       "Schooling (years)",
    "Adult_mortality": "Adult Mortality (per 1000)",
    "BMI":             "BMI",
}

# Fallback validation ranges (overridden by metadata feature_ranges).
_FALLBACK_RANGES = {
    "GDP_per_capita":  {"min": 100.0,  "max": 120000.0},
    "Schooling":       {"min": 0.0,    "max": 15.0},
    "Adult_mortality": {"min": 1.0,    "max": 800.0},
    "BMI":             {"min": 18.0,   "max": 35.0},
}

# ── Singleton loader ──────────────────────────────────────────────
_model = None
_metadata = None


def _get_model_dir() -> str:
    return os.path.join(current_app.root_path, "static", "models")


def load_model():
    global _model, _metadata

    model_dir     = _get_model_dir()
    model_path    = os.path.join(model_dir, "life_expectancy_model.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `python train_model.py` to generate the .pkl files first."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    _model    = joblib.load(model_path)
    _metadata = joblib.load(metadata_path)
    current_app.logger.info("✅ Life expectancy model loaded successfully.")


def get_metadata() -> dict:
    if _metadata is None:
        load_model()
    return _metadata


def get_model():
    if _model is None:
        load_model()
    return _model


def _get_ranges() -> dict:
    """Return per-feature {min, max} ranges from metadata (or fallback)."""
    try:
        meta = get_metadata()
        ranges = meta.get("feature_ranges")
        if ranges:
            return ranges
    except Exception:
        pass
    return _FALLBACK_RANGES


# ── Input validation ──────────────────────────────────────────────
class PredictionInputError(ValueError):
    pass


def validate_input(form_data: dict) -> dict:
    errors = []
    cleaned = {}
    ranges = _get_ranges()

    for col in NUMERIC_FEATURES:
        raw = (form_data.get(col, "") or "").strip()
        label = FEATURE_LABELS.get(col, col)

        if not raw:
            errors.append(f"'{label}' must not be empty.")
            continue

        try:
            val = float(raw)
        except ValueError:
            errors.append(f"'{label}' must be a number.")
            continue

        # Soft range check (allow a little slack beyond observed data).
        rng = ranges.get(col, {})
        lo  = rng.get("min")
        hi  = rng.get("max")
        if lo is not None and hi is not None:
            slack = (hi - lo) * 0.25
            if val < lo - slack or val > hi + slack:
                errors.append(
                    f"'{label}' should be between {lo:.1f} and {hi:.1f}."
                )
                continue

        cleaned[col] = val

    if errors:
        raise PredictionInputError(" | ".join(errors))

    return cleaned


# ── Core prediction ───────────────────────────────────────────────
def predict_life_expectancy(form_data: dict) -> dict:
    result = {
        "success":       False,
        "prediction":    None,
        "formatted":     "",
        "problem_type":  "",
        "input_summary": {},
        "error":         None,
    }

    try:
        cleaned = validate_input(form_data)
        meta    = get_metadata()
        model   = get_model()

        feature_cols = meta.get("feature_columns", FEATURE_COLUMNS)
        input_df = pd.DataFrame([{col: cleaned.get(col) for col in feature_cols}])

        raw_pred = float(model.predict(input_df)[0])

        result.update({
            "success":       True,
            "prediction":    raw_pred,
            "formatted":     f"{raw_pred:.1f} years",
            "problem_type":  meta.get("problem_type", "regression"),
            "input_summary": cleaned,
        })

    except PredictionInputError as e:
        result["error"] = str(e)
    except FileNotFoundError as e:
        current_app.logger.error(f"Model file error: {e}")
        result["error"] = "Model is not available yet. Please contact the administrator."
    except Exception as e:
        current_app.logger.exception(f"Prediction error: {e}")
        result["error"] = f"An error occurred during prediction: {str(e)}"

    return result


# ── Helpers for templates ─────────────────────────────────────────
def get_field_options() -> dict:
    """
    Numeric-only model: return sensible {min, max, step, placeholder}
    metadata per feature for rendering the input form.
    """
    ranges = _get_ranges()
    fields = {}
    for col in NUMERIC_FEATURES:
        rng = ranges.get(col, {})
        lo  = rng.get("min", 0)
        hi  = rng.get("max", 100)
        mean = rng.get("mean", (lo + hi) / 2)
        # Step granularity depends on the magnitude of the range.
        step = 1 if (hi - lo) > 50 else 0.1
        fields[col] = {
            "label":       FEATURE_LABELS.get(col, col),
            "min":         round(lo, 2),
            "max":         round(hi, 2),
            "step":        step,
            "placeholder": f"e.g. {round(mean, 1)}",
        }
    return fields


def get_model_info() -> dict:
    try:
        meta = get_metadata()
        return {
            "problem_type":         meta.get("problem_type", "N/A"),
            "feature_columns":      meta.get("feature_columns", []),
            "numeric_features":     meta.get("numeric_features", []),
            "categorical_features": meta.get("categorical_features", []),
            "target_column":        meta.get("target_column", "Life_expectancy"),
            "model_name":           meta.get("model_name", "Life Expectancy Prediction Model"),
            "algorithm":            meta.get("algorithm", "Random Forest Regressor"),
            "metrics":              meta.get("metrics", {}),
            "dataset_rows":         meta.get("dataset_rows"),
            "dataset_cols":         meta.get("dataset_cols"),
            "training_date":        meta.get("training_date"),
            "feature_labels":       FEATURE_LABELS,
            "model_loaded":         _model is not None,
        }
    except Exception:
        return {"model_loaded": False}
