"""
predict.py  —  Feature engineering + inference for the Traffic Volume Predictor.

From Model_evaluation_preventing_leakage.ipynb Component 4B:
  Removed low-importance (6): ['time_of_day_enc', 'is_workday', 'is_holiday',
                                'is_special_event', 'is_rush_hour', 'is_weekend']
  Retained : 34 features  <- FINAL_lgb_junction{n}.pkl trained on EXACTLY these
"""

import math
import pickle
import pathlib
import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).parent.parent   # repo root

# Exact 34 features the model was trained on (order matters)
GOOD_FEATS = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "day_of_week_sin", "day_of_week_cos",
    "month", "year",
    "junction",
    "temp_norm", "wspd_norm", "pres", "pres_norm",
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h", "lag_168h",
    "rolling_mean_3h",  "rolling_mean_6h",  "rolling_mean_24h",
    "rolling_std_3h",   "rolling_std_6h",   "rolling_std_24h",
    "rolling_max_3h",   "rolling_max_6h",   "rolling_max_24h",
    "rolling_min_3h",   "rolling_min_6h",   "rolling_min_24h",
    "ewma_12h",         "ewma_24h",
]

assert len(GOOD_FEATS) == 34, f"Expected 34, got {len(GOOD_FEATS)}"

WEATHER = {
    "temp": 15.0, "temp_min": 2.0,  "temp_max": 38.0,
    "wspd": 10.0, "wspd_min": 0.0,  "wspd_max": 60.0,
    "pres": 1013.0, "pres_min": 990.0, "pres_max": 1035.0,
}

def _norm(val, vmin, vmax):
    rng = vmax - vmin
    return (val - vmin) / rng if rng else 0.0

_MODEL_CACHE: dict = {}

def get_model(junction: int):
    if junction not in _MODEL_CACHE:
        candidates = [
            BASE / "models" / f"FINAL_lgb_junction{junction}.pkl",
            BASE / "models" / "lightgbm_global.pkl",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    model = pickle.load(f)
                _MODEL_CACHE[junction] = (model, path.name)
                break
        else:
            raise FileNotFoundError(
                f"No model found for junction {junction}. Tried:\n" +
                "\n".join(f"  {p}" for p in candidates)
            )
    return _MODEL_CACHE[junction]

def build_features(junction: int, date, hour: int) -> pd.DataFrame:
    dow      = date.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin  = math.sin(2 * math.pi * dow  / 7)
    dow_cos  = math.cos(2 * math.pi * dow  / 7)
    wd       = WEATHER

    lag_rolling = {f: 0.0 for f in [
        "lag_1h","lag_2h","lag_3h","lag_6h","lag_12h","lag_24h","lag_168h",
        "rolling_mean_3h","rolling_mean_6h","rolling_mean_24h",
        "rolling_std_3h","rolling_std_6h","rolling_std_24h",
        "rolling_max_3h","rolling_max_6h","rolling_max_24h",
        "rolling_min_3h","rolling_min_6h","rolling_min_24h",
        "ewma_12h","ewma_24h",
    ]}

    row = {
        "hour"           : hour,
        "hour_sin"       : hour_sin,
        "hour_cos"       : hour_cos,
        "day_of_week"    : dow,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
        "month"          : date.month,
        "year"           : date.year,
        "junction"       : junction,
        "temp_norm"      : _norm(wd["temp"], wd["temp_min"], wd["temp_max"]),
        "wspd_norm"      : _norm(wd["wspd"], wd["wspd_min"], wd["wspd_max"]),
        "pres"           : wd["pres"],
        "pres_norm"      : _norm(wd["pres"], wd["pres_min"], wd["pres_max"]),
        **lag_rolling,
    }

    return pd.DataFrame([row], columns=GOOD_FEATS)

def predict(junction: int, date, hour: int) -> tuple[float, str]:
    model, model_name = get_model(junction)
    features = build_features(junction, date, hour)
    pred = model.predict(features)[0]
    return max(0.0, round(float(pred), 1)), model_name

def predict_day(junction: int, date) -> list[float]:
    return [predict(junction, date, h)[0] for h in range(24)]
