from __future__ import annotations

from datetime import datetime
from math import cos, pi, sin
from typing import Any, Mapping

import pandas as pd

TARGET = "pH_9021_Clarifier"
HORIZON_STEPS = 6
HORIZON_LABEL = "30min"
HORIZON_MINUTES = 30

UPSTREAM_FEATURES = [
    "pH_9011_Raw",
    "pH_9011_Raw_lag5min",
    "pH_9011_Raw_rollmean_30min",
    "NTU_9011_Raw",
    "NTU_9011_Raw_lag5min",
    "NTU_9011_Raw_rollmean_30min",
    "month",
    "hour",
    "dayofweek",
    "hour_sin",
    "hour_cos",
]

REALTIME_CLARIFIER_FEATURES = UPSTREAM_FEATURES + [
    "pH_9021_Clarifier",
    "pH_9021_Clarifier_lag5min",
    "pH_9021_Clarifier_rollmean_30min",
    "pH_9021_Clarifier_diff",
]

MODEL_FEATURES = REALTIME_CLARIFIER_FEATURES


def time_features(timestamp: datetime) -> dict[str, float]:
    hour = timestamp.hour
    return {
        "month": float(timestamp.month),
        "hour": float(hour),
        "dayofweek": float(timestamp.weekday()),
        "hour_sin": sin(2 * pi * hour / 24),
        "hour_cos": cos(2 * pi * hour / 24),
    }


def coerce_feature_frame(values: dict[str, Any], feature_names: list[str] | None = None) -> pd.DataFrame:
    names = feature_names or MODEL_FEATURES
    row: dict[str, float] = {}
    missing: list[str] = []

    for name in names:
        value = values.get(name)
        if value is None or value == "":
            missing.append(name)
            row[name] = float("nan")
            continue
        try:
            row[name] = float(value)
        except (TypeError, ValueError):
            missing.append(name)
            row[name] = float("nan")

    frame = pd.DataFrame([row], columns=names)
    if missing:
        frame.attrs["missing_features"] = missing
    return frame


def build_features_from_raw(
    raw_ph: float,
    raw_ntu: float,
    database_row: Mapping[str, Any],
    timestamp: datetime,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    names = feature_names or MODEL_FEATURES
    features: dict[str, float] = {}

    for name in names:
        value = database_row.get(name)
        features[name] = float(value)

    features["pH_9011_Raw"] = float(raw_ph)
    features["NTU_9011_Raw"] = float(raw_ntu)
    features.update(time_features(timestamp))
    return features


def build_supervised_frame(source_df: pd.DataFrame, features: list[str], target: str = TARGET) -> tuple[pd.DataFrame, str]:
    target_name = f"{target}_t+{HORIZON_LABEL}"
    future_target = source_df[target].shift(-HORIZON_STEPS).rename(target_name)
    combined = pd.concat([source_df[features], future_target], axis=1).dropna()
    return combined, target_name
