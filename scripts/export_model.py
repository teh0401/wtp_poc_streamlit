from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "Dataset" / "processed" / "feature_engineered_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "pH_9021_Clarifier"
HORIZON_STEPS = 6
HORIZON_LABEL = "30min"

MODEL_FEATURES = [
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
    "pH_9021_Clarifier",
    "pH_9021_Clarifier_lag5min",
    "pH_9021_Clarifier_rollmean_30min",
    "pH_9021_Clarifier_diff",
]

RULE_COLUMNS = [
    "pH_9021_Clarifier",
    "NTU_9041_Treated",
    "NTU_9031_Filtered",
    "pH_9041_Treated",
]


def treated_ph_penalty(ph_value: float, low: float = 6.5, high: float = 9.0) -> float:
    if low <= ph_value <= high:
        return 0.0
    if ph_value < low:
        return low - ph_value
    return ph_value - high


def minmax_scale(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


def interval_to_dict(interval: pd.Interval) -> dict[str, float | str]:
    return {
        "left": float(interval.left),
        "right": float(interval.right),
        "closed": str(interval.closed),
    }


def build_adjustment_rules(df: pd.DataFrame, train_index: pd.Index) -> tuple[dict, pd.DataFrame]:
    rule_train = df.loc[train_index, RULE_COLUMNS].dropna().copy()
    rule_train["treated_pH_penalty_raw"] = rule_train["pH_9041_Treated"].apply(treated_ph_penalty)
    rule_train["NTU_9041_scaled"] = minmax_scale(rule_train["NTU_9041_Treated"])
    rule_train["NTU_9031_scaled"] = minmax_scale(rule_train["NTU_9031_Filtered"])
    rule_train["treated_pH_penalty_scaled"] = minmax_scale(rule_train["treated_pH_penalty_raw"])
    rule_train["performance_score"] = (
        0.5 * rule_train["NTU_9041_scaled"]
        + 0.3 * rule_train["NTU_9031_scaled"]
        + 0.2 * rule_train["treated_pH_penalty_scaled"]
    )

    bin_width = 0.5
    ph_min = np.floor(rule_train[TARGET].min() / bin_width) * bin_width
    ph_max = np.ceil(rule_train[TARGET].max() / bin_width) * bin_width
    bins = np.arange(ph_min, ph_max + bin_width, bin_width)
    rule_train["clarifier_pH_band"] = pd.cut(rule_train[TARGET], bins=bins, include_lowest=True)

    band_summary = (
        rule_train.groupby("clarifier_pH_band", observed=False)
        .agg(
            count=(TARGET, "size"),
            mean_clarifier_pH=(TARGET, "mean"),
            avg_score=("performance_score", "mean"),
            avg_NTU_9041=("NTU_9041_Treated", "mean"),
            avg_NTU_9031=("NTU_9031_Filtered", "mean"),
            avg_pH_9041=("pH_9041_Treated", "mean"),
        )
        .reset_index()
    )
    band_summary = band_summary[band_summary["count"] >= 100].copy()
    best_band_row = band_summary.loc[band_summary["avg_score"].idxmin()]
    low_band = best_band_row["clarifier_pH_band"]
    valid_bands = list(band_summary["clarifier_pH_band"])
    best_idx = valid_bands.index(low_band)
    medium_bands = []
    if best_idx - 1 >= 0:
        medium_bands.append(valid_bands[best_idx - 1])
    if best_idx + 1 < len(valid_bands):
        medium_bands.append(valid_bands[best_idx + 1])

    rules = {
        "method": "train-derived pH band rule",
        "low_band": interval_to_dict(low_band),
        "medium_bands": [interval_to_dict(band) for band in medium_bands],
        "source": "Project Part 2 predict-then-rule logic",
    }
    return rules, band_summary


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    usecols = sorted(set(MODEL_FEATURES + RULE_COLUMNS + [TARGET, "Time_Stamp"]))
    df = pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=["Time_Stamp"])
    df = df.sort_values("Time_Stamp").set_index("Time_Stamp")

    target_name = f"{TARGET}_t+{HORIZON_LABEL}"
    supervised = pd.concat(
        [df[MODEL_FEATURES], df[TARGET].shift(-HORIZON_STEPS).rename(target_name)],
        axis=1,
    ).dropna()

    train_pos, test_pos = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    random_train_idx = df.index[np.sort(train_pos)]
    train_idx = supervised.index.intersection(random_train_idx)
    test_idx = supervised.index.difference(train_idx)

    eval_model = ExtraTreesRegressor(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    eval_model.fit(supervised.loc[train_idx, MODEL_FEATURES], supervised.loc[train_idx, target_name])
    y_true = supervised.loc[test_idx, target_name]
    y_pred = eval_model.predict(supervised.loc[test_idx, MODEL_FEATURES])

    final_model = ExtraTreesRegressor(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final_model.fit(supervised[MODEL_FEATURES], supervised[target_name])

    rules, band_summary = build_adjustment_rules(df, random_train_idx)
    sample_inputs = supervised[MODEL_FEATURES].reset_index().head(250)

    metadata = {
        "model_name": "ExtraTrees",
        "scenario": "Scenario B: Real-time clarifier forecasting",
        "target": target_name,
        "horizon": HORIZON_LABEL,
        "horizon_minutes": 30,
        "trained_rows": int(len(supervised)),
        "random_split_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "random_split_mae": float(mean_absolute_error(y_true, y_pred)),
        "random_split_r2": float(r2_score(y_true, y_pred)),
        "note": "POC advisory model; chronological validation in the notebook showed regime-shift sensitivity.",
    }

    joblib.dump(final_model, ARTIFACT_DIR / "model.joblib", compress=3)
    (ARTIFACT_DIR / "feature_list.json").write_text(
        json.dumps({"features": MODEL_FEATURES}, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "adjustment_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    sample_inputs.to_csv(ARTIFACT_DIR / "sample_inputs.csv", index=False)
    band_summary.to_csv(ARTIFACT_DIR / "band_summary.csv", index=False)

    print(f"Artifacts written to {ARTIFACT_DIR}")
    print(json.dumps(metadata, indent=2))
    print(json.dumps(rules, indent=2))


if __name__ == "__main__":
    main()
