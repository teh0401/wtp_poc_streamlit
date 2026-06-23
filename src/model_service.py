from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.adjustment_rules import classify_adjustment
from src.features import coerce_feature_frame


@dataclass(frozen=True)
class PredictionResult:
    predicted_clarifier_ph: float
    adjustment_level: str
    warnings: list[str]
    metadata: dict[str, Any]


class ClarifierModelService:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.feature_names = self._load_json("feature_list.json")["features"]
        self.metadata = self._load_json("metadata.json")
        self.adjustment_rules = self._load_json("adjustment_rules.json")

    def _load_json(self, filename: str) -> dict[str, Any]:
        with (self.artifact_dir / filename).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def predict(self, values: dict[str, Any]) -> PredictionResult:
        frame = coerce_feature_frame(values, self.feature_names)
        missing = frame.attrs.get("missing_features", [])
        warnings: list[str] = []
        if missing:
            warnings.append(f"Missing or invalid features: {', '.join(missing)}")
        if frame.isna().any(axis=None):
            raise ValueError(warnings[0] if warnings else "Input contains missing values.")

        predicted_ph = float(self.model.predict(frame)[0])
        adjustment_level = classify_adjustment(predicted_ph, self.adjustment_rules)

        return PredictionResult(
            predicted_clarifier_ph=predicted_ph,
            adjustment_level=adjustment_level,
            warnings=warnings,
            metadata=self.metadata,
        )
