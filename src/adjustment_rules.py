from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Band:
    left: float
    right: float
    closed: str = "right"

    def contains(self, value: float) -> bool:
        if self.closed == "both":
            return self.left <= value <= self.right
        if self.closed == "left":
            return self.left <= value < self.right
        if self.closed == "neither":
            return self.left < value < self.right
        return self.left < value <= self.right

    def label(self) -> str:
        left_bracket = "[" if self.closed in {"left", "both"} else "("
        right_bracket = "]" if self.closed in {"right", "both"} else ")"
        return f"{left_bracket}{self.left:g}, {self.right:g}{right_bracket}"


def band_from_dict(value: dict[str, Any]) -> Band:
    return Band(
        left=float(value["left"]),
        right=float(value["right"]),
        closed=str(value.get("closed", "right")),
    )


def classify_adjustment(predicted_ph: float, rules: dict[str, Any]) -> str:
    low_band = band_from_dict(rules["low_band"])
    medium_bands = [band_from_dict(item) for item in rules["medium_bands"]]

    if low_band.contains(predicted_ph):
        return "Low"
    if any(band.contains(predicted_ph) for band in medium_bands):
        return "Medium"
    return "High"


def describe_rules(rules: dict[str, Any]) -> dict[str, str]:
    low = band_from_dict(rules["low_band"]).label()
    medium = ", ".join(band_from_dict(item).label() for item in rules["medium_bands"])
    return {"Low": low, "Medium": medium or "None", "High": "Outside Low and Medium bands"}
