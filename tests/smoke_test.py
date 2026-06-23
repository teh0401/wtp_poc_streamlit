import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_service import ClarifierModelService


def main() -> None:
    service = ClarifierModelService(ROOT / "artifacts")
    samples = pd.read_csv(ROOT / "artifacts" / "sample_inputs.csv")
    values = samples.iloc[0][service.feature_names].to_dict()
    result = service.predict(values)
    assert result.adjustment_level in {"Low", "Medium", "High"}
    assert 0.0 < result.predicted_clarifier_ph < 14.0
    print(
        {
            "predicted_clarifier_ph": round(result.predicted_clarifier_ph, 4),
            "adjustment_level": result.adjustment_level,
        }
    )


if __name__ == "__main__":
    main()
