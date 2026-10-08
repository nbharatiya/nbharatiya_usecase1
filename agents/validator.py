"""Layer-specific structural and quality validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.quality import quality_score


def validate_layer(layer: str, frame: pd.DataFrame, source_column_count: int = 0) -> dict[str, Any]:
    failures: list[str] = []
    score = quality_score(frame)
    if layer == "bronze":
        if len(frame) <= 0:
            failures.append("Bronze output has no rows.")
        if not {"_load_timestamp", "_source_file"}.issubset(frame.columns):
            failures.append("Bronze audit columns are missing.")
        if len(frame.columns) < source_column_count:
            failures.append("Bronze output has fewer columns than the source.")
    elif layer == "silver":
        if "pk_silver_id" not in frame.columns:
            failures.append("pk_silver_id is missing.")
        elif frame["pk_silver_id"].isna().any():
            failures.append("pk_silver_id contains nulls.")
        if score < 70:
            failures.append(f"Average quality score {score:.1f} is below 70.")
    elif layer == "gold":
        if len(frame) <= 0:
            failures.append("Gold output has no rows.")
        if "pk_gold_id" not in frame.columns:
            failures.append("pk_gold_id is missing.")
        if not any(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns if column != "pk_gold_id"):
            failures.append("Gold output has no numeric business column.")
    else:
        failures.append(f"Unknown validation layer: {layer}.")
    return {
        "valid": not failures,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "quality_score": score,
        "failures": failures,
    }
