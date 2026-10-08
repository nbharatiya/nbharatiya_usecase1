"""Quality profiling and score helpers shared by validators and the UI."""

from __future__ import annotations

from typing import Any

import pandas as pd


def quality_score(frame: pd.DataFrame) -> float:
    """Score data quality while respecting source-specific schemas in unions."""
    business_columns = [
        column
        for column in frame.columns
        if column not in {"_load_timestamp", "_source_file"}
        and not str(column).startswith("pk_")
    ]
    if frame.empty or not business_columns:
        return 0.0

    if "_source_file" in frame.columns:
        weighted_score = 0.0
        scored_rows = 0
        for _, source_frame in frame.groupby("_source_file", dropna=False, sort=False):
            source_columns = [column for column in business_columns if source_frame[column].notna().any()]
            if not source_columns:
                continue
            completeness = 1.0 - float(source_frame[source_columns].isna().mean().mean())
            uniqueness = sum(
                source_frame[column].nunique(dropna=True) / max(len(source_frame), 1)
                for column in source_columns
            ) / len(source_columns)
            score = 100 * (0.75 * completeness + 0.25 * uniqueness)
            weighted_score += score * len(source_frame)
            scored_rows += len(source_frame)
        if scored_rows:
            return round(max(0.0, min(100.0, weighted_score / scored_rows)), 2)

    completeness = 1.0 - float(frame[business_columns].isna().mean().mean())
    uniqueness = sum(
        frame[column].nunique(dropna=True) / max(len(frame), 1)
        for column in business_columns
    ) / len(business_columns)
    return round(max(0.0, min(100.0, (0.75 * completeness + 0.25 * uniqueness) * 100)), 2)


def column_quality(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in frame.columns:
        series = frame[column]
        null_pct = float(series.isna().mean() * 100) if len(series) else 0.0
        rows.append({
            "column": str(column),
            "null_pct": round(null_pct, 2),
            "unique_count": int(series.nunique(dropna=True)),
            "quality": round(max(0.0, 100.0 - null_pct), 2),
        })
    return rows
