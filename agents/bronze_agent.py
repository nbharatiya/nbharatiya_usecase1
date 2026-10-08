"""Overlay STTM mappings on source CSVs and write auditable Bronze Parquet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import BRONZE_DIR


def _cast(series: pd.Series, data_type: str) -> pd.Series:
    kind = data_type.lower().strip()
    if kind in {"int64", "int", "integer"}:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if kind in {"float64", "float", "double"}:
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if kind in {"datetime64", "datetime", "timestamp"}:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if kind in {"str", "string", "object"}:
        return series.astype("string")
    return series


def load_bronze(run_id: str, file_info: dict[str, Any], mappings: list[dict[str, Any]]) -> Path:
    source_path = Path(file_info["path"])
    frame = pd.read_csv(source_path, low_memory=False)
    rename_map: dict[str, str] = {}
    casts: dict[str, str] = {}
    for mapping in mappings:
        source = str(mapping.get("source_column", ""))
        target = str(mapping.get("target_column", source))
        if source and source in frame.columns and target:
            rename_map[source] = target
            casts[target] = str(mapping.get("data_type", "str"))
    frame = frame.rename(columns=rename_map)
    for column, kind in casts.items():
        if column in frame.columns:
            frame[column] = _cast(frame[column], kind)
    frame["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
    frame["_source_file"] = source_path.name
    output_dir = BRONZE_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source_path.stem}.parquet"
    frame.to_parquet(output, index=False)
    return output
