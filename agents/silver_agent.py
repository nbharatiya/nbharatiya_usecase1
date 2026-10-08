"""Consolidate Bronze data, apply approved quality rules, and key Silver rows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import SILVER_DIR


def _fill_value(strategy: str, series: pd.Series) -> Any:
    value = strategy.strip().lower()
    if value in {"zero", "0", "fill=0"}:
        return 0
    if value in {"unknown", "fill=unknown", "empty", "fill=empty"}:
        return "unknown" if "unknown" in value else ""
    if value in {"mean", "fill=mean"} and pd.api.types.is_numeric_dtype(series):
        return series.mean()
    if value in {"median", "fill=median"} and pd.api.types.is_numeric_dtype(series):
        return series.median()
    if value in {"mode", "fill=mode"}:
        modes = series.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else None
    function_match = re.search(
        r"fill_nulls?\s*\(\s*(['\"])(.*?)\1\s*\)",
        strategy,
        flags=re.IGNORECASE,
    )
    if function_match:
        return function_match.group(2)
    assignment_match = re.search(r"fill\s*=\s*([^;,]+)", strategy, flags=re.IGNORECASE)
    return assignment_match.group(1).strip().strip("'\"") if assignment_match else None


def load_silver(run_id: str, bronze_paths: list[str], mappings: list[dict[str, Any]]) -> Path:
    frames = [pd.read_parquet(path) for path in bronze_paths]
    if not frames:
        raise ValueError("No Bronze Parquet inputs were found.")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    dedup_keys: list[str] = []
    fill_strategies: dict[str, Any] = {}
    for mapping in mappings:
        target = str(mapping.get("target_column", ""))
        source = str(mapping.get("source_column", target))
        column = target if target in frame.columns else source
        directives = f"{mapping.get('transformation', '')} {mapping.get('notes', '')}"
        if column not in frame.columns:
            continue
        strategy = _fill_value(directives, frame[column])
        if strategy is not None:
            fill_strategies[column] = strategy
        if re.search(r"dedup[_ -]?key\s*=?\s*true|dedup_key", directives, flags=re.IGNORECASE):
            dedup_keys.append(column)
    if dedup_keys:
        dedup_keys = list(dict.fromkeys(dedup_keys))
        has_key = frame[dedup_keys].notna().all(axis=1)
        duplicate_keys = frame.loc[has_key].duplicated(subset=dedup_keys, keep="first")
        frame = frame.drop(index=duplicate_keys[duplicate_keys].index).reset_index(drop=True)
    for column, strategy in fill_strategies.items():
        frame[column] = frame[column].fillna(strategy)
    frame.insert(0, "pk_silver_id", range(1, len(frame) + 1))
    output_dir = SILVER_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "silver.parquet"
    frame.to_parquet(output, index=False)
    return output
