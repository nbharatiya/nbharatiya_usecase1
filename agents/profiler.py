"""CSV profiling for the pipeline's source datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def profile_files(files: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    for file_info in files:
        path = Path(file_info["path"])
        frame = pd.read_csv(path, low_memory=False)
        columns = []
        for name in frame.columns:
            values = frame[name]
            columns.append({
                "name": str(name),
                "dtype": str(values.dtype),
                "null_pct": round(float(values.isna().mean() * 100), 2) if len(values) else 0.0,
                "unique_count": int(values.nunique(dropna=True)),
                "top_values": [str(value) for value in values.value_counts(dropna=True).head(5).index],
            })
        profiles.append({
            "file_name": file_info.get("name", path.name),
            "path": str(path),
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "columns": columns,
        })
    combined = {"files": profiles, "total_rows": sum(item["row_count"] for item in profiles)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "combined_profile.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined
