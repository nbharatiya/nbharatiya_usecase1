"""Persist approved STTM snapshots for reproducibility."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import STTM_HISTORY_DIR


def save_sttm(run_id: str, layer: str, mappings: list[dict[str, Any]]) -> Path:
    STTM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "layer": layer,
        "version": datetime.now(timezone.utc).isoformat(),
        "mappings": mappings,
    }
    path = STTM_HISTORY_DIR / f"{run_id}_{layer}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
