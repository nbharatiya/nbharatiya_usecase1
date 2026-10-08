"""Node implementations for the IDAMP medallion graph."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agents.auditor import log_stage_event
from agents.bronze_agent import load_bronze
from agents.gold_agent import load_gold
from agents.profiler import profile_files
from agents.reporter import create_report
from agents.silver_agent import load_silver
from agents.sttm_generator import generate_sttm
from agents.validator import validate_layer
from core.config import LANDING_DIR, MAX_VALIDATION_ATTEMPTS
from core.sttm_versioning import save_sttm
from graph.state import PipelineState


def _tracked(state: PipelineState, stage: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = operation()
        validation = next((value for key, value in result.items() if key.endswith("_validation")), {})
        rows = validation.get("rows")
        score = validation.get("quality_score")
        log_stage_event(state["run_id"], stage, "completed", rows, score, started_at=started)
        return result
    except Exception as exc:
        log_stage_event(state["run_id"], stage, "failed", error_message=str(exc), started_at=started)
        return {"error": f"{stage}: {exc}", "current_step": "error"}


def profile_node(state: PipelineState) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        run_dir = LANDING_DIR / state["run_id"]
        profile = profile_files(state["uploaded_files_info"], run_dir)
        return {"profile": profile, "current_step": "bronze_sttm"}
    return _tracked(state, "profile", operation)


def _sttm_node(state: PipelineState, layer: str) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        mappings = generate_sttm(layer, state["profile"], state["business_intent"], state.get("sttm_hint", ""))
        if layer == "gold":
            mappings = mappings[:8]
        save_sttm(state["run_id"], layer, mappings)
        return {f"{layer}_sttm": mappings, "current_step": f"{layer}_hitl"}
    return _tracked(state, f"{layer}_sttm", operation)


def bronze_sttm_node(state: PipelineState) -> dict[str, Any]:
    return _sttm_node(state, "bronze")


def silver_sttm_node(state: PipelineState) -> dict[str, Any]:
    return _sttm_node(state, "silver")


def gold_sttm_node(state: PipelineState) -> dict[str, Any]:
    return _sttm_node(state, "gold")


def _hitl_node(state: PipelineState, layer: str) -> dict[str, Any]:
    if state.get("error"):
        raise RuntimeError(state["error"])
    if layer not in state.get("approved_layers", []):
        raise RuntimeError(f"{layer.title()} STTM must be approved before loading.")
    return {"current_step": f"{layer}_load"}


def bronze_hitl_node(state: PipelineState) -> dict[str, Any]:
    return _hitl_node(state, "bronze")


def silver_hitl_node(state: PipelineState) -> dict[str, Any]:
    return _hitl_node(state, "silver")


def gold_hitl_node(state: PipelineState) -> dict[str, Any]:
    return _hitl_node(state, "gold")


def _load_node(state: PipelineState, layer: str) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        if layer == "bronze":
            paths = [str(load_bronze(state["run_id"], info, state["bronze_sttm"])) for info in state["uploaded_files_info"]]
            return {"bronze_paths": paths, "current_step": "bronze_validate"}
        if layer == "silver":
            path = load_silver(state["run_id"], state["bronze_paths"], state["silver_sttm"])
            return {"silver_paths": [str(path)], "current_step": "silver_validate"}
        path = load_gold(
            state["run_id"],
            state["silver_paths"],
            state["gold_sttm"],
            max_rows=int(state.get("max_gold_rows", 8)),
            business_intent=state["business_intent"],
        )
        return {"gold_paths": [str(path)], "current_step": "gold_validate"}
    counts = dict(state.get("retry_counts", {}))
    counts[layer] = counts.get(layer, 0) + 1
    result = _tracked(state, f"{layer}_load", operation)
    result["retry_counts"] = counts
    return result


def bronze_load_node(state: PipelineState) -> dict[str, Any]:
    return _load_node(state, "bronze")


def silver_load_node(state: PipelineState) -> dict[str, Any]:
    return _load_node(state, "silver")


def gold_load_node(state: PipelineState) -> dict[str, Any]:
    return _load_node(state, "gold")


def _validate_node(state: PipelineState, layer: str) -> dict[str, Any]:
    if state.get("error"):
        return {"current_step": "error"}

    def operation() -> dict[str, Any]:
        path_key = f"{layer}_paths"
        frame = pd.concat([pd.read_parquet(path) for path in state[path_key]], ignore_index=True)
        source_count = max((len(item.get("columns", [])) for item in state.get("profile", {}).get("files", [])), default=0) if layer == "bronze" else 0
        validation = validate_layer(layer, frame, source_count)
        counts = dict(state.get("retry_counts", {}))
        if validation["valid"]:
            counts[layer] = 0
        return {f"{layer}_validation": validation, "retry_counts": counts, "current_step": f"{layer}_validate"}
    return _tracked(state, f"{layer}_validate", operation)


def bronze_validate_node(state: PipelineState) -> dict[str, Any]:
    return _validate_node(state, "bronze")


def silver_validate_node(state: PipelineState) -> dict[str, Any]:
    return _validate_node(state, "silver")


def gold_validate_node(state: PipelineState) -> dict[str, Any]:
    return _validate_node(state, "gold")


def report_node(state: PipelineState) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        output = create_report(state["run_id"], state["business_intent"], state["gold_paths"][0])
        result_frame = output.get("report_result_df")
        if isinstance(result_frame, pd.DataFrame):
            output["report_result_df"] = result_frame.to_dict(orient="records")
        return {**output, "current_step": "complete"}
    return _tracked(state, "report", operation)
