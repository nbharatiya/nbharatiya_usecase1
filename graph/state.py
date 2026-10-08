"""Shared state contract for the LangGraph pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

class PipelineState(TypedDict, total=False):
    run_id: str
    business_intent: str
    uploaded_files_info: list[dict[str, Any]]
    profile: dict[str, Any]
    bronze_sttm: list[dict[str, Any]]
    silver_sttm: list[dict[str, Any]]
    gold_sttm: list[dict[str, Any]]
    bronze_paths: list[str]
    silver_paths: list[str]
    gold_paths: list[str]
    bronze_validation: dict[str, Any]
    silver_validation: dict[str, Any]
    gold_validation: dict[str, Any]
    report_path: str
    report_sql: str
    report_result_df: list[dict[str, Any]]
    current_step: str
    sttm_hint: str
    error: str | None
    approved_layers: list[str]
    retry_counts: dict[str, int]
    strict_validation: bool
    auto_approve: bool
    max_gold_rows: int
    custom_report_sql: str
