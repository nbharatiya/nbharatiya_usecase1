"""Conditional routing after layer validation."""

from __future__ import annotations

from typing import Any

from core.config import MAX_VALIDATION_ATTEMPTS
from graph.state import PipelineState


def route_after_validation(layer: str, next_stage: str):
    def route(state: PipelineState) -> str:
        validation = state.get(f"{layer}_validation", {})
        if validation.get("valid"):
            return next_stage
        if state.get("error"):
            return "end"
        if int(state.get("retry_counts", {}).get(layer, 0)) < MAX_VALIDATION_ATTEMPTS:
            return f"{layer}_load"
        return "end"
    return route
