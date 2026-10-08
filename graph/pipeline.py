"""Construct and cache the interruptible IDAMP StateGraph."""

from __future__ import annotations

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.edges import route_after_validation
from graph.nodes import (
    bronze_hitl_node,
    bronze_load_node,
    bronze_sttm_node,
    bronze_validate_node,
    gold_hitl_node,
    gold_load_node,
    gold_sttm_node,
    gold_validate_node,
    profile_node,
    report_node,
    silver_hitl_node,
    silver_load_node,
    silver_sttm_node,
    silver_validate_node,
)
from graph.state import PipelineState


@st.cache_resource(show_spinner=False)
def get_pipeline():
    builder = StateGraph(PipelineState)
    nodes = {
        "profile_node": profile_node,
        "bronze_sttm_node": bronze_sttm_node,
        "bronze_hitl_node": bronze_hitl_node,
        "bronze_load_node": bronze_load_node,
        "bronze_validate_node": bronze_validate_node,
        "silver_sttm_node": silver_sttm_node,
        "silver_hitl_node": silver_hitl_node,
        "silver_load_node": silver_load_node,
        "silver_validate_node": silver_validate_node,
        "gold_sttm_node": gold_sttm_node,
        "gold_hitl_node": gold_hitl_node,
        "gold_load_node": gold_load_node,
        "gold_validate_node": gold_validate_node,
        "report_node": report_node,
    }
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "profile_node")
    builder.add_edge("profile_node", "bronze_sttm_node")
    builder.add_edge("bronze_sttm_node", "bronze_hitl_node")
    builder.add_edge("bronze_hitl_node", "bronze_load_node")
    builder.add_edge("bronze_load_node", "bronze_validate_node")
    builder.add_conditional_edges("bronze_validate_node", route_after_validation("bronze", "silver_sttm_node"), {"silver_sttm_node": "silver_sttm_node", "bronze_load": "bronze_load_node", "end": END})
    builder.add_edge("silver_sttm_node", "silver_hitl_node")
    builder.add_edge("silver_hitl_node", "silver_load_node")
    builder.add_edge("silver_load_node", "silver_validate_node")
    builder.add_conditional_edges("silver_validate_node", route_after_validation("silver", "gold_sttm_node"), {"gold_sttm_node": "gold_sttm_node", "silver_load": "silver_load_node", "end": END})
    builder.add_edge("gold_sttm_node", "gold_hitl_node")
    builder.add_edge("gold_hitl_node", "gold_load_node")
    builder.add_edge("gold_load_node", "gold_validate_node")
    builder.add_conditional_edges("gold_validate_node", route_after_validation("gold", "report_node"), {"report_node": "report_node", "gold_load": "gold_load_node", "end": END})
    builder.add_edge("report_node", END)
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["bronze_hitl_node", "silver_hitl_node", "gold_hitl_node"],
    )


def retry_failed_stage(graph, config: dict, state: PipelineState) -> None:
    """Clear a node error and resume from that stage's immediate predecessor."""
    message = str(state.get("error") or "")
    stage = message.split(":", 1)[0]
    if stage.endswith("_validate"):
        layer = stage.removesuffix("_validate")
        if f"{layer}_paths" not in state:
            stage = f"{layer}_load"
    predecessors = {
        "bronze_sttm": "profile_node",
        "bronze_load": "bronze_hitl_node",
        "bronze_validate": "bronze_load_node",
        "silver_sttm": "bronze_validate_node",
        "silver_load": "silver_hitl_node",
        "silver_validate": "silver_load_node",
        "gold_sttm": "silver_validate_node",
        "gold_load": "gold_hitl_node",
        "gold_validate": "gold_load_node",
        "report": "gold_validate_node",
    }
    predecessor = predecessors.get(stage)
    if predecessor:
        graph.update_state(config, {"error": None, "current_step": stage}, as_node=predecessor)
        graph.invoke(None, config)
        return
    if stage == "profile":
        initial = {
            key: state[key]
            for key in (
                "run_id", "business_intent", "uploaded_files_info", "approved_layers",
                "retry_counts", "strict_validation", "auto_approve", "max_gold_rows", "sttm_hint",
            )
            if key in state
        }
        initial["error"] = None
        graph.invoke(initial, config)
        return
    graph.update_state(config, {"error": None})
    graph.invoke(None, config)
