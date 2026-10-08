"""IDAMP: reviewable, intent-driven medallion data pipeline."""

from __future__ import annotations

import json
import importlib
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.auditor import finish_run, get_recent_runs, get_stage_events, start_run
from agents.sttm_generator import generate_sttm
from app.components import STEPS, inject_styles, render_stepper
from core.config import GOLD_DIR, LANDING_DIR, ensure_directories
from core.quality import column_quality
from core.sttm_versioning import save_sttm
from graph.pipeline import get_pipeline, retry_failed_stage

ensure_directories()
st.set_page_config(page_title="IDAMP | Intent-Driven Data Pipeline", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")

if "files" not in st.session_state:
    st.session_state.files = {}
if "removed_files" not in st.session_state:
    st.session_state.removed_files = set()
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "review_result" not in st.session_state:
    st.session_state.review_result = None
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

inject_styles(st.session_state.light_mode)


def _safe_json(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    return value


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _current_snapshot():
    run_id = st.session_state.run_id
    if not run_id:
        return None, None
    graph = get_pipeline()
    config = _config(run_id)
    return graph, graph.get_state(config)


def _advance_auto_approvals(graph, config, enabled: bool) -> None:
    if not enabled:
        return
    for _ in range(3):
        snapshot = graph.get_state(config)
        if not snapshot.next:
            return
        node = snapshot.next[0]
        if not node.endswith("_hitl_node"):
            return
        layer = node.removesuffix("_hitl_node")
        approved = list(snapshot.values.get("approved_layers", []))
        if layer not in approved:
            approved.append(layer)
        graph.update_state(config, {"approved_layers": approved})
        graph.invoke(None, config)


def _step_index(snapshot) -> int:
    if snapshot is None:
        return 0
    values = snapshot.values
    if values.get("report_path"):
        return 7
    stage = values.get("current_step", "source")
    if snapshot.next:
        node = snapshot.next[0]
        stage = node.removesuffix("_node")
    mapping = {
        "source": 0, "profile": 1, "bronze_sttm": 2, "bronze_hitl": 2,
        "bronze_load": 3, "bronze_validate": 3, "silver_sttm": 4,
        "silver_hitl": 4, "silver_load": 5, "silver_validate": 5,
        "gold_sttm": 6, "gold_hitl": 6, "gold_load": 6,
        "gold_validate": 6, "report": 7, "complete": 7,
    }
    return mapping.get(stage, 0)


def _completed_steps(snapshot) -> set[int]:
    if snapshot is None:
        return set()
    values = snapshot.values
    done = {0} if values.get("uploaded_files_info") else set()
    if values.get("profile"):
        done.add(1)
    if values.get("bronze_validation", {}).get("valid"):
        done.update({2, 3})
    if values.get("silver_validation", {}).get("valid"):
        done.update({4, 5})
    if values.get("gold_validation", {}).get("valid"):
        done.add(6)
    if values.get("report_path"):
        done.add(7)
    return done


def _render_lineage(mappings: list[dict[str, Any]], layer: str) -> None:
    if not mappings:
        return
    labels: list[str] = []
    sources: list[int] = []
    targets: list[int] = []
    for mapping in mappings:
        source = str(mapping.get("source_column", ""))
        target = str(mapping.get("target_column", ""))
        for label in (source, target):
            if label not in labels:
                labels.append(label)
        sources.append(labels.index(source))
        targets.append(labels.index(target))
    figure = go.Figure(go.Sankey(
        node={"label": labels, "pad": 14, "thickness": 16, "color": ["#55d6be" if i % 2 == 0 else "#e4ad57" for i in range(len(labels))]},
        link={"source": sources, "target": targets, "value": [1] * len(sources), "color": "rgba(85,214,190,.24)"},
    ))
    figure.update_layout(height=320, margin={"l": 4, "r": 4, "t": 10, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", font_color="#dce8e8")
    st.plotly_chart(figure, use_container_width=True, key=f"lineage_{layer}")


def _mapping_editor(layer: str, mappings: list[dict[str, Any]], run_id: str, strict: bool) -> None:
    st.subheader(f"{layer.title()} source-to-target mapping")
    st.caption("Review the generated mapping. Source columns stay fixed; edits are checkpointed when approved.")
    if not mappings:
        st.error("No mapping was generated. Use Regenerate with a more specific hint.")
        return
    search_col, type_col, audit_col, sort_col = st.columns([2, 1, 1, 1])
    search = search_col.text_input("Search columns", key=f"{layer}_search", label_visibility="collapsed", placeholder="Search mappings")
    type_filter = type_col.multiselect("Data type", ["str", "int64", "float64", "datetime64"], key=f"{layer}_type_filter", placeholder="Types")
    include_audit = audit_col.toggle("Audit columns", value=True, key=f"{layer}_audit_toggle")
    sort_by = sort_col.selectbox("Sort", ["Source", "Target"], key=f"{layer}_sort", label_visibility="collapsed")
    edited = [dict(item) for item in mappings]
    errors: list[str] = []
    for row in edited:
        messages = []
        if not str(row.get("source_column", "")).strip() or not str(row.get("target_column", "")).strip():
            messages.append("Source and target are required")
        if row.get("data_type") not in {"str", "int64", "float64", "datetime64"}:
            messages.append("Unsupported type")
        row["validation"] = "; ".join(messages) if messages else "OK"
        if messages:
            errors.append(f"{row.get('source_column', '(blank)')}: {', '.join(messages)}")
    frame = pd.DataFrame(edited)
    if not include_audit and layer == "bronze":
        frame = frame[~frame["target_column"].astype(str).str.startswith("_")]
    if type_filter:
        frame = frame[frame["data_type"].isin(type_filter)]
    if search:
        frame = frame[frame.astype(str).apply(lambda row: row.str.contains(search, case=False, regex=False).any(), axis=1)]
    sort_key = "source_column" if sort_by == "Source" else "target_column"
    frame = frame.sort_values(sort_key, kind="stable")
    col_config = {
        "source_column": st.column_config.TextColumn("Source column", disabled=True),
        "target_column": st.column_config.TextColumn("Target column"),
        "transformation": st.column_config.TextColumn("Transformation", width="large"),
        "data_type": st.column_config.SelectboxColumn("Data type", options=["str", "int64", "float64", "datetime64"], required=True),
        "notes": st.column_config.TextColumn("Notes", width="medium"),
        "validation": st.column_config.TextColumn("⚠ Validation", disabled=True),
    }
    edited_frame = st.data_editor(
        frame,
        column_config=col_config,
        disabled=["source_column", "validation"],
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        key=f"{layer}_editor_{run_id}",
    )
    changed = {str(row.get("source_column")): row for row in edited_frame.to_dict(orient="records")}
    for index, row in enumerate(edited):
        edited[index].update(changed.get(str(row.get("source_column")), {}))
    failures = []
    for row in edited:
        if not str(row.get("source_column", "")).strip() or not str(row.get("target_column", "")).strip():
            failures.append(f"{row.get('source_column', '(blank)')}: missing source or target")
        if row.get("data_type") not in {"str", "int64", "float64", "datetime64"}:
            failures.append(f"{row.get('source_column', '(blank)')}: unsupported data type")
    if strict and failures:
        st.error("Mapping validation must pass before approval: " + "; ".join(failures))
    else:
        st.caption(f"{len(edited)} mappings · {sum(1 for item in edited if item.get('transformation'))} transformations · {len(failures)} blocking issues")
    with st.expander("Column lineage", expanded=False):
        for row in edited:
            st.markdown(f"`{row.get('source_column')}` → `{row.get('target_column')}` · {row.get('transformation') or 'direct'} · {row.get('notes') or 'No notes'}")
    action1, action2, action3, action4 = st.columns([1.1, 1.1, 1, 1])
    approve = action1.button("Approve mapping", type="primary", disabled=bool(failures), key=f"{layer}_approve")
    with action2.popover("Regenerate"):
        hint = st.text_area("Guidance for the next mapping", key=f"{layer}_hint", placeholder="Describe a rename, data type, aggregation, or deduplication rule")
        if st.button("Generate again", key=f"{layer}_regenerate"):
            try:
                with st.spinner("Generating a revised mapping…"):
                    snapshot = get_pipeline().get_state(_config(run_id))
                    revised = generate_sttm(layer, snapshot.values["profile"], snapshot.values["business_intent"], hint)
                    if layer == "gold":
                        revised = revised[:8]
                    save_sttm(run_id, layer, revised)
                    get_pipeline().update_state(_config(run_id), {f"{layer}_sttm": revised, "sttm_hint": hint})
                st.toast("Mapping regenerated", icon="✅")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    action3.download_button("Copy JSON", json.dumps(edited, indent=2), file_name=f"{layer}_sttm.json", mime="application/json", key=f"{layer}_copy_json")
    action4.download_button("Download CSV", pd.DataFrame(edited).to_csv(index=False), file_name=f"{layer}_sttm.csv", mime="text/csv", key=f"{layer}_csv")
    if approve:
        graph = get_pipeline()
        config = _config(run_id)
        snapshot = graph.get_state(config)
        approved = list(snapshot.values.get("approved_layers", []))
        if layer not in approved:
            approved.append(layer)
        graph.update_state(config, {f"{layer}_sttm": edited, "approved_layers": approved})
        save_sttm(run_id, layer, edited)
        try:
            with st.status(f"Running {layer.title()} load and validation…", expanded=True) as status:
                graph.invoke(None, config)
                _advance_auto_approvals(graph, config, bool(snapshot.values.get("auto_approve", False)))
                status.update(label=f"{layer.title()} stage finished", state="complete")
            st.toast(f"{layer.title()} mapping approved", icon="✅")
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline paused with an error: {exc}")


def _render_layer_results(layer: str, snapshot) -> None:
    values = snapshot.values
    validation = values.get(f"{layer}_validation", {})
    paths = values.get(f"{layer}_paths", [])
    if not paths:
        return
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    st.subheader(f"{layer.title()} layer results")
    tiles = st.columns(3)
    tiles[0].markdown(f'<div class="kpi"><small>Rows out</small><strong>{len(frame):,}</strong></div>', unsafe_allow_html=True)
    tiles[1].markdown(f'<div class="kpi"><small>Columns</small><strong>{len(frame.columns)}</strong></div>', unsafe_allow_html=True)
    tiles[2].markdown(f'<div class="kpi"><small>Quality</small><strong>{validation.get("quality_score", 0):.1f}%</strong></div>', unsafe_allow_html=True)
    if validation.get("valid"):
        st.success(f"{layer.title()} validation passed")
    elif validation:
        st.error("Validation: " + "; ".join(validation.get("failures", [])))
    with st.expander("Per-file quality and sample", expanded=False):
        if layer == "bronze":
            chart_rows = []
            for path in paths:
                item = pd.read_parquet(path)
                chart_rows.extend([{"file": Path(path).name, "quality": row["quality"], "type": "complete"} for row in column_quality(item)])
            if chart_rows:
                quality_frame = pd.DataFrame(chart_rows)
                st.plotly_chart(px.bar(quality_frame, x="file", y="quality", color="type", barmode="stack", title="Column quality by file"), use_container_width=True)
        sample = st.toggle("Show sample rows", key=f"{layer}_sample")
        if sample:
            st.dataframe(frame.head(50), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(column_quality(frame)), use_container_width=True, hide_index=True)
        st.download_button(f"Download {layer.title()} Parquet", Path(paths[0]).read_bytes(), file_name=Path(paths[0]).name, mime="application/octet-stream", key=f"{layer}_parquet")
    mappings = values.get(f"{layer}_sttm", [])
    if st.button(f"Show {layer.title()} lineage diagram", key=f"{layer}_lineage"):
        _render_lineage(mappings, layer)


def _render_audit(run_id: str | None) -> None:
    st.markdown('<div class="idamp-eyebrow">Run observability</div>', unsafe_allow_html=True)
    if run_id:
        events = get_stage_events(run_id)
        for event in reversed(events[-10:]):
            icon = "✓" if event["status"] == "completed" else "!"
            with st.expander(f"{icon} {event['stage']} · {event['status']}", expanded=False):
                st.write(f"Rows out: {event['rows_out'] if event['rows_out'] is not None else '—'}")
                st.write(f"Quality: {event['quality_score'] if event['quality_score'] is not None else '—'}")
                st.write(f"Started: {event['started_at']}")
                if event["error_message"]:
                    st.error(event["error_message"])
    st.markdown('<div class="idamp-eyebrow" style="margin-top:18px">Recent runs</div>', unsafe_allow_html=True)
    for run in get_recent_runs(5):
        if st.button(f"{run['status']} · {run['run_id'][:8]}", key=f"history_{run['run_id']}", use_container_width=True):
            st.session_state.review_result = run["run_id"]
            st.rerun()


def _render_report(snapshot) -> None:
    values = snapshot.values
    st.subheader("Executive report")
    sql = values.get("report_sql", "")
    report_path = values.get("report_path")
    if not report_path or not Path(report_path).exists():
        st.info("The report is not available yet.")
        return
    with st.expander("Report SQL · Edit and run", expanded=False):
        custom_sql = st.text_area("Read-only DuckDB SQL", value=sql, height=120, key="custom_sql")
        if st.button("Run custom SQL", type="primary"):
            try:
                from agents.reporter import _safe_query
                gold_path = values["gold_paths"][0]
                result = _safe_query(custom_sql, pd.read_parquet(gold_path))
                st.session_state.custom_result = result
                st.session_state.custom_sql_text = custom_sql
                st.toast("Query completed", icon="✅")
            except Exception as exc:
                st.error(f"Query rejected: {exc}")
        st.code(st.session_state.get("custom_sql_text", sql), language="sql")
    if isinstance(st.session_state.get("custom_result"), pd.DataFrame):
        custom = st.session_state.custom_result
        st.dataframe(custom, use_container_width=True, hide_index=True)
        if st.button("Quick chart", key="quick_chart"):
            numeric = custom.select_dtypes(include="number").columns.tolist()
            if numeric:
                st.plotly_chart(px.bar(custom, y=numeric[0]), use_container_width=True)
            else:
                st.info("The result has no numeric columns to chart.")
    report_tab, raw_tab, summary_tab = st.tabs(["Report", "Raw Data", "Pipeline Summary"])
    with report_tab:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        components.html(Path(report_path).read_text(encoding="utf-8"), height=720, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with raw_tab:
        result_data = values.get("report_result_df", [])
        if isinstance(result_data, list):
            result = pd.DataFrame(result_data)
            st.dataframe(result, use_container_width=True, hide_index=True)
    with summary_tab:
        st.json(_safe_json({key: value for key, value in values.items() if key.endswith("_validation") or key.endswith("_paths")}))
    result_data = values.get("report_result_df", [])
    result_df = pd.DataFrame(result_data) if isinstance(result_data, list) else pd.DataFrame()
    csv = result_df.to_csv(index=False) if isinstance(result_df, pd.DataFrame) else ""
    log_json = json.dumps(get_stage_events(values["run_id"]), indent=2, default=str)
    downloads = st.columns(3)
    downloads[0].download_button("Download HTML", Path(report_path).read_bytes(), file_name=Path(report_path).name, mime="text/html")
    downloads[1].download_button("Download CSV", csv, file_name="report_results.csv", mime="text/csv")
    downloads[2].download_button("Download pipeline log", log_json, file_name="pipeline_log.json", mime="application/json")


def _reset() -> None:
    st.session_state.run_id = None
    st.session_state.files = {}
    st.session_state.removed_files = set()
    st.session_state.review_result = None
    st.session_state.custom_result = None
    st.session_state.pop("intent", None)
    st.toast("Session reset", icon="🔄")
    st.rerun()


def _reload_pipeline_agents() -> None:
    from agents import bronze_agent, gold_agent, silver_agent, validator
    from core import quality
    from graph import nodes

    for module in (quality, bronze_agent, silver_agent, gold_agent, validator, nodes):
        importlib.reload(module)


def _rewind_pipeline(target: int) -> None:
    if target == 0:
        _reset()
    graph = get_pipeline()
    config = _config(st.session_state.run_id)
    predecessor = {
        1: "profile_node",
        2: "profile_node",
        3: "bronze_hitl_node",
        4: "bronze_validate_node",
        5: "silver_hitl_node",
        6: "silver_validate_node",
        7: "gold_validate_node",
    }.get(target)
    approvals = {
        1: [], 2: [], 3: ["bronze"], 4: ["bronze"],
        5: ["bronze", "silver"], 6: ["bronze", "silver"],
        7: ["bronze", "silver", "gold"],
    }.get(target, [])
    if not predecessor:
        return
    try:
        _reload_pipeline_agents()
        graph.update_state(config, {"approved_layers": approvals, "retry_counts": {}, "error": None}, as_node=predecessor)
        with st.spinner("Rewinding to the selected checkpoint…"):
            graph.invoke(None, config)
            current = graph.get_state(config)
            _advance_auto_approvals(graph, config, bool(current.values.get("auto_approve", False)))
        st.session_state.pop("rewind_confirmation", None)
        st.toast("Pipeline rewound", icon="🔄")
        st.rerun()
    except Exception as exc:
        st.error(f"Unable to rewind this run: {exc}")


left, main, right = st.columns([1.2, 3.5, 1.3], gap="medium")
with main:
    st.markdown('<section class="hero"><div class="idamp-eyebrow">Intent-driven agentic medallion pipeline</div><h1>Make the data explain itself.</h1><p>Move from raw CSVs to reviewed, decision-ready insight with transparent mappings at every layer.</p></section>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    f1.markdown('<div class="feature"><b>Intent-led</b><span>Transformations follow your business question.</span></div>', unsafe_allow_html=True)
    f2.markdown('<div class="feature"><b>Human-approved</b><span>Review each mapping before data changes.</span></div>', unsafe_allow_html=True)
    f3.markdown('<div class="feature"><b>Traceable</b><span>Every stage records quality and lineage.</span></div>', unsafe_allow_html=True)

snapshot_graph, snapshot = _current_snapshot()
active = _step_index(snapshot)
render_stepper(active, _completed_steps(snapshot))
if st.session_state.get("rewind_confirmation") is not None:
    target = int(st.session_state.rewind_confirmation)
    st.warning(f"Rewind this run to {STEPS[target]}? Later layer approvals and outputs will be regenerated.")
    confirm_col, cancel_col, _ = st.columns([1, 1, 6])
    if confirm_col.button("Confirm rewind", type="primary", key="confirm_rewind"):
        _rewind_pipeline(target)
    if cancel_col.button("Cancel", key="cancel_rewind"):
        st.session_state.pop("rewind_confirmation", None)
        st.rerun()

with left:
    st.markdown('<div class="idamp-eyebrow">Workspace</div>', unsafe_allow_html=True)
    active_run = st.session_state.run_id
    if not active_run:
        uploads = st.file_uploader("Add source CSVs", type=["csv"], accept_multiple_files=True, key="source_upload")
        incoming = {file.name: file for file in uploads or []}
        for name, file in incoming.items():
            if name not in st.session_state.removed_files:
                st.session_state.files.setdefault(name, file.getvalue())
        for name in list(st.session_state.files):
            item_col, remove_col = st.columns([4, 1])
            item_col.caption(f"CSV · {name} · {len(st.session_state.files[name]):,} bytes")
            if remove_col.button("×", key=f"remove_{name}", help=f"Remove {name}"):
                del st.session_state.files[name]
                st.session_state.removed_files.add(name)
                st.rerun()
        for suggestion in ["Analyse Q1 sales by region", "Find monthly revenue trends", "Compare product categories", "Summarise customer retention"]:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.intent = suggestion
                st.rerun()
        intent = st.text_area("Business intent", key="intent", height=100, max_chars=500, placeholder="e.g. Analyse Q1 sales by region")
        st.caption(f"{len(intent)} / 500")
        start_disabled = not st.session_state.files or not intent.strip()
        if st.button("Start pipeline", type="primary", disabled=start_disabled, use_container_width=True):
            run_id = str(uuid.uuid4())
            run_dir = LANDING_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            file_info = []
            for name, content in st.session_state.files.items():
                path = run_dir / Path(name).name
                path.write_bytes(content)
                file_info.append({"name": Path(name).name, "path": str(path)})
            start_run(run_id, intent.strip())
            state = {
                "run_id": run_id,
                "business_intent": intent.strip(),
                "uploaded_files_info": file_info,
                "approved_layers": [],
                "retry_counts": {},
                "strict_validation": st.session_state.get("strict_validation", True),
                "auto_approve": st.session_state.get("auto_approve", False),
                "max_gold_rows": st.session_state.get("max_gold_rows", 8),
                "sttm_hint": "",
                "error": None,
            }
            st.session_state.run_id = run_id
            try:
                with st.spinner("Profiling sources and generating the Bronze mapping…"):
                    graph = get_pipeline()
                    config = _config(run_id)
                    graph.invoke(state, config)
                    _advance_auto_approvals(graph, config, state["auto_approve"])
                st.toast("Pipeline started", icon="✅")
                st.rerun()
            except Exception as exc:
                finish_run(run_id, "failed", str(exc))
                st.error(str(exc))
    else:
        values = snapshot.values if snapshot else {}
        st.markdown(f'<div class="quote">{values.get("business_intent", "")}</div>', unsafe_allow_html=True)
        st.caption(f"Run ID · {active_run[:8]}")
        started = datetime.fromisoformat(get_recent_runs(20)[0]["started_at"].replace("Z", "+00:00")) if get_recent_runs(20) else datetime.now(timezone.utc)
        elapsed = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
        st.metric("Elapsed", f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        if values.get("error"):
            st.error(values["error"])
        elif values.get("report_path"):
            st.success("Pipeline healthy · report ready")
        else:
            st.info("Pipeline active · waiting for mapping review" if snapshot and snapshot.next else "Pipeline processing")
        for item in values.get("uploaded_files_info", []):
            st.caption(f"CSV · {item['name']}")
        if st.button("Reset session", use_container_width=True):
            st.session_state.confirm_reset = True
        if st.session_state.get("confirm_reset"):
            st.warning("Reset this session view? Audit history and generated files will remain.")
            yes, no = st.columns(2)
            if yes.button("Confirm", key="confirm_reset_yes"):
                st.session_state.confirm_reset = False
                _reset()
            if no.button("Cancel", key="confirm_reset_no"):
                st.session_state.confirm_reset = False
                st.rerun()
        export_values = _safe_json(values)
        st.download_button("Export session JSON", json.dumps(export_values, indent=2, default=str), file_name=f"idamp_{active_run[:8]}.json", mime="application/json", use_container_width=True)
    with st.expander("Settings", expanded=False):
        st.toggle("Auto-approve generated mappings", key="auto_approve", value=False)
        st.toggle("Strict mapping validation", key="strict_validation", value=True)
        st.slider("Maximum Gold output rows", 4, 12, 8, key="max_gold_rows")
    st.divider()
    light = st.toggle("Light mode", key="light_mode")
    if light != st.session_state.get("_last_mode", False):
        st.session_state._last_mode = light
        st.rerun()

with main:
    if snapshot is None:
        st.markdown('<div class="glass"><span class="idamp-eyebrow">Data flow</span><p style="font:15px Consolas;color:#9cb2b7">CSV → PROFILE → BRONZE → SILVER → GOLD → EXECUTIVE REPORT</p></div>', unsafe_allow_html=True)
        st.info("Upload one or more CSV files and describe the business question to begin.")
    else:
        values = snapshot.values
        if values.get("error"):
            st.error(f"Pipeline stopped: {values['error']}")
            if st.button("Retry from current checkpoint"):
                try:
                    _reload_pipeline_agents()
                    graph = get_pipeline()
                    config = _config(values["run_id"])
                    failed_stage = str(values["error"]).split(":", 1)[0]
                    layer = failed_stage.removesuffix("_validate") if failed_stage.endswith("_validate") else ""
                    if layer and f"{layer}_paths" not in values:
                        graph.update_state(
                            config,
                            {"error": None, "current_step": f"{layer}_load"},
                            as_node=f"{layer}_hitl_node",
                        )
                        graph.invoke(None, config)
                    else:
                        retry_failed_stage(graph, config, values)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        waiting_layer = None
        if snapshot.next:
            node = snapshot.next[0]
            if node.endswith("_hitl_node"):
                waiting_layer = node.removesuffix("_hitl_node")
        if waiting_layer:
            _mapping_editor(waiting_layer, values.get(f"{waiting_layer}_sttm", []), values["run_id"], values.get("strict_validation", True))
        for layer in ("bronze", "silver", "gold"):
            if values.get(f"{layer}_paths"):
                _render_layer_results(layer, snapshot)
        if values.get("report_path"):
            st.balloons()
            _render_report(snapshot)
        elif values.get("gold_validation", {}).get("valid"):
            st.info("Gold validation passed. Generating the executive report…")
        if snapshot.next == () and not values.get("report_path") and not values.get("error"):
            failures = [values.get(f"{layer}_validation", {}).get("failures", []) for layer in ("bronze", "silver", "gold")]
            messages = [message for group in failures for message in group]
            if messages:
                st.error("Pipeline ended after validation retries: " + "; ".join(messages))
                failed_layer = next((layer for layer in ("gold", "silver", "bronze") if values.get(f"{layer}_validation") and not values[f"{layer}_validation"].get("valid")), None)
                if failed_layer and st.button(f"Retry {failed_layer.title()} load from approved STTM", key=f"retry_{failed_layer}_load"):
                    try:
                        _reload_pipeline_agents()
                        graph = get_pipeline()
                        config = _config(values["run_id"])
                        retries = dict(values.get("retry_counts", {}))
                        retries[failed_layer] = 0
                        graph.update_state(
                            config,
                            {"retry_counts": retries, "error": None, "current_step": f"{failed_layer}_load"},
                            as_node=f"{failed_layer}_hitl_node",
                        )
                        graph.invoke(None, config)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Unable to retry {failed_layer} load: {exc}")
                finish_run(values["run_id"], "failed", "; ".join(messages))
        elif values.get("report_path"):
            finish_run(values["run_id"], "completed")

with right:
    _render_audit(st.session_state.run_id)
    if st.session_state.get("review_result"):
        run_id = st.session_state.review_result
        report_path = PROJECT_ROOT / "reports" / f"report_{run_id[:8]}.html"
        if report_path.exists():
            st.markdown("### Run review")
            components.html(report_path.read_text(encoding="utf-8"), height=400, scrolling=True)
        else:
            st.caption("This run has no completed report.")

components.html(
    """<script>
    document.addEventListener('keydown', function(event) {
      if (event.ctrlKey && event.key === 'Enter') {
        const buttons = Array.from(window.parent.document.querySelectorAll('button'));
        const target = buttons.find(button => button.innerText.includes('Start pipeline') || button.innerText.includes('Approve mapping'));
        if (target && !target.disabled) target.click();
      }
      if (event.key.toLowerCase() === 'r' && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
        const target = Array.from(window.parent.document.querySelectorAll('button')).find(button => button.innerText.includes('Reset session'));
        if (target) target.click();
      }
      if (event.key === 'ArrowRight' && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
        const target = Array.from(window.parent.document.querySelectorAll('button')).find(button => button.innerText.includes('Approve mapping'));
        if (target && !target.disabled) target.click();
      }
    });
    </script>""",
    height=0,
)
