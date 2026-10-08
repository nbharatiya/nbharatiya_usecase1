"""Generate an executive HTML report and safe DuckDB query results."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px

from agents.sttm_generator import generate_report_sql, generate_report_text
from core.config import REPORTS_DIR

_FORBIDDEN_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|PRAGMA|CALL|SET|RESET)\b", re.IGNORECASE)
_EXTERNAL_READ = re.compile(r"\b(read_csv|read_parquet|read_json|parquet_scan|glob)\s*\(", re.IGNORECASE)


def _safe_query(sql: str, frame: pd.DataFrame) -> pd.DataFrame:
    statement = sql.strip().rstrip(";").strip()
    if ";" in statement or not re.match(r"^(SELECT|WITH)\b", statement, re.IGNORECASE):
        raise ValueError("Report SQL must be a single SELECT or WITH query.")
    if _FORBIDDEN_SQL.search(statement) or _EXTERNAL_READ.search(statement):
        raise ValueError("Report SQL contains a disallowed operation.")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("gold_data", frame)
        return connection.execute(statement).df()
    finally:
        connection.close()


def create_report(run_id: str, intent: str, gold_path: str | Path) -> dict[str, Any]:
    gold = pd.read_parquet(gold_path)
    sql = generate_report_sql(intent, [str(column) for column in gold.columns])
    result = _safe_query(sql, gold)
    body = generate_report_text(intent, [str(column) for column in result.columns], result.head(30).to_dict(orient="records"))
    body = re.sub(r"^```(?:html)?\s*|\s*```$", "", body.strip(), flags=re.IGNORECASE)

    chart_html = ""
    numeric = [column for column in result.select_dtypes(include="number").columns if column != "pk_gold_id"]
    dimensions = [column for column in result.columns if column not in numeric and column != "pk_gold_id"]
    if numeric and len(result):
        if dimensions:
            figure = px.bar(result, x=dimensions[0], y=numeric[0], title=f"{numeric[0]} by {dimensions[0]}")
        else:
            figure = px.line(result, y=numeric[0], title=numeric[0])
        chart_html = figure.to_html(full_html=False, include_plotlyjs=True)

    document = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>IDAMP Executive Report</title><style>"
        "body{font:16px/1.6 'Segoe UI',sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#17242b}"
        "h1,h2{color:#123c43}header{border-bottom:1px solid #cbd7d9;padding-bottom:18px}"
        "table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d8e0e2;padding:8px;text-align:left}"
        "</style></head><body><header><p>IDAMP / EXECUTIVE ANALYSIS</p>"
        f"<h1>{html.escape(intent)}</h1><p>Run {html.escape(run_id[:8])}</p></header>"
        f"<main>{body}{chart_html}</main></body></html>"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"report_{run_id[:8]}.html"
    report_path.write_text(document, encoding="utf-8")
    return {"report_path": str(report_path), "report_sql": sql, "report_result_df": result}
