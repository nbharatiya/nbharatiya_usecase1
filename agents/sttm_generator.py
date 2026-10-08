"""LLM-generated STTMs and resilient JSON-array parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import GROQ_MODEL, LLM_TEMPERATURE, MAX_LLM_ATTEMPTS, get_llm_client
from core.retry import retry_call


def parse_json_array(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array, recovering a truncated final object when possible."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()
    attempts = [candidate]
    match = re.search(r"\[[\s\S]*\]", candidate)
    if match:
        attempts.append(match.group(0))
    for value in attempts:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed
        except json.JSONDecodeError:
            pass
    start = candidate.find("[")
    if start >= 0:
        body = candidate[start + 1 :]
        last_object_end = body.rfind("}")
        if last_object_end >= 0:
            recovered = "[" + body[: last_object_end + 1].rstrip().rstrip(",") + "]"
            try:
                parsed = json.loads(recovered)
                if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
    raise ValueError("LLM response did not contain a recoverable JSON array of STTM entries.")


def _chat(system: str, prompt: str) -> str:
    client = get_llm_client()

    def request() -> str:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The LLM returned an empty response.")
        return content

    return retry_call(request, attempts=MAX_LLM_ATTEMPTS)


def generate_sttm(
    layer: str,
    profile: dict[str, Any],
    business_intent: str,
    hint: str = "",
) -> list[dict[str, Any]]:
    columns = [column for file in profile.get("files", []) for column in file.get("columns", [])]
    layer_guidance = {
        "bronze": "Preserve every source column, propose safe renames and source-compatible casts. Include audit columns as additional mapping entries only if useful.",
        "silver": "Specify null-fill strategy and identify deduplication keys in notes as dedup_key=true where appropriate. Preserve useful source attributes.",
        "gold": "Return no more than 8 business-facing outputs, driven by intent. Use transformation expressions such as GROUP BY region, SUM(amount), COUNT(*), AVG(amount), or FILTER date >= 2025-01-01.",
    }
    schema = '[{"source_column":"...","target_column":"...","transformation":"...","data_type":"str|int64|float64|datetime64","notes":"..."}]'
    prompt = (
        f"Business intent: {business_intent}\nLayer: {layer}\nLayer guidance: {layer_guidance[layer]}\n"
        f"Source profile: {json.dumps(columns, ensure_ascii=False)}\nAdditional user guidance: {hint or 'none'}\n"
        f"Return only a JSON array with this schema: {schema}. Use source column names exactly."
    )
    return parse_json_array(_chat("You design auditable source-to-target mappings for data pipelines. Return valid JSON only.", prompt))


def generate_report_text(intent: str, columns: list[str], result_preview: list[dict[str, Any]]) -> str:
    prompt = (
        "Write a concise executive HTML report body (HTML fragment only; no scripts, stylesheets, or markdown). "
        "Use semantic headings and paragraphs, explain the observed results without inventing facts.\n"
        f"Business intent: {intent}\nAvailable result columns: {json.dumps(columns)}\n"
        f"Query result sample: {json.dumps(result_preview, default=str)}"
    )
    return _chat("You are a careful analytics report writer. Make only claims supported by the supplied result.", prompt)


def generate_report_sql(intent: str, columns: list[str]) -> str:
    prompt = (
        "Write one read-only DuckDB SELECT statement using the view gold_data. Do not use DDL, DML, comments, "
        "or multiple statements. Use only the listed columns and a reasonable result limit.\n"
        f"Intent: {intent}\nColumns: {json.dumps(columns)}"
    )
    return _chat("You write safe, simple DuckDB analytical SQL.", prompt).strip().strip("`")
