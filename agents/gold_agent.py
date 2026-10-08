"""Apply an intentionally small, auditable Gold transformation grammar."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core.config import GOLD_DIR

_AGGREGATION = re.compile(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(\s*([\w.*]+)\s*\)(?:\s+AS\s+([\w]+))?", re.IGNORECASE)
_GROUP_BY = re.compile(r"GROUP\s+BY\s+([\w]+(?:\s*,\s*[\w]+)*)", re.IGNORECASE)
_DATE_FILTER = re.compile(r"(?:FILTER\s+)?([\w]+)\s*(>=|>|<=|<|=)\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?", re.IGNORECASE)


def _apply_date_filters(
    frame: pd.DataFrame,
    transformations: str,
    allow_latest_year_fallback: bool,
) -> pd.DataFrame:
    filters = _DATE_FILTER.findall(transformations)

    def apply(rules: list[tuple[str, str, str]]) -> pd.DataFrame:
        result = frame
        for column, operator, value in rules:
            if column not in result.columns:
                continue
            dates = pd.to_datetime(result[column], errors="coerce", utc=True, format="mixed")
            boundary = pd.Timestamp(value, tz="UTC")
            mask = {
                ">=": dates >= boundary,
                ">": dates > boundary,
                "<=": dates <= boundary,
                "<": dates < boundary,
                "=": dates == boundary,
            }[operator]
            result = result.loc[mask].copy()
        return result

    filtered = apply(filters)
    requested_years = {int(value[:4]) for _, _, value in filters}
    if filtered.empty and allow_latest_year_fallback and len(requested_years) == 1:
        date_columns = list(dict.fromkeys(column for column, _, _ in filters if column in frame.columns))
        observed_years = [
            int(values.max().year)
            for column in date_columns
            if (values := pd.to_datetime(frame[column], errors="coerce", utc=True, format="mixed").dropna()).size
        ]
        if observed_years:
            requested_year = next(iter(requested_years))
            latest_year = max(observed_years)
            if latest_year < requested_year:
                shifted_filters = []
                for column, operator, value in filters:
                    shifted = pd.Timestamp(value, tz="UTC").replace(year=latest_year)
                    shifted_filters.append((column, operator, shifted.strftime("%Y-%m-%d")))
                filtered = apply(shifted_filters)
    return filtered


def _enrich_fact_from_union(frame: pd.DataFrame, mappings: list[dict[str, object]]) -> pd.DataFrame:
    if "_source_file" not in frame.columns:
        return frame
    groups = [(name, rows.copy()) for name, rows in frame.groupby("_source_file", dropna=False, sort=False)]
    transformations = " ".join(str(item.get("transformation", "")) for item in mappings)
    group_columns = list(dict.fromkeys(
        column.strip()
        for match in _GROUP_BY.finditer(transformations)
        for column in match.group(1).split(",")
        if column.strip()
    ))
    aggregate_sources = {
        source
        for _, source, _ in _AGGREGATION.findall(transformations)
        if source != "*"
    }
    date_sources = list(dict.fromkeys(column for column, _, _ in _DATE_FILTER.findall(transformations)))
    if not groups or not (aggregate_sources or date_sources):
        return frame

    def fact_score(group_index: int) -> tuple[int, int, int, int, int]:
        candidate = groups[group_index][1]
        measure_cardinality = sum(
            int(candidate[source].nunique(dropna=True))
            for source in aggregate_sources
            if source in candidate.columns
        )
        date_cardinality = sum(
            int(candidate[column].nunique(dropna=True))
            for column in date_sources
            if column in candidate.columns
        )
        measure_coverage = sum(
            int(candidate[source].notna().sum())
            for source in aggregate_sources
            if source in candidate.columns
        )
        date_coverage = sum(
            int(candidate[column].notna().sum())
            for column in date_sources
            if column in candidate.columns
        )
        return measure_cardinality, date_cardinality, measure_coverage, date_coverage, len(candidate)

    fact_index = max(
        range(len(groups)),
        key=fact_score,
    )
    fact = groups[fact_index][1]
    missing_groups = [
        column
        for column in group_columns
        if column not in fact.columns or not fact[column].notna().any()
    ]
    for index, (_, dimension) in enumerate(groups):
        if index == fact_index or not missing_groups:
            continue
        requested = [column for column in missing_groups if column in dimension.columns and dimension[column].notna().any()]
        if not requested:
            continue
        keys = [
            column
            for column in fact.columns
            if re.fullmatch(r"[\w]+_id(?:_\d+)?", str(column))
            and not str(column).startswith("pk_")
            and column in dimension.columns
            and dimension[column].notna().all()
            and dimension[column].is_unique
            and fact[column].notna().any()
        ]
        if not keys:
            continue
        lookup = dimension[list(dict.fromkeys(keys + requested))].drop_duplicates(subset=keys, keep="first")
        joined_names = {column: f"_idamp_dimension_{column}" for column in requested}
        lookup = lookup.rename(columns=joined_names)
        fact = fact.merge(lookup, on=keys, how="left", suffixes=("", "_dimension"))
        for column in requested:
            dimension_column = joined_names[column]
            if column in fact.columns:
                fact[column] = fact[column].combine_first(fact[dimension_column])
            else:
                fact[column] = fact[dimension_column]
            fact = fact.drop(columns=[dimension_column])
        missing_groups = [column for column in missing_groups if not fact[column].notna().any()]
    if missing_groups:
        raise ValueError(f"Gold group-by columns could not be joined from Silver source data: {missing_groups}.")
    return fact


def load_gold(
    run_id: str,
    silver_paths: list[str],
    mappings: list[dict[str, object]],
    max_rows: int = 8,
    business_intent: str = "",
) -> Path:
    inputs = [pd.read_parquet(path) for path in silver_paths]
    if not inputs:
        raise ValueError("No Silver Parquet input was found.")
    frame = inputs[0]
    for next_frame in inputs[1:]:
        shared = [column for column in frame.columns if column in next_frame.columns]
        if not shared:
            raise ValueError("Silver tables have no shared columns for an outer join.")
        frame = frame.merge(next_frame, on=shared, how="outer", suffixes=("", "_right"))

    transformations = " ".join(str(item.get("transformation", "")) for item in mappings)
    frame = _enrich_fact_from_union(frame, mappings)
    frame = _apply_date_filters(
        frame,
        transformations,
        allow_latest_year_fallback=not bool(re.search(r"\b(?:19|20)\d{2}\b", business_intent)),
    )
    groups: list[str] = []
    for match in _GROUP_BY.finditer(transformations):
        groups.extend(column.strip() for column in match.group(1).split(",") if column.strip())
    groups = list(dict.fromkeys(column for column in groups if column in frame.columns))

    aggregates: list[tuple[str, str, str]] = []
    aggregate_keys: set[tuple[str, str, str]] = set()
    targets: dict[str, tuple[str, str, str]] = {}
    for mapping in mappings:
        expression = str(mapping.get("transformation", ""))
        for function, source, alias in _AGGREGATION.findall(expression):
            target = str(mapping.get("target_column") or alias or f"{function.lower()}_{source.replace('.', '_')}")
            if source == "*" or source in frame.columns:
                aggregate_key = (function.upper(), source, target)
                if aggregate_key in aggregate_keys:
                    continue
                previous = targets.get(target)
                if previous is not None and previous != aggregate_key:
                    raise ValueError(f"Gold STTM assigns conflicting aggregations to target column {target!r}.")
                aggregate_keys.add(aggregate_key)
                targets[target] = aggregate_key
                aggregates.append(aggregate_key)
    if aggregates:
        result = pd.DataFrame(index=frame.groupby(groups, dropna=False).size().index) if groups else pd.DataFrame(index=[0])
        if groups:
            result = result.reset_index().drop(columns=["size"], errors="ignore")
        else:
            result = pd.DataFrame([{}])
        for function, source, target in aggregates:
            if source == "*":
                grouped = frame.groupby(groups, dropna=False).size() if groups else pd.Series([len(frame)])
            else:
                grouped_source = frame.groupby(groups, dropna=False)[source] if groups else frame[source]
                operation = {"SUM": "sum", "COUNT": "count", "AVG": "mean", "MIN": "min", "MAX": "max"}[function]
                grouped = getattr(grouped_source, operation)()
            values = grouped.reset_index(drop=True) if groups else pd.Series([grouped])
            result[target] = values.to_numpy()
        if set(groups).intersection(targets):
            raise ValueError("Gold STTM reuses a group-by column name for an aggregate target.")
        selected = [column for column in groups if column in result.columns] + [target for _, _, target in aggregates]
        result = result[selected]
    else:
        requested = []
        for mapping in mappings:
            source = str(mapping.get("source_column", ""))
            target = str(mapping.get("target_column", source))
            if source in frame.columns:
                requested.append((source, target))
        if requested:
            result = frame[[source for source, _ in requested]].copy()
            result.columns = [target for _, target in requested]
        else:
            result = frame.copy()
        if len(result.columns) > 8:
            result = result.iloc[:, :8]

    result = result.head(max(1, int(max_rows))).reset_index(drop=True)
    result.insert(0, "pk_gold_id", range(1, len(result) + 1))
    output_dir = GOLD_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "gold.parquet"
    result.to_parquet(output, index=False)
    return output
