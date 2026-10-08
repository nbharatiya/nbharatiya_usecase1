"""Focused tests for the deterministic IDAMP pipeline components."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agents import auditor, bronze_agent, gold_agent, reporter, silver_agent
from agents.sttm_generator import parse_json_array
from agents.validator import validate_layer
from core.quality import quality_score
from graph import nodes as graph_nodes
from graph.pipeline import get_pipeline, retry_failed_stage


class PipelineComponentTests(unittest.TestCase):
    def test_sttm_parser_recovers_truncated_array(self) -> None:
        parsed = parse_json_array('[{"source_column":"amount","target_column":"revenue"}, {"source_column":"region"')
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["target_column"], "revenue")

    def test_quality_score_uses_source_specific_columns(self) -> None:
        frame = pd.DataFrame({
            "_source_file": ["sales.csv", "products.csv"],
            "transaction_id": ["t-1", None],
            "product_name": [None, "Widget"],
        })
        self.assertEqual(quality_score(frame), 100.0)

    def test_silver_preserves_null_dedup_keys_before_filling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bronze_path = root / "bronze.parquet"
            pd.DataFrame({"transaction_id": ["t-1", None, None, "t-1"], "amount": [1, 2, 3, 4]}).to_parquet(bronze_path)
            mappings = [{
                "source_column": "transaction_id",
                "target_column": "transaction_id",
                "transformation": "fill_nulls('UNKNOWN_TXN')",
                "data_type": "str",
                "notes": "dedup_key=true",
            }]
            with patch.object(silver_agent, "SILVER_DIR", root / "silver"):
                output = silver_agent.load_silver("test-run", [str(bronze_path)], mappings)
            result = pd.read_parquet(output)
            self.assertEqual(len(result), 3)
            self.assertEqual(int(result["transaction_id"].eq("UNKNOWN_TXN").sum()), 2)

    def test_bronze_parses_mixed_date_formats(self) -> None:
        values = pd.Series(["01/15/2025", "16-01-2025"])
        converted = bronze_agent._cast(values, "datetime64")
        self.assertEqual(int(converted.notna().sum()), 2)

    def test_gold_deduplicates_repeated_aggregate_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_path = root / "silver.parquet"
            pd.DataFrame({
                "region": ["North", "South"],
                "amount": [10.0, 20.0],
                "transaction_date": pd.to_datetime(["2025-01-10", "2025-02-10"], utc=True),
            }).to_parquet(silver_path)
            aggregate = {
                "source_column": "amount",
                "target_column": "avg_amount",
                "transformation": "FILTER transaction_date >= '2025-01-01' AND transaction_date < '2025-04-01' THEN AVG(amount)",
                "data_type": "float64",
                "notes": "",
            }
            mappings = [
                {"source_column": "region", "target_column": "region", "transformation": "GROUP BY region", "data_type": "str", "notes": ""},
                aggregate,
                aggregate.copy(),
            ]
            with patch.object(gold_agent, "GOLD_DIR", root / "gold"):
                output = gold_agent.load_gold("test-run", [str(silver_path)], mappings)
            result = pd.read_parquet(output)
            self.assertEqual(result.columns.tolist(), ["pk_gold_id", "region", "avg_amount"])
            self.assertEqual(result["avg_amount"].tolist(), [10.0, 20.0])

    def test_gold_joins_region_and_uses_latest_q1_without_explicit_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_path = root / "silver.parquet"
            pd.DataFrame({
                "pk_silver_id": [1, 2, 3, 4, 5],
                "_source_file": ["sales.csv"] * 3 + ["stores.csv"] * 2,
                "store_id_2": ["s1", "s1", "s2", "s1", "s2"],
                "transaction_date": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15", None, None], utc=True),
                "total_amount": [100.0, 50.0, 200.0, None, None],
                "region": [None, None, None, "North", "South"],
            }).to_parquet(silver_path)
            mappings = [
                {"source_column": "region", "target_column": "region", "transformation": "GROUP BY region", "data_type": "str", "notes": ""},
                {"source_column": "total_amount", "target_column": "total_q1", "transformation": "FILTER transaction_date >= '2025-01-01' AND transaction_date < '2025-04-01' THEN SUM(total_amount)", "data_type": "float64", "notes": ""},
            ]
            with patch.object(gold_agent, "GOLD_DIR", root / "gold"):
                output = gold_agent.load_gold(
                    "test-run",
                    [str(silver_path)],
                    mappings,
                    business_intent="Analyse Q1 sales by region",
                )
            result = pd.read_parquet(output).set_index("region")
            self.assertEqual(result["total_q1"].to_dict(), {"North": 150.0, "South": 200.0})

    def test_validate_node_preserves_load_error_when_output_missing(self) -> None:
        result = graph_nodes.gold_validate_node({
            "run_id": "test-run",
            "error": "gold_load: duplicate output columns",
        })
        self.assertEqual(result, {"current_step": "error"})

    def test_retry_maps_missing_gold_output_to_gold_load(self) -> None:
        class FakeGraph:
            def __init__(self):
                self.predecessor = None
                self.updates = None
                self.invoked = False

            def update_state(self, config, values, as_node=None):
                self.predecessor = as_node
                self.updates = values

            def invoke(self, value, config):
                self.invoked = True

        graph = FakeGraph()
        retry_failed_stage(graph, {"configurable": {"thread_id": "test-run"}}, {
            "run_id": "test-run",
            "error": "gold_validate: 'gold_paths'",
        })
        self.assertEqual(graph.predecessor, "gold_hitl_node")
        self.assertIsNone(graph.updates["error"])
        self.assertTrue(graph.invoked)

    def test_medallion_transformations_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sales.csv"
            pd.DataFrame({
                "id": [1, 2, 3],
                "region": ["North", "South", None],
                "amount": [10.0, 20.0, 30.0],
            }).to_csv(source, index=False)
            bronze_mappings = [
                {"source_column": "id", "target_column": "id", "transformation": "cast", "data_type": "int64", "notes": ""},
                {"source_column": "region", "target_column": "region", "transformation": "rename", "data_type": "str", "notes": ""},
                {"source_column": "amount", "target_column": "revenue", "transformation": "rename", "data_type": "float64", "notes": ""},
            ]
            with patch.object(bronze_agent, "BRONZE_DIR", root / "bronze"), patch.object(silver_agent, "SILVER_DIR", root / "silver"), patch.object(gold_agent, "GOLD_DIR", root / "gold"):
                bronze_path = bronze_agent.load_bronze("run-test", {"path": str(source)}, bronze_mappings)
                bronze_frame = pd.read_parquet(bronze_path)
                bronze_validation = validate_layer("bronze", bronze_frame, 3)
                self.assertTrue(bronze_validation["valid"], bronze_validation["failures"])

                silver_mappings = [
                    {"source_column": "id", "target_column": "id", "transformation": "", "data_type": "int64", "notes": "dedup_key=true"},
                    {"source_column": "region", "target_column": "region", "transformation": "fill=unknown", "data_type": "str", "notes": ""},
                    {"source_column": "amount", "target_column": "revenue", "transformation": "fill=0", "data_type": "float64", "notes": ""},
                ]
                silver_path = silver_agent.load_silver("run-test", [str(bronze_path)], silver_mappings)
                silver_frame = pd.read_parquet(silver_path)
                silver_validation = validate_layer("silver", silver_frame)
                self.assertTrue(silver_validation["valid"], silver_validation["failures"])
                self.assertTrue(silver_frame["pk_silver_id"].is_unique)
                self.assertEqual(len(silver_frame), 3)

                gold_mappings = [
                    {"source_column": "region", "target_column": "region", "transformation": "GROUP BY region", "data_type": "str", "notes": ""},
                    {"source_column": "revenue", "target_column": "total_revenue", "transformation": "SUM(revenue) GROUP BY region", "data_type": "float64", "notes": ""},
                ]
                gold_path = gold_agent.load_gold("run-test", [str(silver_path)], gold_mappings)
                gold_frame = pd.read_parquet(gold_path)
                gold_validation = validate_layer("gold", gold_frame)
                self.assertTrue(gold_validation["valid"], gold_validation["failures"])
                self.assertEqual(gold_frame["total_revenue"].sum(), 60.0)
                self.assertGreaterEqual(quality_score(silver_frame), 70.0)

    def test_audit_trail_and_read_only_report_sql(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(auditor, "AUDIT_DB", Path(directory) / "audit.db"):
                auditor.start_run("run-1", "sales by region")
                auditor.log_stage_event("run-1", "profile", "completed", rows_out=3, quality_score=95.0)
                self.assertEqual(auditor.get_stage_events("run-1")[0]["stage"], "profile")
                self.assertEqual(auditor.get_recent_runs(1)[0]["run_id"], "run-1")
            data = pd.DataFrame({"revenue": [2, 5]})
            self.assertEqual(reporter._safe_query("SELECT SUM(revenue) AS total FROM gold_data", data).iloc[0, 0], 7)
            with self.assertRaises(ValueError):
                reporter._safe_query("DELETE FROM gold_data", data)

    def test_graph_pauses_for_bronze_approval_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame({"region": ["North", "South"], "revenue": [10, 20]}).to_csv(source, index=False)
            mapping = [
                {"source_column": "region", "target_column": "region", "transformation": "", "data_type": "str", "notes": ""},
                {"source_column": "revenue", "target_column": "revenue", "transformation": "", "data_type": "float64", "notes": ""},
            ]

            def fake_generate(layer, profile, intent, hint=""):
                if layer == "gold":
                    return [
                        {"source_column": "region", "target_column": "region", "transformation": "GROUP BY region", "data_type": "str", "notes": ""},
                        {"source_column": "revenue", "target_column": "total_revenue", "transformation": "SUM(revenue) GROUP BY region", "data_type": "float64", "notes": ""},
                    ]
                return mapping

            run_id = f"graph-test-{Path(directory).name}"
            config = {"configurable": {"thread_id": run_id}}
            initial = {
                "run_id": run_id,
                "business_intent": "Revenue by region",
                "uploaded_files_info": [{"name": source.name, "path": str(source)}],
                "approved_layers": [],
                "retry_counts": {},
                "sttm_hint": "",
            }
            graph = get_pipeline()
            with (
                patch("graph.nodes.generate_sttm", side_effect=fake_generate),
                patch("graph.nodes.save_sttm"),
                patch("graph.nodes.log_stage_event"),
                patch("graph.nodes.create_report", return_value={"report_path": "report.html", "report_sql": "SELECT * FROM gold_data", "report_result_df": pd.DataFrame({"region": ["North"], "total_revenue": [10]})}),
                patch.object(bronze_agent, "BRONZE_DIR", root / "bronze"),
                patch.object(silver_agent, "SILVER_DIR", root / "silver"),
                patch.object(gold_agent, "GOLD_DIR", root / "gold"),
            ):
                graph.invoke(initial, config)
                paused = graph.get_state(config)
                self.assertEqual(paused.next, ("bronze_hitl_node",))
                graph.update_state(config, {"approved_layers": ["bronze"]})
                graph.invoke(None, config)
                resumed = graph.get_state(config)
                self.assertEqual(resumed.next, ("silver_hitl_node",))
                self.assertTrue(resumed.values["bronze_validation"]["valid"])
                graph.update_state(config, {"approved_layers": ["bronze", "silver"]})
                graph.invoke(None, config)
                resumed = graph.get_state(config)
                self.assertEqual(resumed.next, ("gold_hitl_node",))
                self.assertTrue(resumed.values["silver_validation"]["valid"])
                graph.update_state(config, {"approved_layers": ["bronze", "silver", "gold"]})
                graph.invoke(None, config)
                completed = graph.get_state(config)
                self.assertEqual(completed.next, ())
                self.assertTrue(completed.values["gold_validation"]["valid"])
                self.assertEqual(completed.values["report_path"], "report.html")
                graph.update_state(config, {"approved_layers": ["bronze"], "retry_counts": {}, "error": None}, as_node="bronze_validate_node")
                graph.invoke(None, config)
                rewound = graph.get_state(config)
                self.assertEqual(rewound.next, ("silver_hitl_node",))

    def test_retry_replays_failed_bronze_sttm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame({"region": ["North"], "revenue": [10]}).to_csv(source, index=False)
            mapping = [{"source_column": "region", "target_column": "region", "transformation": "", "data_type": "str", "notes": ""}]
            run_id = f"graph-retry-{root.name}"
            config = {"configurable": {"thread_id": run_id}}
            initial = {
                "run_id": run_id,
                "business_intent": "Revenue by region",
                "uploaded_files_info": [{"name": source.name, "path": str(source)}],
                "approved_layers": [],
                "retry_counts": {},
                "sttm_hint": "",
            }
            graph = get_pipeline()
            with (
                patch("graph.nodes.generate_sttm", side_effect=[RuntimeError("transient LLM error"), mapping]),
                patch("graph.nodes.save_sttm"),
                patch("graph.nodes.log_stage_event"),
                patch.object(graph_nodes, "LANDING_DIR", root / "landing"),
            ):
                graph.invoke(initial, config)
                failed = graph.get_state(config)
                self.assertTrue(failed.values["error"].startswith("bronze_sttm:"))
                retry_failed_stage(graph, config, failed.values)
                retried = graph.get_state(config)
                self.assertEqual(retried.next, ("bronze_hitl_node",))
                self.assertIsNone(retried.values.get("error"))
                self.assertEqual(retried.values["bronze_sttm"], mapping)


if __name__ == "__main__":
    unittest.main()
