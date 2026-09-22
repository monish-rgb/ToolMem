"""Hermetic unit tests for sales dataset analysis benchmark harness."""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from toolatlas.llm_ab_data_analysis import (
    generate_dataset,
    verify_analysis_report,
    seed_analysis_memory,
    TASK_SUMMARY,
)
from toolatlas.memory_server import create_memory_server
from toolatlas.readonly_benchmark import _value
from mcp import Client


def test_generate_dataset_deterministic(tmp_path: Path):
    gt1 = generate_dataset(tmp_path, n_rows=50, seed=42)
    csv_file = tmp_path / "sales_analytics.csv"
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 50
        assert "transaction_id" in rows[0]
        assert "revenue" in rows[0]
        assert "status" in rows[0]

    assert gt1["total_rows"] == 50
    assert gt1["completed_revenue"] > 0
    assert gt1["top_region"] in ("North America", "Europe", "Asia-Pacific", "Latin America")
    assert gt1["refunded_count"] >= 0
    assert 5.0 <= gt1["avg_discount_rate"] <= 30.0


def test_verify_analysis_report_passes(tmp_path: Path):
    gt = {
        "completed_revenue": 125450.00,
        "top_region": "North America",
        "refunded_count": 8,
        "avg_discount_rate": 15.2,
    }
    report_content = """# Executive Sales Analytics Report

## Key Performance Indicators
- **Total Completed Revenue:** $125,450.00
- **Top Performing Region:** North America (leading sales across all quarters)
- **Refunded Transactions:** 8 orders were returned or canceled.
- **Average Discount Rate:** 15.2% across enterprise and retail accounts.

## Analysis Summary
Performance is strong in North America, with low refund rates maintaining profitability.
"""
    (tmp_path / "report.md").write_text(report_content, encoding="utf-8")

    passed, msg = verify_analysis_report(tmp_path, gt)
    assert passed is True
    assert "All 4 metrics verified successfully" in msg


def test_verify_analysis_report_fails_missing_file(tmp_path: Path):
    gt = {
        "completed_revenue": 100000.0,
        "top_region": "Europe",
        "refunded_count": 5,
        "avg_discount_rate": 10.0,
    }
    passed, msg = verify_analysis_report(tmp_path, gt)
    assert passed is False
    assert "report.md does not exist" in msg


def test_verify_analysis_report_fails_wrong_revenue(tmp_path: Path):
    gt = {
        "completed_revenue": 150000.00,
        "top_region": "Asia-Pacific",
        "refunded_count": 10,
        "avg_discount_rate": 12.0,
    }
    report_content = """# Report
Revenue: $20,000.00
Top Region: Asia-Pacific
Refunds: 10
Average Discount: 12.0%
This is a summary analysis.
"""
    (tmp_path / "report.md").write_text(report_content, encoding="utf-8")

    passed, msg = verify_analysis_report(tmp_path, gt)
    assert passed is False
    assert "completed_revenue mismatch" in msg


@pytest.mark.asyncio
async def test_seed_analysis_memory(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    server = create_memory_server(db_path)

    async with Client(server) as memory:
        await seed_analysis_memory(memory)
        guidance = _value(
            await memory.call_tool(
                "get_guidance",
                {"task": TASK_SUMMARY, "top_k": 2, "read_budget": 6, "token_budget": 384},
            )
        )
        assert guidance is not None
        playbook = guidance.get("playbook", [])
        assert len(playbook) >= 2
        tool_names = [step["tool"] for step in playbook]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
