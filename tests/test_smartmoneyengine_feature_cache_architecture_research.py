"""Tests for SmartMoneyEngine feature cache architecture research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.smartmoneyengine_feature_cache_architecture_research import (
    COST_CATEGORIES,
    MODULE_INVOCATION_WEIGHTS,
    RESEARCH_MODULE_FAMILIES,
    FeatureCacheArchitectureError,
    SmartMoneyEngineFeatureCacheArchitectureResearch,
    generate_feature_cache_architecture_report,
)


def test_constants() -> None:
    assert len(COST_CATEGORIES) == 7
    assert len(RESEARCH_MODULE_FAMILIES) == 6
    assert set(MODULE_INVOCATION_WEIGHTS) == set(RESEARCH_MODULE_FAMILIES)


def test_category_shares_sum_to_100() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _StubBenchmark:
        category: str
        runtime_seconds: float = 1.0
        memory_bytes: int = 1000
        invocations_per_benchmark: int = 10
        notes: str = "stub"

    engine = SmartMoneyEngineFeatureCacheArchitectureResearch(symbols=("NIFTY50",))
    benchmarks = {category: _StubBenchmark(category=category) for category in COST_CATEGORIES}
    _, total_runtime, total_memory = engine._module_profiles(benchmarks)
    runtime_share, memory_share = engine._category_shares(benchmarks, total_runtime, total_memory)
    assert abs(sum(runtime_share.values()) - 100.0) < 0.1
    assert abs(sum(memory_share.values()) - 100.0) < 0.1


def test_repeated_inventory() -> None:
    engine = SmartMoneyEngineFeatureCacheArchitectureResearch(symbols=("NIFTY50",))
    inventory = engine._repeated_computation_inventory()
    assert len(inventory) >= 5
    assert any(item["computation"].startswith("_market_levels") for item in inventory)


def test_cache_design() -> None:
    engine = SmartMoneyEngineFeatureCacheArchitectureResearch(symbols=("NIFTY50",))
    frame = pd.DataFrame({"Close": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0]})
    runtime_share = {category: 100 / len(COST_CATEGORIES) for category in COST_CATEGORIES}
    design, tiers, reduction, overhead = engine._cache_design(runtime_share, frame)
    assert design["name"] == "SmartMoneyEngine Shared Feature Cache"
    assert len(tiers) == 4
    assert reduction > 0
    assert overhead >= 0


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-01-01", "end_date": "2026-01-01", "research_window_days": 365}',
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_feature_cache_architecture.json"

    class _FakeReport:
        benchmark_symbol = "NIFTY50"
        benchmark_timeframe = "5M"
        benchmark_bars = 1000
        expected_runtime_reduction_pct = 55.0
        expected_memory_overhead_mb = 12.5
        category_runtime_share_pct = {category: 14.29 for category in COST_CATEGORIES}
        bottlenecks_identified = ["market_level", "candle_pattern"]

        def as_dict(self) -> dict:
            return {
                "benchmark_symbol": self.benchmark_symbol,
                "benchmark_timeframe": self.benchmark_timeframe,
                "benchmark_bars": self.benchmark_bars,
                "expected_runtime_reduction_pct": self.expected_runtime_reduction_pct,
                "category_runtime_share_pct": self.category_runtime_share_pct,
                "cache_tiers": [],
            }

    def _fake_run(self: SmartMoneyEngineFeatureCacheArchitectureResearch, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineFeatureCacheArchitectureResearch, "run", _fake_run)

    report = generate_feature_cache_architecture_report(
        report_path=destination,
        filter_report_path=filter_report,
        symbols=("NIFTY50",),
    )
    assert report.expected_runtime_reduction_pct == 55.0
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["benchmark_symbol"] == "NIFTY50"
    assert "shared_feature_cache_design" not in payload or payload["benchmark_bars"] == 1000


def test_generate_report_missing_filter() -> None:
    with pytest.raises(FeatureCacheArchitectureError):
        generate_feature_cache_architecture_report(filter_report_path=Path("missing.json"))
