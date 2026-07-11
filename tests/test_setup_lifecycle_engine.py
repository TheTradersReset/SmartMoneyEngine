"""Tests for the Setup Lifecycle Engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.signals.setup_lifecycle_engine import (
    MAX_SETUP_LIFETIME_BARS,
    LifecycleStage,
    SetupDirection,
    SetupLifecycleEngine,
    SetupLifecycleError,
    generate_setup_lifecycle_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _frame(length: int = 60) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.3
        rows.append(
            {
                "Date": (base + pd.Timedelta(minutes=5 * index)).isoformat(),
                "Open": price,
                "High": price + 0.8,
                "Low": price - 0.5,
                "Close": price + 0.2,
                "Trend": "BULLISH",
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": pd.NA,
                "Bearish_OB_Low": pd.NA,
                "Bullish_OB_Mitigated": pd.NA,
                "Bearish_OB_Mitigated": pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def test_lifecycle_stage_order_has_eight_stages() -> None:
    assert len(LifecycleStage) == 8


def test_max_lifetime_default_is_50() -> None:
    assert MAX_SETUP_LIFETIME_BARS == 50


def test_liquidity_event_creates_bullish_setup() -> None:
    frame = _frame(20)
    frame.loc[5, "Sell_Liquidity_Sweep"] = 99.0
    engine = SetupLifecycleEngine()
    setups = engine.track(frame)
    bullish = [setup for setup in setups if setup.direction == SetupDirection.BULLISH.value]
    assert bullish
    assert bullish[0].current_stage in {
        LifecycleStage.LIQUIDITY_EVENT.value,
        LifecycleStage.BOS_EVENT.value,
    }


def test_events_across_multiple_candles() -> None:
    frame = _frame(30)
    frame.loc[5, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[10, "Bullish_BOS"] = 103.0
    frame.loc[15, "Bullish_CHOCH"] = 104.0
    engine = SetupLifecycleEngine()
    setups = engine.track(frame)
    setup = next(setup for setup in setups if setup.direction == SetupDirection.BULLISH.value)
    stages = {event["stage"] for event in setup.stage_history}
    assert LifecycleStage.LIQUIDITY_EVENT.value in stages
    assert LifecycleStage.BOS_EVENT.value in stages
    assert LifecycleStage.CHOCH_EVENT.value in stages
    assert setup.stage_history[0]["bar_index"] != setup.stage_history[-1]["bar_index"]


def test_setup_completes_on_entry_trigger_path() -> None:
    frame = _frame(25)
    frame.loc[5, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[8, "Bullish_BOS"] = 102.0
    frame.loc[12, "Bullish_FVG_Top"] = 104.0
    frame.loc[12, "Bullish_FVG_Bottom"] = 103.0
    frame.loc[14, "Close"] = 103.5
    engine = SetupLifecycleEngine()
    setups = engine.track(frame)
    completed = [setup for setup in setups if setup.completed]
    assert completed
    assert completed[0].current_stage == LifecycleStage.ENTRY_TRIGGER.value


def test_setup_expires_after_max_lifetime() -> None:
    frame = _frame(70)
    frame.loc[5, "Sell_Liquidity_Sweep"] = 99.0
    engine = SetupLifecycleEngine(max_lifetime_bars=10)
    setups = engine.track(frame)
    expired = [setup for setup in setups if setup.expired]
    assert expired
    assert expired[0].current_stage == LifecycleStage.SETUP_EXPIRATION.value


def test_bearish_lifecycle_from_buy_side_sweep() -> None:
    frame = _frame(20)
    frame.loc[7, "Buy_Liquidity_Sweep"] = 110.0
    frame.loc[9, "Bearish_BOS"] = 108.0
    engine = SetupLifecycleEngine()
    setups = engine.track(frame)
    bearish = [setup for setup in setups if setup.direction == SetupDirection.BEARISH.value]
    assert bearish
    stages = {event["stage"] for event in bearish[0].stage_history}
    assert LifecycleStage.BOS_EVENT.value in stages


def test_run_builds_aggregate_report() -> None:
    frame = _frame(40)
    frame.loc[5, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[10, "Bullish_BOS"] = 103.0
    engine = SetupLifecycleEngine()
    report = engine.run(frame)
    assert report.total_setups >= 1
    assert report.total_candles == 40
    assert "average_duration" in report.as_dict()


def test_run_rejects_empty_frame() -> None:
    engine = SetupLifecycleEngine()
    with pytest.raises(SetupLifecycleError):
        engine.run(pd.DataFrame())


def test_generate_report_writes_json(tmp_path: Path) -> None:
    frame = _frame(30)
    frame.loc[5, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[10, "Bullish_BOS"] = 103.0
    csv_path = tmp_path / "pipeline.csv"
    report_path = tmp_path / "setup_lifecycle_report.json"
    frame.to_csv(csv_path, index=False)

    report = generate_setup_lifecycle_report(
        pipeline_csv=csv_path,
        report_path=report_path,
        symbol="NIFTY50",
        timeframe="5",
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["total_setups"] >= 1
    assert "entry_triggers" in payload
    assert report.total_candles == 30


@pytest.mark.integration
def test_real_pipeline_lifecycle_if_present() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real pipeline CSV not available.")

    engine = SetupLifecycleEngine(symbol="NIFTY50", timeframe="5")
    report = engine.run_from_csv(pipeline_csv)

    assert report.total_candles > 1000
    assert report.total_setups > 0
    assert report.average_duration >= 0
