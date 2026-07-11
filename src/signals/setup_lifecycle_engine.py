"""
Setup Lifecycle Engine for SmartMoneyEngine.

Tracks institutional setup evolution across multiple candles instead of
requiring all structural events on a single bar.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.decision_engine import DecisionEngine, TradeDecision

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "setup_lifecycle_report.json"

MAX_SETUP_LIFETIME_BARS = 50


class SetupLifecycleError(Exception):
    """Raised when setup lifecycle tracking fails."""


class SetupDirection(str, Enum):
    """Directional bias for a lifecycle setup."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class LifecycleStage(str, Enum):
    """Ordered lifecycle stages for institutional setups."""

    LIQUIDITY_EVENT = "liquidity_event"
    BOS_EVENT = "bos_event"
    CHOCH_EVENT = "choch_event"
    FVG_CREATION = "fvg_creation"
    ORDER_BLOCK_CREATION = "order_block_creation"
    RETEST_EVENT = "retest_event"
    ENTRY_TRIGGER = "entry_trigger"
    SETUP_EXPIRATION = "setup_expiration"


STAGE_ORDER: tuple[LifecycleStage, ...] = (
    LifecycleStage.LIQUIDITY_EVENT,
    LifecycleStage.BOS_EVENT,
    LifecycleStage.CHOCH_EVENT,
    LifecycleStage.FVG_CREATION,
    LifecycleStage.ORDER_BLOCK_CREATION,
    LifecycleStage.RETEST_EVENT,
    LifecycleStage.ENTRY_TRIGGER,
    LifecycleStage.SETUP_EXPIRATION,
)

STAGE_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}


@dataclass
class StageEvent:
    """One lifecycle stage transition."""

    stage: str
    bar_index: int
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupLifecycleRecord:
    """Tracked setup evolving across multiple candles."""

    setup_id: str
    direction: str
    created_time: str
    created_bar: int
    last_event_time: str
    last_event_bar: int
    current_stage: str
    completed: bool
    expired: bool
    duration_bars: int
    stage_history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupLifecycleReport:
    """Aggregate setup lifecycle report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_candles: int
    total_setups: int
    completed_setups: int
    expired_setups: int
    active_at_end: int
    average_duration: float
    entry_triggers: int
    max_lifetime_bars: int
    execution_time_seconds: float
    setups: list[dict[str, Any]] = field(default_factory=list)
    entry_trigger_setups: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SetupLifecycleEngine:
    """
    Track setup evolution across candles with a 50-bar lifetime window.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Source timeframe label such as ``5``.
    max_lifetime_bars : int, optional
        Maximum candles a setup may remain active.
    """

    REQUIRED_COLUMNS = (
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Trend",
        "Bullish_BOS",
        "Bearish_BOS",
        "Bullish_CHOCH",
        "Bearish_CHOCH",
        "Bullish_FVG_Top",
        "Bullish_FVG_Bottom",
        "Bearish_FVG_Top",
        "Bearish_FVG_Bottom",
        "Bullish_OB_High",
        "Bullish_OB_Low",
        "Bearish_OB_High",
        "Bearish_OB_Low",
        "Bullish_OB_Mitigated",
        "Bearish_OB_Mitigated",
        "Buy_Liquidity_Sweep",
        "Sell_Liquidity_Sweep",
    )

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        max_lifetime_bars: int = MAX_SETUP_LIFETIME_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_lifetime_bars = max_lifetime_bars
        self._setup_counter = 0

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise SetupLifecycleError(f"Pipeline frame missing lifecycle columns: {missing}")

    def _next_setup_id(self, direction: SetupDirection, bar_index: int) -> str:
        self._setup_counter += 1
        suffix = uuid.uuid4().hex[:8]
        return f"{direction.value}_{bar_index}_{self._setup_counter}_{suffix}"

    @staticmethod
    def _stage_rank(stage: LifecycleStage) -> int:
        return STAGE_INDEX[stage]

    def _can_advance(self, current: LifecycleStage, target: LifecycleStage) -> bool:
        return self._stage_rank(target) > self._stage_rank(current)

    def _record_stage(
        self,
        setup: SetupLifecycleRecord,
        stage: LifecycleStage,
        bar_index: int,
        timestamp: str,
    ) -> None:
        if not self._can_advance(LifecycleStage(setup.current_stage), stage):
            if setup.current_stage != stage.value:
                return
        setup.current_stage = stage.value
        setup.last_event_bar = bar_index
        setup.last_event_time = timestamp
        setup.duration_bars = bar_index - setup.created_bar
        setup.stage_history.append(
            StageEvent(stage=stage.value, bar_index=bar_index, timestamp=timestamp).as_dict()
        )

    def _create_setup(
        self,
        direction: SetupDirection,
        bar_index: int,
        timestamp: str,
    ) -> SetupLifecycleRecord:
        setup = SetupLifecycleRecord(
            setup_id=self._next_setup_id(direction, bar_index),
            direction=direction.value,
            created_time=timestamp,
            created_bar=bar_index,
            last_event_time=timestamp,
            last_event_bar=bar_index,
            current_stage=LifecycleStage.LIQUIDITY_EVENT.value,
            completed=False,
            expired=False,
            duration_bars=0,
            stage_history=[
                StageEvent(
                    stage=LifecycleStage.LIQUIDITY_EVENT.value,
                    bar_index=bar_index,
                    timestamp=timestamp,
                ).as_dict()
            ],
        )
        return setup

    def _detect_liquidity_starts(
        self,
        row: pd.Series,
        bar_index: int,
        timestamp: str,
    ) -> list[SetupLifecycleRecord]:
        setups: list[SetupLifecycleRecord] = []
        if self._is_active(row.get("Sell_Liquidity_Sweep")):
            setups.append(self._create_setup(SetupDirection.BULLISH, bar_index, timestamp))
        if self._is_active(row.get("Buy_Liquidity_Sweep")):
            setups.append(self._create_setup(SetupDirection.BEARISH, bar_index, timestamp))
        return setups

    def _has_bos(self, row: pd.Series, direction: SetupDirection) -> bool:
        if direction == SetupDirection.BULLISH:
            return self._is_active(row.get("Bullish_BOS"))
        return self._is_active(row.get("Bearish_BOS"))

    def _has_choch(self, row: pd.Series, direction: SetupDirection) -> bool:
        if direction == SetupDirection.BULLISH:
            return self._is_active(row.get("Bullish_CHOCH"))
        return self._is_active(row.get("Bearish_CHOCH"))

    def _has_fvg(self, row: pd.Series, direction: SetupDirection) -> bool:
        if direction == SetupDirection.BULLISH:
            return self._is_active(row.get("Bullish_FVG_Top")) and self._is_active(
                row.get("Bullish_FVG_Bottom")
            )
        return self._is_active(row.get("Bearish_FVG_Top")) and self._is_active(
            row.get("Bearish_FVG_Bottom")
        )

    def _has_ob(self, row: pd.Series, direction: SetupDirection) -> bool:
        if direction == SetupDirection.BULLISH:
            return (
                self._is_active(row.get("Bullish_OB_High"))
                and self._is_active(row.get("Bullish_OB_Low"))
                and not self._is_active(row.get("Bullish_OB_Mitigated"))
            )
        return (
            self._is_active(row.get("Bearish_OB_High"))
            and self._is_active(row.get("Bearish_OB_Low"))
            and not self._is_active(row.get("Bearish_OB_Mitigated"))
        )

    def _is_retest(self, row: pd.Series, direction: SetupDirection) -> bool:
        low = self._to_float(row.get("Low")) or 0.0
        high = self._to_float(row.get("High")) or 0.0
        close = self._to_float(row.get("Close")) or 0.0

        if direction == SetupDirection.BULLISH:
            ob_high = self._to_float(row.get("Bullish_OB_High"))
            ob_low = self._to_float(row.get("Bullish_OB_Low"))
            if ob_high is not None and ob_low is not None and not self._is_active(
                row.get("Bullish_OB_Mitigated")
            ):
                if ob_low <= low <= ob_high and close >= ob_low:
                    return True

            fvg_top = self._to_float(row.get("Bullish_FVG_Top"))
            fvg_bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
            if fvg_top is not None and fvg_bottom is not None:
                if fvg_bottom <= close <= fvg_top:
                    return True
            return False

        ob_high = self._to_float(row.get("Bearish_OB_High"))
        ob_low = self._to_float(row.get("Bearish_OB_Low"))
        if ob_high is not None and ob_low is not None and not self._is_active(
            row.get("Bearish_OB_Mitigated")
        ):
            if ob_low <= high <= ob_high and close <= ob_high:
                return True

        fvg_top = self._to_float(row.get("Bearish_FVG_Top"))
        fvg_bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
        if fvg_top is not None and fvg_bottom is not None:
            if fvg_bottom <= close <= fvg_top:
                return True
        return False

    def _minimum_structure_met(self, setup: SetupLifecycleRecord) -> bool:
        reached = {event["stage"] for event in setup.stage_history}
        has_core = (
            LifecycleStage.LIQUIDITY_EVENT.value in reached
            and LifecycleStage.BOS_EVENT.value in reached
        )
        has_zone = (
            LifecycleStage.FVG_CREATION.value in reached
            or LifecycleStage.ORDER_BLOCK_CREATION.value in reached
        )
        return has_core and has_zone

    def _advance_setup(
        self,
        setup: SetupLifecycleRecord,
        row: pd.Series,
        bar_index: int,
        timestamp: str,
    ) -> None:
        if setup.completed or setup.expired:
            return

        direction = SetupDirection(setup.direction)

        if self._has_bos(row, direction):
            self._record_stage(setup, LifecycleStage.BOS_EVENT, bar_index, timestamp)
        if self._has_choch(row, direction):
            self._record_stage(setup, LifecycleStage.CHOCH_EVENT, bar_index, timestamp)
        if self._has_fvg(row, direction):
            self._record_stage(setup, LifecycleStage.FVG_CREATION, bar_index, timestamp)
        if self._has_ob(row, direction):
            self._record_stage(setup, LifecycleStage.ORDER_BLOCK_CREATION, bar_index, timestamp)
        if self._is_retest(row, direction):
            self._record_stage(setup, LifecycleStage.RETEST_EVENT, bar_index, timestamp)

        if (
            self._minimum_structure_met(setup)
            and self._is_retest(row, direction)
            and LifecycleStage(setup.current_stage)
            in {
                LifecycleStage.RETEST_EVENT,
                LifecycleStage.ORDER_BLOCK_CREATION,
                LifecycleStage.FVG_CREATION,
            }
        ):
            self._record_stage(setup, LifecycleStage.ENTRY_TRIGGER, bar_index, timestamp)
            setup.completed = True

    def _expire_setup(self, setup: SetupLifecycleRecord, bar_index: int, timestamp: str) -> None:
        if setup.completed or setup.expired:
            return
        setup.expired = True
        setup.current_stage = LifecycleStage.SETUP_EXPIRATION.value
        setup.last_event_bar = bar_index
        setup.last_event_time = timestamp
        setup.duration_bars = bar_index - setup.created_bar
        setup.stage_history.append(
            StageEvent(
                stage=LifecycleStage.SETUP_EXPIRATION.value,
                bar_index=bar_index,
                timestamp=timestamp,
            ).as_dict()
        )

    def track(self, frame: pd.DataFrame) -> list[SetupLifecycleRecord]:
        """Scan pipeline candles and track setup lifecycles."""
        self._validate_frame(frame)
        working = frame.reset_index(drop=True)
        active: list[SetupLifecycleRecord] = []
        finalized: list[SetupLifecycleRecord] = []

        for bar_index, row in working.iterrows():
            timestamp = str(row.get("Date"))

            still_active: list[SetupLifecycleRecord] = []
            for setup in active:
                age = bar_index - setup.created_bar
                if age >= self.max_lifetime_bars and not setup.completed:
                    self._expire_setup(setup, bar_index, timestamp)
                    finalized.append(setup)
                    continue
                self._advance_setup(setup, row, bar_index, timestamp)
                if setup.completed:
                    finalized.append(setup)
                else:
                    still_active.append(setup)
            active = still_active

            for setup in self._detect_liquidity_starts(row, bar_index, timestamp):
                self._advance_setup(setup, row, bar_index, timestamp)
                if setup.completed:
                    finalized.append(setup)
                else:
                    active.append(setup)

        end_index = len(working) - 1
        end_timestamp = str(working.iloc[-1]["Date"]) if len(working) else ""
        for setup in active:
            age = end_index - setup.created_bar
            if age >= self.max_lifetime_bars and not setup.completed:
                self._expire_setup(setup, end_index, end_timestamp)
            finalized.append(setup)

        return finalized

    def run(self, frame: pd.DataFrame, source_csv: str = "") -> SetupLifecycleReport:
        """Track lifecycles and build aggregate report."""
        started = time.perf_counter()
        if frame.empty:
            raise SetupLifecycleError("Pipeline frame is empty.")

        setups = self.track(frame)
        completed = [setup for setup in setups if setup.completed]
        expired = [setup for setup in setups if setup.expired and not setup.completed]
        active_at_end = [
            setup
            for setup in setups
            if not setup.completed and not setup.expired
        ]
        durations = [setup.duration_bars for setup in setups if setup.duration_bars > 0]
        average_duration = round(sum(durations) / len(durations), 2) if durations else 0.0

        entry_triggers = [
            setup.as_dict()
            for setup in completed
            if setup.current_stage == LifecycleStage.ENTRY_TRIGGER.value
        ]

        return SetupLifecycleReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=source_csv,
            total_candles=len(frame),
            total_setups=len(setups),
            completed_setups=len(completed),
            expired_setups=len(expired),
            active_at_end=len(active_at_end),
            average_duration=average_duration,
            entry_triggers=len(entry_triggers),
            max_lifetime_bars=self.max_lifetime_bars,
            execution_time_seconds=round(time.perf_counter() - started, 3),
            setups=[setup.as_dict() for setup in setups],
            entry_trigger_setups=entry_triggers,
        )

    def run_from_csv(self, pipeline_csv: Path | str) -> SetupLifecycleReport:
        """Load pipeline CSV and track setup lifecycles."""
        csv_path = Path(pipeline_csv)
        loader = DecisionEngine(symbol=self.symbol, timeframe=self.timeframe)
        frame = loader.load_pipeline_csv(csv_path)
        return self.run(frame, source_csv=str(csv_path))


def generate_setup_lifecycle_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
    max_lifetime_bars: int = MAX_SETUP_LIFETIME_BARS,
) -> SetupLifecycleReport:
    """Run setup lifecycle tracking and export JSON report."""
    engine = SetupLifecycleEngine(
        symbol=symbol,
        timeframe=timeframe,
        max_lifetime_bars=max_lifetime_bars,
    )
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    report = engine.run_from_csv(csv_path)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Setup lifecycle tracking completed: setups=%s completed=%s expired=%s entries=%s",
        report.total_setups,
        report.completed_setups,
        report.expired_setups,
        report.entry_triggers,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_setup_lifecycle_report()
        print("Setup Lifecycle Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Total Setups: {report.total_setups}")
        print(f"Completed: {report.completed_setups}")
        print(f"Expired: {report.expired_setups}")
        print(f"Entry Triggers: {report.entry_triggers}")
        print(f"Average Duration: {report.average_duration} bars")
        print(f"Max Lifetime: {report.max_lifetime_bars} bars")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except SetupLifecycleError as exc:
        logger.error("Setup lifecycle error: %s", exc)
        print(f"Setup lifecycle error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected setup lifecycle failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
