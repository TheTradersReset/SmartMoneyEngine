"""
Trade Validation Engine — post-signal objective evaluation.

Consumes persisted BUY/SELL signals and forward candles. Never influences
signal generation, scoring, or pipeline execution.
"""

from __future__ import annotations

from src.core.logger import logger
from src.trade_validation.config import TradeValidationConfig
from src.trade_validation.evaluator import evaluate_signal
from src.trade_validation.models import CandleBar, SignalRecord, TradeValidationResult
from src.trade_validation.signal_reader import SignalReader
from src.trade_validation.storage import TradeValidationDatabase


class TradeValidationEngine:
    """
    Poll the signal database, evaluate new signals, and persist validation rows.

    Designed as a standalone process or scheduled job — not wired into
    RealtimeSignalPipeline.
    """

    def __init__(self, config: TradeValidationConfig | None = None) -> None:
        self.config = config or TradeValidationConfig()
        self.reader = SignalReader(self.config.signal_db_path)
        self.store = TradeValidationDatabase(self.config.validation_db_path)

    def close(self) -> None:
        self.reader.close()
        self.store.close()

    def run_once(self) -> list[TradeValidationResult]:
        """Process new signals and refresh OPEN validations."""
        results: list[TradeValidationResult] = []
        last_id = self.store.get_last_processed_signal_id()
        signals = self.reader.fetch_signals_after_id(
            after_id=last_id,
            include_rejected=self.config.evaluate_rejected_signals,
        )
        for signal in signals:
            result = self._evaluate_and_persist(signal)
            results.append(result)
            self.store.set_last_processed_signal_id(signal.id)

        for row in self.store.open_validations():
            signal = self._signal_from_open_row(row)
            refreshed = self._evaluate_and_persist(signal)
            if refreshed.outcome != "OPEN":
                results.append(refreshed)
        return results

    def evaluate_signal_direct(
        self,
        signal: SignalRecord,
        forward_candles: list[CandleBar],
    ) -> TradeValidationResult:
        """Evaluate one signal in memory (tests / batch replay)."""
        return evaluate_signal(signal, forward_candles, config=self.config)

    def _evaluate_and_persist(self, signal: SignalRecord) -> TradeValidationResult:
        candles = self.reader.fetch_forward_candles(
            symbol=signal.symbol,
            after_timestamp=signal.timestamp,
            limit=self.config.evaluation_window_bars,
        )
        result = evaluate_signal(signal, candles, config=self.config)
        self.store.upsert_validation(result)
        logger.info(
            "Validated signal id=%s %s %s outcome=%s pnl=%s",
            signal.id,
            signal.direction,
            signal.timestamp,
            result.outcome,
            result.pnl,
        )
        return result

    @staticmethod
    def _signal_from_open_row(row: dict) -> SignalRecord:
        import json

        reason_codes = row.get("reason_codes")
        if isinstance(reason_codes, str):
            try:
                reason_codes = json.loads(reason_codes)
            except json.JSONDecodeError:
                reason_codes = []
        return SignalRecord(
            id=int(row["source_signal_id"]),
            timestamp=row["signal_timestamp"],
            direction=row["direction"],
            entry=float(row["entry_price"]),
            engine_version="UNKNOWN",
            accepted=True,
            symbol=row["symbol"],
            signal_score=float(row["signal_score"]) if row.get("signal_score") is not None else None,
            reason_codes=tuple(reason_codes or ()),
            raw_payload={},
        )


def main() -> None:
    """CLI entry: python -m src.trade_validation.engine"""
    engine = TradeValidationEngine()
    try:
        results = engine.run_once()
        logger.info("Trade validation cycle complete: %s records processed.", len(results))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
