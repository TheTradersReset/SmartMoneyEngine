"""Read-only access to the signal-engine SQLite database."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.trade_validation.config import TradeValidationConfig
from src.trade_validation.models import CandleBar, SignalRecord


def _normalize_ts(value: str) -> str:
    """Normalize timestamp strings for cross-table joins."""
    text = value.strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.isoformat()
    except ValueError:
        return text


class SignalReader:
    """Consumes persisted signals and candles without modifying the signal DB."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def fetch_signals_after_id(
        self,
        *,
        after_id: int = 0,
        include_rejected: bool = True,
        limit: int = 500,
    ) -> list[SignalRecord]:
        query = """
            SELECT * FROM signals
            WHERE id > ? AND direction IN ('BUY', 'SELL')
        """
        params: list[Any] = [after_id]
        if not include_rejected:
            query += " AND accepted = 1"
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        decisions = self._decision_index()
        default_symbol = TradeValidationConfig().default_symbol
        records: list[SignalRecord] = []
        for row in rows:
            payload = dict(row)
            raw = json.loads(payload.get("raw_json") or "{}")
            ts_key = _normalize_ts(payload["timestamp"])
            decision = decisions.get(ts_key, {})
            direction = payload["direction"]
            score = decision.get("buy_score") if direction == "BUY" else decision.get("sell_score")
            if score is None:
                score = _score_from_raw(raw, direction)
            reason_codes = decision.get("reason_codes")
            if not reason_codes:
                reason_codes = _reason_codes_from_raw(raw)
            records.append(
                SignalRecord(
                    id=int(payload["id"]),
                    timestamp=payload["timestamp"],
                    direction=direction,
                    entry=float(payload["entry"]),
                    engine_version=payload["engine_version"],
                    accepted=bool(payload["accepted"]),
                    symbol=decision.get("symbol") or default_symbol,
                    signal_score=float(score) if score is not None else None,
                    reason_codes=tuple(reason_codes or ()),
                    raw_payload=raw,
                ),
            )
        return records

    def fetch_forward_candles(
        self,
        *,
        symbol: str,
        after_timestamp: str,
        limit: int,
    ) -> list[CandleBar]:
        rows = self._conn.execute(
            """
            SELECT timestamp, open, high, low, close
            FROM candles
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (symbol, after_timestamp, limit),
        ).fetchall()
        return [
            CandleBar(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            for row in rows
        ]

    def _decision_index(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT timestamp, symbol, buy_score, sell_score, reason_codes FROM signal_decisions",
        ).fetchall()
        index: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = dict(row)
            try:
                payload["reason_codes"] = json.loads(payload.get("reason_codes") or "[]")
            except json.JSONDecodeError:
                payload["reason_codes"] = []
            index[_normalize_ts(payload["timestamp"])] = payload
        return index


def _score_from_raw(raw: dict[str, Any], direction: str) -> float | None:
    evaluation = raw.get("evaluation") or {}
    layer5 = evaluation.get("layer5") or {}
    for key in ("formula_completion_pct", "gate_pass_pct", "score"):
        value = layer5.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _reason_codes_from_raw(raw: dict[str, Any]) -> list[str]:
    evaluation = raw.get("evaluation") or {}
    layer5 = evaluation.get("layer5") or {}
    codes = layer5.get("reason_codes")
    return list(codes) if isinstance(codes, list) else []
