"""
Day-by-day SELL emission tracer matching campaign replay semantics.
Wrappers only — no strategy code changes.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.candle_diagnostics import _sell_gate_score
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.replay.data_feed import HistoricalDataFeed, window_for_day
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")
DB = ROOT / "data" / "paper" / "strategy_validation_campaign.db"
OUT = ROOT / "outputs" / "strategy_validation" / "sell_emission_trace_daybyday.json"


def _norm_keys(ts: Any) -> set[str]:
    dt = pd.Timestamp(ts)
    return {
        str(ts),
        dt.isoformat(),
        dt.strftime("%Y-%m-%d %H:%M:%S"),
        dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """
        SELECT timestamp, decision, final_signal, sell_score, reason_codes
        FROM signal_decisions
        WHERE sell_score >= 99.9
        ORDER BY timestamp
        """
    ).fetchall()
    conn.close()
    print(f"targets={len(rows)}", flush=True)

    target_meta: dict[str, Any] = {}
    target_keys: set[str] = set()
    days: set[date] = set()
    for ts, decision, final_signal, sell_score, reason_codes in rows:
        for k in _norm_keys(ts):
            target_meta[k] = {
                "timestamp": ts,
                "decision": decision,
                "final_signal": final_signal,
                "sell_score": sell_score,
                "reason_codes": reason_codes,
            }
            target_keys.add(k)
        days.add(pd.Timestamp(ts).date())

    feed = HistoricalDataFeed()
    feed.load()
    traces: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for day in sorted(days):
        print(f"DAY {day}", flush=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        pipe = RealtimeSignalPipeline(db=PaperSignalDatabase(tmp.name))
        window = window_for_day(day)
        warm = feed.warm_start_frame(window)
        if not warm.empty:
            pipe.warm_start_from_frame(warm)

        sell_eng = pipe.sell_engine._engine
        orig_eval = sell_eng.evaluate_bar
        orig_outcome = sell_eng._trade_outcome
        capture: dict[str, Any] = {}

        def wrapped_outcome(frame, bar, direction, _capture=capture, _eng=sell_eng, _orig=orig_outcome):
            end = min(len(frame) - 1, bar + FORWARD_BARS)
            forward = frame.iloc[bar + 1 : end + 1]
            entry = round(float(frame.iloc[bar]["Close"]), 2)
            stop, risk = _eng.trade_engine._structural_stop(frame, bar, entry, direction)
            result = _orig(frame, bar, direction)
            _capture["trade_outcome_called"] = True
            _capture["trade_outcome"] = {
                "bar": bar,
                "direction": direction,
                "frame_len": int(len(frame)),
                "forward_bars_available": int(len(forward)),
                "FORWARD_BARS_const": int(FORWARD_BARS),
                "entry": entry,
                "stop": float(stop),
                "risk": float(risk),
                "risk_le_0": bool(risk <= 0),
                "forward_empty": bool(forward.empty),
                "return_value": result if result else {},
                "return_empty": not bool(result),
            }
            return result

        def wrapped_eval(*, _capture=capture, _eng=sell_eng, _orig=orig_eval, **kwargs):
            _capture.clear()
            _capture["trade_outcome_called"] = False
            _capture["trade_outcome"] = None
            _eng._trade_outcome = wrapped_outcome  # type: ignore[method-assign]
            try:
                ev = _orig(**kwargs)
            finally:
                _eng._trade_outcome = orig_outcome  # type: ignore[method-assign]
            _capture["evaluation"] = ev
            return ev

        sell_eng.evaluate_bar = wrapped_eval  # type: ignore[method-assign]
        orig_handle = pipe._handle_closed_candle

        def traced_handle(candle, _pipe=pipe, _capture=capture, _orig_handle=orig_handle):
            _orig_handle(candle)
            keys = _norm_keys(candle.timestamp)
            if keys.isdisjoint(target_keys):
                return
            ev = _capture.get("evaluation")
            if ev is None:
                counters["no_evaluation"] += 1
                return
            layer1 = ev.get("layer1") or {}
            layer2 = ev.get("layer2") or {}
            layer3 = ev.get("layer3") or {}
            layer5 = ev.get("layer5") or {}
            layer5_pass = bool(layer5.get("pass"))
            outcome_called = bool(_capture.get("trade_outcome_called"))
            verdict = ev.get("verdict")
            sig = _pipe.sell_engine.to_signal(ev)
            db_row = None
            for k in keys:
                if k in target_meta:
                    db_row = target_meta[k]
                    break
            to_signal_return = None
            if sig is None:
                if verdict != "SELL":
                    to_signal_return = (
                        "sell_v6.py:87-88: if evaluation.get('verdict') != 'SELL': return None"
                    )
                else:
                    to_signal_return = "sell_v6.py:90-91: if entry <= 0: return None"

            trace = {
                "timestamp": str(ev.get("timestamp") or candle.timestamp.isoformat()),
                "bar_in_frame": ev.get("bar"),
                "runtime_sell_gate_score": _sell_gate_score(ev),
                "db_sell_score": db_row["sell_score"] if db_row else None,
                "db_decision": db_row["decision"] if db_row else None,
                "db_final_signal": db_row["final_signal"] if db_row else None,
                "layer1": {
                    "active": layer1.get("active"),
                    "failed_breakout_present": layer1.get("failed_breakout_present"),
                    "events_detected": layer1.get("events_detected"),
                },
                "layer2": {
                    "aligned": layer2.get("aligned"),
                    "htf_trend": layer2.get("htf_trend"),
                    "vwap_state": layer2.get("vwap_state"),
                    "vwap_gate_passes": layer2.get("vwap_gate_passes"),
                    "v4_ema_bearish": layer2.get("v4_ema_bearish"),
                    "ema_structure": layer2.get("ema_structure"),
                    "direction": layer2.get("direction"),
                },
                "layer3": {
                    "confirmed": layer3.get("confirmed"),
                    "confirmation_candle": layer3.get("confirmation_candle"),
                    "volume_bucket": layer3.get("volume_bucket"),
                },
                "layer5": {
                    "pass": layer5_pass,
                    "reason_codes": list(layer5.get("reason_codes") or []),
                },
                "layer5_actually_passed": layer5_pass,
                "trade_outcome_called": outcome_called,
                "trade_outcome": _capture.get("trade_outcome"),
                "layer4_present": ev.get("layer4") is not None,
                "evaluate_bar_verdict": verdict,
                "to_signal_executed": sig is not None,
                "to_signal_return_if_not": to_signal_return,
                "persistence_reached": False if sig is None else True,
                "persistence_reject_reason": (
                    "never_reached — to_signal() returned None"
                    if sig is None
                    else "would enter candidate path"
                ),
            }
            traces.append(trace)
            counters["traced"] += 1
            counters[f"layer5_pass={layer5_pass}"] += 1
            counters[f"trade_outcome_called={outcome_called}"] += 1
            counters[f"verdict={verdict}"] += 1
            counters[f"score_match={abs(float(trace['runtime_sell_gate_score'])-float(trace['db_sell_score'] or -1))<0.05}"] += 1
            if outcome_called and _capture.get("trade_outcome"):
                to = _capture["trade_outcome"]
                counters[f"forward_empty={to.get('forward_empty')}"] += 1
                counters[f"return_empty={to.get('return_empty')}"] += 1
            print(
                f"TRACE {trace['timestamp']} rt={trace['runtime_sell_gate_score']} "
                f"L5={layer5_pass} codes={layer5.get('reason_codes')} "
                f"out={outcome_called} fwd={(None if not _capture.get('trade_outcome') else _capture['trade_outcome'].get('forward_bars_available'))} "
                f"verdict={verdict}",
                flush=True,
            )

        pipe._handle_closed_candle = traced_handle  # type: ignore[method-assign]

        for candle in feed.iter_candles(window):
            pipe.ingest_closed_candle(candle)

    disappear: Counter[str] = Counter()
    for t in traces:
        if not t.get("layer5_actually_passed"):
            disappear["died_at_layer5"] += 1
        elif t.get("trade_outcome_called") and (t.get("trade_outcome") or {}).get("return_empty"):
            disappear["died_at_trade_outcome_empty"] += 1
        elif t.get("evaluate_bar_verdict") == "SELL" and not t.get("to_signal_executed"):
            disappear["died_at_to_signal"] += 1
        elif t.get("to_signal_executed"):
            disappear["reached_to_signal"] += 1
        else:
            disappear["other"] += 1

    # Focus subset where runtime score == 100 (true sell_score==100 under this path)
    r100 = [t for t in traces if float(t.get("runtime_sell_gate_score") or 0) >= 99.9]
    d100 = [t for t in traces if float(t.get("db_sell_score") or 0) >= 99.9]

    payload = {
        "generated_at": datetime.now(tz=IST).isoformat(),
        "db_targets": len(rows),
        "traced_count": len(traces),
        "runtime_score_100_count": len(r100),
        "counters": dict(counters),
        "disappearance_histogram_all_traced": dict(disappear),
        "disappearance_runtime_score_100": dict(
            Counter(
                (
                    "died_at_layer5"
                    if not t["layer5_actually_passed"]
                    else "died_at_trade_outcome_empty"
                    if (t.get("trade_outcome") or {}).get("return_empty")
                    else "other"
                )
                for t in r100
            )
        ),
        "layer5_fail_reasons_runtime_100": dict(
            Counter(
                code
                for t in r100
                if not t["layer5_actually_passed"]
                for code in t["layer5"]["reason_codes"]
            )
        ),
        "traces": traces,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("WROTE", OUT, flush=True)
    print("DISAPPEAR_ALL", dict(disappear), flush=True)
    print("DISAPPEAR_R100", payload["disappearance_runtime_score_100"], flush=True)
    print("L5FAIL_R100", payload["layer5_fail_reasons_runtime_100"], flush=True)
    print("COUNTERS", dict(counters), flush=True)
    print("traced", len(traces), "r100", len(r100), "d100_rows_seen", len(d100), flush=True)


if __name__ == "__main__":
    main()
