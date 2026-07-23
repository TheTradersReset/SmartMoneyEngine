"""
Strategy Validation Campaign Runner.

Replays the EXISTING ReplayEngine day-by-day over a date range and writes
one evidence report. Does not call BUY_V3 / SELL_V6 directly.
Does not modify strategy logic.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.core.logger import logger
from src.replay.data_feed import DEFAULT_HISTORY_CSV
from src.replay.repeated_replay import run_repeated_replay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN_DB = PROJECT_ROOT / "data" / "paper" / "strategy_validation_campaign.db"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "strategy_validation"


def _parse_ts(value: str) -> datetime | None:
    text = (value or "").strip().replace("T", " ")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_decisions(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT timestamp, decision, final_signal, buy_score, sell_score
        FROM signal_decisions
        ORDER BY timestamp ASC
        """,
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _load_signals(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT timestamp, direction, accepted, engine_version
        FROM signals
        ORDER BY timestamp ASC
        """,
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def build_campaign_report(
    *,
    decisions: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    date_from: date,
    date_to: date,
    db_path: Path,
) -> dict[str, Any]:
    total_candles = len(decisions)
    decision_counter = Counter(str(r.get("decision") or r.get("final_signal") or "UNKNOWN") for r in decisions)
    buy_n = decision_counter.get("BUY", 0)
    sell_n = decision_counter.get("SELL", 0)
    no_trade_n = decision_counter.get("NO_TRADE", 0)

    # Prefer persisted signal rows for BUY/SELL timestamps when present;
    # fall back to decision==BUY/SELL.
    buy_ts = [str(s["timestamp"]) for s in signals if str(s.get("direction")) == "BUY"]
    sell_ts = [str(s["timestamp"]) for s in signals if str(s.get("direction")) == "SELL"]
    if not buy_ts:
        buy_ts = [str(d["timestamp"]) for d in decisions if str(d.get("decision")) == "BUY"]
    if not sell_ts:
        sell_ts = [str(d["timestamp"]) for d in decisions if str(d.get("decision")) == "SELL"]

    days_with_buy: set[str] = set()
    days_with_sell: set[str] = set()
    days_all: set[str] = set()
    daily: dict[str, dict[str, int]] = defaultdict(lambda: {"BUY": 0, "SELL": 0, "NO_TRADE": 0, "candles": 0})
    monthly: dict[str, dict[str, int]] = defaultdict(lambda: {"BUY": 0, "SELL": 0, "NO_TRADE": 0, "candles": 0})

    for row in decisions:
        ts = _parse_ts(str(row.get("timestamp") or ""))
        if ts is None:
            continue
        day = ts.date().isoformat()
        month = f"{ts.year:04d}-{ts.month:02d}"
        days_all.add(day)
        decision = str(row.get("decision") or "UNKNOWN")
        daily[day]["candles"] += 1
        monthly[month]["candles"] += 1
        if decision in ("BUY", "SELL", "NO_TRADE"):
            daily[day][decision] += 1
            monthly[month][decision] += 1

    for ts_text in buy_ts:
        ts = _parse_ts(ts_text)
        if ts is not None:
            days_with_buy.add(ts.date().isoformat())
    for ts_text in sell_ts:
        ts = _parse_ts(ts_text)
        if ts is not None:
            days_with_sell.add(ts.date().isoformat())

    days_no_signals = sorted(days_all - days_with_buy - days_with_sell)
    total = max(total_candles, 1)

    if buy_n == 0 and sell_n == 0 and len(buy_ts) == 0 and len(sell_ts) == 0:
        verdict = "C. Strategy generated zero signals"
        verdict_code = "C"
    elif (buy_n + sell_n + len(buy_ts) + len(sell_ts)) <= max(1, total_candles // 500):
        # Extremely rare relative to candle volume
        verdict = "B. Strategy almost never generates signals"
        verdict_code = "B"
    else:
        verdict = "A. Strategy generates signals normally"
        verdict_code = "A"

    # If we only have signal-table counts, align summary BUY/SELL to max of decision vs signal rows.
    buy_signals = max(buy_n, len(buy_ts))
    sell_signals = max(sell_n, len(sell_ts))

    return {
        "meta": {
            "campaign_from": date_from.isoformat(),
            "campaign_to": date_to.isoformat(),
            "db_path": str(db_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "question": "Does BUY_V3 / SELL_V6 actually generate signals over historical data?",
        },
        "summary": {
            "total_trading_days": len(days_all),
            "total_candles": total_candles,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "no_trade_decisions": no_trade_n,
            "buy_pct": round(100.0 * buy_signals / total, 4),
            "sell_pct": round(100.0 * sell_signals / total, 4),
            "no_trade_pct": round(100.0 * no_trade_n / total, 4),
            "first_buy_timestamp": buy_ts[0] if buy_ts else None,
            "last_buy_timestamp": buy_ts[-1] if buy_ts else None,
            "first_sell_timestamp": sell_ts[0] if sell_ts else None,
            "last_sell_timestamp": sell_ts[-1] if sell_ts else None,
            "days_containing_buy": len(days_with_buy),
            "days_containing_sell": len(days_with_sell),
            "days_containing_no_signals": len(days_no_signals),
        },
        "top_20_buy_timestamps": buy_ts[:20],
        "top_20_sell_timestamps": sell_ts[:20],
        "daily_signal_counts": [
            {"date": day, **daily[day]} for day in sorted(daily)
        ],
        "monthly_signal_counts": [
            {"month": month, **monthly[month]} for month in sorted(monthly)
        ],
        "days_containing_buy_list": sorted(days_with_buy),
        "days_containing_sell_list": sorted(days_with_sell),
        "days_containing_no_signals_list": days_no_signals,
        "verdict": {
            "code": verdict_code,
            "text": verdict,
            "answer": (
                "YES" if verdict_code == "A" else
                "RARELY" if verdict_code == "B" else
                "NO"
            ),
        },
    }


def export_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "strategy_validation_report.json"
    csv_path = out_dir / "strategy_validation_report.csv"
    html_path = out_dir / "strategy_validation_report.html"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows: list[dict[str, str]] = []

    def add(section: str, key: str, value: Any) -> None:
        rows.append({"section": section, "key": str(key), "value": str(value)})

    for key, value in (report.get("summary") or {}).items():
        add("summary", key, value)
    add("verdict", "code", report["verdict"]["code"])
    add("verdict", "text", report["verdict"]["text"])
    add("verdict", "answer", report["verdict"]["answer"])
    for ts in report.get("top_20_buy_timestamps") or []:
        add("top_20_buy_timestamps", ts, "BUY")
    for ts in report.get("top_20_sell_timestamps") or []:
        add("top_20_sell_timestamps", ts, "SELL")
    for item in report.get("daily_signal_counts") or []:
        add("daily_signal_counts", item["date"], json.dumps(item))
    for item in report.get("monthly_signal_counts") or []:
        add("monthly_signal_counts", item["month"], json.dumps(item))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)

    s = report["summary"]
    v = report["verdict"]
    daily_rows = "".join(
        f"<tr><td>{html.escape(i['date'])}</td><td>{i['BUY']}</td><td>{i['SELL']}</td>"
        f"<td>{i['NO_TRADE']}</td><td>{i['candles']}</td></tr>"
        for i in report.get("daily_signal_counts") or []
    )
    monthly_rows = "".join(
        f"<tr><td>{html.escape(i['month'])}</td><td>{i['BUY']}</td><td>{i['SELL']}</td>"
        f"<td>{i['NO_TRADE']}</td><td>{i['candles']}</td></tr>"
        for i in report.get("monthly_signal_counts") or []
    )
    buy_list = "".join(f"<li>{html.escape(ts)}</li>" for ts in (report.get("top_20_buy_timestamps") or [])[:20])
    sell_list = "".join(f"<li>{html.escape(ts)}</li>" for ts in (report.get("top_20_sell_timestamps") or [])[:20])

    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Strategy Validation Report</title>
<style>
body{{margin:0;font-family:Segoe UI,Tahoma,sans-serif;background:#0f1520;color:#e8eef7}}
main{{max-width:980px;margin:0 auto;padding:28px 16px 64px}}
h1,h2{{color:#7eb6ff}} table{{width:100%;border-collapse:collapse;background:#1a2433;margin:12px 0}}
th,td{{border-bottom:1px solid #2d3d55;padding:8px;text-align:left}} .verdict{{padding:14px;border:1px solid #3d9cf0;border-radius:8px}}
</style></head><body><main>
<h1>Strategy Validation Campaign Report</h1>
<p>Question: Does BUY_V3 / SELL_V6 actually generate signals over historical data?</p>
<div class="verdict"><strong>{html.escape(v['text'])}</strong><br/>Answer: {html.escape(v['answer'])}</div>
<h2>Summary</h2>
<table><tbody>
<tr><td>Total Trading Days</td><td>{s['total_trading_days']}</td></tr>
<tr><td>Total Candles</td><td>{s['total_candles']}</td></tr>
<tr><td>BUY Signals</td><td>{s['buy_signals']}</td></tr>
<tr><td>SELL Signals</td><td>{s['sell_signals']}</td></tr>
<tr><td>NO_TRADE Decisions</td><td>{s['no_trade_decisions']}</td></tr>
<tr><td>BUY %</td><td>{s['buy_pct']}</td></tr>
<tr><td>SELL %</td><td>{s['sell_pct']}</td></tr>
<tr><td>NO_TRADE %</td><td>{s['no_trade_pct']}</td></tr>
<tr><td>First BUY</td><td>{html.escape(str(s['first_buy_timestamp']))}</td></tr>
<tr><td>Last BUY</td><td>{html.escape(str(s['last_buy_timestamp']))}</td></tr>
<tr><td>First SELL</td><td>{html.escape(str(s['first_sell_timestamp']))}</td></tr>
<tr><td>Last SELL</td><td>{html.escape(str(s['last_sell_timestamp']))}</td></tr>
<tr><td>Days containing BUY</td><td>{s['days_containing_buy']}</td></tr>
<tr><td>Days containing SELL</td><td>{s['days_containing_sell']}</td></tr>
<tr><td>Days containing no signals</td><td>{s['days_containing_no_signals']}</td></tr>
</tbody></table>
<h2>Top 20 BUY timestamps</h2><ol>{buy_list or '<li>None</li>'}</ol>
<h2>Top 20 SELL timestamps</h2><ol>{sell_list or '<li>None</li>'}</ol>
<h2>Daily signal counts</h2>
<table><thead><tr><th>Date</th><th>BUY</th><th>SELL</th><th>NO_TRADE</th><th>Candles</th></tr></thead>
<tbody>{daily_rows}</tbody></table>
<h2>Monthly signal counts</h2>
<table><thead><tr><th>Month</th><th>BUY</th><th>SELL</th><th>NO_TRADE</th><th>Candles</th></tr></thead>
<tbody>{monthly_rows}</tbody></table>
</main></body></html>
""",
        encoding="utf-8",
    )
    return json_path, csv_path, html_path


def run_campaign(
    *,
    date_from: date,
    date_to: date,
    csv_path: Path | str = DEFAULT_HISTORY_CSV,
    signal_db_path: Path | str = DEFAULT_CAMPAIGN_DB,
    out_dir: Path | str = DEFAULT_OUT,
    replay: bool = True,
) -> dict[str, Any]:
    signal_db_path = Path(signal_db_path)
    out_dir = Path(out_dir)

    if replay:
        # Reuse existing repeated replay → existing ReplayEngine only.
        run_repeated_replay(
            start=date_from,
            end=date_to,
            csv_path=csv_path,
            signal_db_path=signal_db_path,
        )

    decisions = _load_decisions(signal_db_path)
    signals = _load_signals(signal_db_path)
    report = build_campaign_report(
        decisions=decisions,
        signals=signals,
        date_from=date_from,
        date_to=date_to,
        db_path=signal_db_path,
    )
    json_path, csv_path_out, html_path = export_reports(report, out_dir)
    report["meta"]["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path_out),
        "html": str(html_path),
    }
    print(f"[CAMPAIGN] verdict={report['verdict']['text']}", flush=True)
    print(f"[CAMPAIGN] BUY={report['summary']['buy_signals']} SELL={report['summary']['sell_signals']} "
          f"NO_TRADE={report['summary']['no_trade_decisions']}", flush=True)
    print(f"[CAMPAIGN] wrote {json_path}", flush=True)
    print(f"[CAMPAIGN] wrote {csv_path_out}", flush=True)
    print(f"[CAMPAIGN] wrote {html_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy Validation Campaign Runner (Replay-only evidence).")
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--signal-db", type=Path, default=DEFAULT_CAMPAIGN_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip replay; aggregate an existing campaign DB only",
    )
    args = parser.parse_args(argv)

    try:
        report = run_campaign(
            date_from=date.fromisoformat(args.date_from),
            date_to=date.fromisoformat(args.date_to),
            csv_path=args.csv,
            signal_db_path=args.signal_db,
            out_dir=args.out,
            replay=not args.report_only,
        )
        return 0 if report["verdict"]["code"] in {"A", "B", "C"} else 1
    except Exception as exc:
        logger.exception("Strategy validation campaign failed.")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
