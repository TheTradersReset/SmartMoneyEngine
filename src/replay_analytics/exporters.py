"""Export analytics payloads to JSON, CSV, and HTML."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def export_json(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def export_csv(report: dict[str, Any], path: Path) -> Path:
    """
    Flat CSV with sectioned rows for spreadsheet consumption.

    Columns: section, key, value, extra
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def add(section: str, key: str, value: Any, extra: str = "") -> None:
        rows.append({"section": section, "key": key, "value": value, "extra": extra})

    summary = report.get("replay_summary") or {}
    for key, value in summary.items():
        add("replay_summary", key, value)

    stats = report.get("decision_statistics") or {}
    for key in ("BUY", "SELL", "NO_TRADE", "BUY_pct", "SELL_pct", "NO_TRADE_pct"):
        add("decision_statistics", key, stats.get(key))

    scores = report.get("score_statistics") or {}
    for key, value in scores.items():
        add("score_statistics", key, value)

    for item in (report.get("rule_statistics") or {}).get("top_rejection_reasons") or []:
        add("top_rejection_reasons", item["code"], item["count"], f"pct={item['pct']}")

    for item in (report.get("rule_statistics") or {}).get("top_passed_rules") or []:
        add("top_passed_rules", item["code"], item["count"], f"pct={item['pct']}")

    for item in (report.get("rule_statistics") or {}).get("rule_frequency") or []:
        add("rule_frequency", item["code"], item["rejection_count"], f"pct={item['pct']}")

    market = report.get("market_statistics") or {}
    for dist_name in ("trend_distribution", "regime_distribution", "vwap_distribution", "htf_distribution"):
        for item in market.get(dist_name) or []:
            add(dist_name, item["label"], item["count"], f"pct={item['pct']}")

    for item in report.get("daily_summary") or []:
        add(
            "daily_summary",
            item["period"],
            item["decisions"],
            (
                f"BUY={item['BUY']} SELL={item['SELL']} NO_TRADE={item['NO_TRADE']} "
                f"avg_buy={item['avg_buy_score']} avg_sell={item['avg_sell_score']}"
            ),
        )

    for item in report.get("monthly_summary") or []:
        add(
            "monthly_summary",
            item["period"],
            item["decisions"],
            (
                f"BUY={item['BUY']} SELL={item['SELL']} NO_TRADE={item['NO_TRADE']} "
                f"avg_buy={item['avg_buy_score']} avg_sell={item['avg_sell_score']}"
            ),
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "key", "value", "extra"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    thead = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def export_html(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("replay_summary") or {}
    decisions = report.get("decision_statistics") or {}
    scores = report.get("score_statistics") or {}
    rules = report.get("rule_statistics") or {}
    market = report.get("market_statistics") or {}
    meta = report.get("meta") or {}

    rejection_rows = [
        [item["code"], item["count"], f"{item['pct']}%"]
        for item in rules.get("top_rejection_reasons") or []
    ]
    passed_rows = [
        [item["code"], item["count"], f"{item['pct']}%"]
        for item in rules.get("top_passed_rules") or []
    ]
    daily_rows = [
        [
            item["period"],
            item["decisions"],
            item["BUY"],
            item["SELL"],
            item["NO_TRADE"],
            item["avg_buy_score"],
            item["avg_sell_score"],
        ]
        for item in report.get("daily_summary") or []
    ]
    monthly_rows = [
        [
            item["period"],
            item["decisions"],
            item["BUY"],
            item["SELL"],
            item["NO_TRADE"],
            item["avg_buy_score"],
            item["avg_sell_score"],
        ]
        for item in report.get("monthly_summary") or []
    ]

    def dist_table(name: str) -> str:
        rows = [[item["label"], item["count"], f"{item['pct']}%"] for item in market.get(name) or []]
        return _table(["Label", "Count", "Pct"], rows)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Replay Analytics Report</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa7b8;
      --accent: #3d9cf0;
      --line: #2a3648;
      --buy: #3ecf8e;
      --sell: #f07178;
      --neutral: #c3a86b;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(160deg, #0f1419 0%, #162033 100%);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
    h1 {{ font-size: 1.8rem; margin: 0 0 8px; }}
    h2 {{ font-size: 1.2rem; margin: 28px 0 12px; color: var(--accent); }}
    .meta {{ color: var(--muted); margin-bottom: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .card .label {{ color: var(--muted); font-size: 0.85rem; }}
    .card .value {{ font-size: 1.4rem; font-weight: 650; margin-top: 4px; }}
    .buy {{ color: var(--buy); }}
    .sell {{ color: var(--sell); }}
    .notrade {{ color: var(--neutral); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 8px;
    }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; font-size: 0.92rem; }}
    th {{ background: #223049; color: #c9d6e8; }}
    tr:last-child td {{ border-bottom: none; }}
    .note {{ color: var(--muted); font-size: 0.85rem; margin-top: 6px; }}
  </style>
</head>
<body>
<main>
  <h1>Replay Analytics Report</h1>
  <div class="meta">
    Source DB: {html.escape(str(meta.get("db_path") or ""))}<br/>
    Generated: {html.escape(str(meta.get("generated_at") or ""))}<br/>
    Window: {html.escape(str(summary.get("replay_window_start") or "N/A"))}
    → {html.escape(str(summary.get("replay_window_end") or "N/A"))}
  </div>

  <h2>1. Replay Summary</h2>
  <div class="cards">
    <div class="card"><div class="label">Total Candles</div><div class="value">{summary.get("total_candles")}</div></div>
    <div class="card"><div class="label">Total Decisions</div><div class="value">{summary.get("total_decisions")}</div></div>
    <div class="card"><div class="label">BUY Signals</div><div class="value buy">{summary.get("buy_signals")}</div></div>
    <div class="card"><div class="label">SELL Signals</div><div class="value sell">{summary.get("sell_signals")}</div></div>
    <div class="card"><div class="label">NO_TRADE</div><div class="value notrade">{summary.get("no_trade_count")}</div></div>
  </div>

  <h2>2. Decision Statistics</h2>
  {_table(["Metric", "Value"], [
      ["BUY %", f"{decisions.get('BUY_pct')}%"],
      ["SELL %", f"{decisions.get('SELL_pct')}%"],
      ["NO_TRADE %", f"{decisions.get('NO_TRADE_pct')}%"],
  ])}

  <h2>3. Score Statistics</h2>
  {_table(["Metric", "Value"], [
      ["Average Buy Score", scores.get("average_buy_score")],
      ["Average Sell Score", scores.get("average_sell_score")],
      ["Maximum Buy Score", scores.get("maximum_buy_score")],
      ["Maximum Sell Score", scores.get("maximum_sell_score")],
  ])}

  <h2>4. Rule Statistics</h2>
  <h3 style="color:var(--muted);font-size:1rem;">Top Rejection Reasons</h3>
  {_table(["Code", "Count", "Pct"], rejection_rows)}
  <h3 style="color:var(--muted);font-size:1rem;">Top Passed Rules</h3>
  {_table(["Code", "Count", "Pct"], passed_rows)}

  <h2>5. Market Statistics</h2>
  <h3 style="color:var(--muted);font-size:1rem;">Trend</h3>
  {dist_table("trend_distribution")}
  <h3 style="color:var(--muted);font-size:1rem;">Regime</h3>
  {dist_table("regime_distribution")}
  <h3 style="color:var(--muted);font-size:1rem;">VWAP</h3>
  {dist_table("vwap_distribution")}
  <p class="note">{html.escape(str(market.get("vwap_note") or ""))}</p>
  <h3 style="color:var(--muted);font-size:1rem;">HTF</h3>
  {dist_table("htf_distribution")}

  <h2>6. Daily Summary</h2>
  {_table(["Day", "Decisions", "BUY", "SELL", "NO_TRADE", "Avg Buy", "Avg Sell"], daily_rows)}

  <h2>7. Monthly Summary</h2>
  {_table(["Month", "Decisions", "BUY", "SELL", "NO_TRADE", "Avg Buy", "Avg Sell"], monthly_rows)}
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path
