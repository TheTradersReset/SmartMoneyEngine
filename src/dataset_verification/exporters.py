"""Export dataset validation reports (JSON / CSV / HTML). Read-only outputs."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    def add(section: str, key: str, value: Any, extra: str = "") -> None:
        rows.append({"section": section, "key": str(key), "value": str(value), "extra": extra})

    meta = report.get("meta") or {}
    for key, value in meta.items():
        add("meta", key, value)

    health = report.get("health") or {}
    for key, value in health.items():
        add("health", key, value)

    coverage = report.get("coverage") or {}
    for key, value in coverage.items():
        add("coverage", key, value)

    integrity = report.get("integrity") or {}
    for key, value in integrity.items():
        add("integrity", key, value)

    for name, payload in (report.get("checks") or {}).items():
        status = payload.get("status") if isinstance(payload, dict) else payload
        add("checks", name, status, json.dumps(payload, default=str)[:500] if isinstance(payload, dict) else "")

    for issue in report.get("issues") or []:
        add(
            "issues",
            issue.get("check", ""),
            issue.get("severity", ""),
            issue.get("message", ""),
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "key", "value", "extra"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_html(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = report.get("meta") or {}
    health = report.get("health") or {}
    coverage = report.get("coverage") or {}
    integrity = report.get("integrity") or {}
    checks = report.get("checks") or {}
    issues = report.get("issues") or []

    check_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str((payload or {}).get('status')))}</td></tr>"
        for name, payload in checks.items()
        if isinstance(payload, dict)
    )
    issue_rows = "".join(
        f"<tr><td>{html.escape(str(i.get('severity')))}</td>"
        f"<td>{html.escape(str(i.get('check')))}</td>"
        f"<td>{html.escape(str(i.get('message')))}</td></tr>"
        for i in issues[:200]
    )
    score = health.get("health_score")
    verdict = health.get("verdict")
    band = health.get("band")

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Dataset Validation Report</title>
  <style>
    body {{ margin:0; font-family:"Segoe UI",Tahoma,sans-serif; background:#101826; color:#e8eef7; }}
    main {{ max-width:1000px; margin:0 auto; padding:28px 18px 60px; }}
    h1 {{ margin:0 0 8px; }}
    h2 {{ color:#6eb6ff; margin-top:28px; }}
    .meta {{ color:#9db0c7; margin-bottom:18px; }}
    .score {{
      display:inline-block; padding:14px 18px; border-radius:10px;
      background:#1b2738; border:1px solid #31465f; margin:8px 0 16px;
    }}
    .READY {{ border-color:#3ecf8e; }}
    .WARN {{ border-color:#c3a86b; }}
    .BLOCK {{ border-color:#f07178; }}
    table {{ width:100%; border-collapse:collapse; background:#1b2738; border:1px solid #31465f; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #31465f; text-align:left; font-size:0.92rem; }}
    th {{ background:#24344a; }}
  </style>
</head>
<body>
<main>
  <h1>Dataset Validation Report</h1>
  <div class="meta">
    Symbol: {html.escape(str(meta.get("symbol")))} |
    Resolution: {html.escape(str(meta.get("resolution")))} |
    Bars: {html.escape(str(meta.get("bar_count")))} |
    Source: {html.escape(str(meta.get("source") or ""))}<br/>
    Generated: {html.escape(str(meta.get("generated_at") or ""))}
  </div>

  <div class="score {html.escape(str(band or ''))}">
    <div>Health Score: <strong>{html.escape(str(score))}</strong></div>
    <div>Verdict: {html.escape(str(verdict))}</div>
  </div>

  <h2>Coverage</h2>
  <table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>
  {''.join(f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>' for k,v in coverage.items())}
  </tbody></table>

  <h2>Integrity</h2>
  <table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>
  {''.join(f'<tr><td>{html.escape(str(k))}</td><td><code>{html.escape(str(v))}</code></td></tr>' for k,v in integrity.items())}
  </tbody></table>

  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th></tr></thead><tbody>
  {check_rows}
  </tbody></table>

  <h2>Issues</h2>
  <table><thead><tr><th>Severity</th><th>Check</th><th>Message</th></tr></thead><tbody>
  {issue_rows or '<tr><td colspan="3">None</td></tr>'}
  </tbody></table>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path
