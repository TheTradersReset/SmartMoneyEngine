"""
FastAPI status dashboard for live paper trading.

Lightweight HTML + JSON API. Auto-refreshes every 2 seconds.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from src.live_paper.metrics import LiveMetrics
from src.paper_trading.trade_manager import PaperTradeManager
from src.storage.sqlite import PaperSignalDatabase

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SmartMoneyEngine Live Paper</title>
  <style>
    :root { --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#9aa7b8; --ok:#3dd68c; --bad:#ff6b6b; --accent:#5b9fd4; }
    body { margin:0; font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
    header { padding:1rem 1.25rem; border-bottom:1px solid #243044; }
    h1 { margin:0; font-size:1.15rem; letter-spacing:0.02em; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:0.75rem; padding:1rem; }
    .card { background:var(--card); border-radius:10px; padding:0.9rem 1rem; }
    .label { color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; }
    .value { font-size:1.25rem; margin-top:0.35rem; font-variant-numeric:tabular-nums; }
    .ok { color:var(--ok); } .bad { color:var(--bad); }
    #curve { width:100%; height:80px; background:#101821; border-radius:8px; }
    #errors { font-size:0.85rem; color:var(--muted); white-space:pre-wrap; max-height:160px; overflow:auto; }
    .wide { grid-column: 1 / -1; }
  </style>
</head>
<body>
  <header><h1>SmartMoneyEngine — Live Paper Trading</h1></header>
  <div class="grid" id="grid">
    <div class="card"><div class="label">Market Status</div><div class="value" id="market_status">-</div></div>
    <div class="card"><div class="label">WS Status</div><div class="value" id="ws_status">-</div></div>
    <div class="card"><div class="label">Heartbeat</div><div class="value" id="heartbeat">-</div></div>
    <div class="card"><div class="label">Current Candle</div><div class="value" id="candle" style="font-size:0.95rem">-</div></div>
    <div class="card"><div class="label">Today Signals</div><div class="value" id="today_signals">-</div></div>
    <div class="card"><div class="label">Open Trades</div><div class="value" id="open_trades">-</div></div>
    <div class="card"><div class="label">Closed Trades</div><div class="value" id="closed_trades">-</div></div>
    <div class="card"><div class="label">Win Rate</div><div class="value" id="win_rate">-</div></div>
    <div class="card"><div class="label">Running PnL</div><div class="value" id="running_pnl">-</div></div>
    <div class="card"><div class="label">Avg Latency</div><div class="value" id="avg_latency">-</div></div>
    <div class="card"><div class="label">DB Status</div><div class="value" id="db_ok">-</div></div>
    <div class="card"><div class="label">CPU</div><div class="value" id="cpu_pct">-</div></div>
    <div class="card"><div class="label">Memory</div><div class="value" id="mem_pct">-</div></div>
    <div class="card wide"><div class="label">Equity Curve</div><svg id="curve" viewBox="0 0 400 80" preserveAspectRatio="none"></svg></div>
    <div class="card wide"><div class="label">Recent Errors</div><div id="errors">-</div></div>
  </div>
  <script>
    function fmtTs(epoch) {
      if (!epoch) return 'n/a';
      const d = new Date(epoch * 1000);
      return d.toLocaleTimeString();
    }
    function drawCurve(points) {
      const svg = document.getElementById('curve');
      svg.innerHTML = '';
      if (!points || points.length < 2) {
        svg.innerHTML = '<text x="12" y="40" fill="#9aa7b8" font-size="12">No equity data</text>';
        return;
      }
      const vals = points.map(p => Number(p.equity) || 0);
      const min = Math.min(...vals), max = Math.max(...vals);
      const span = (max - min) || 1;
      const w = 400, h = 80, pad = 4;
      let d = '';
      points.forEach((p, i) => {
        const x = pad + (i / (points.length - 1)) * (w - pad * 2);
        const y = h - pad - ((Number(p.equity) - min) / span) * (h - pad * 2);
        d += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
      });
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d.trim());
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#5b9fd4');
      path.setAttribute('stroke-width', '2');
      svg.appendChild(path);
    }
    async function refresh() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const m = data.metrics || {};
        document.getElementById('market_status').textContent = m.market_status || '-';
        const ws = document.getElementById('ws_status');
        ws.textContent = m.ws_status || '-';
        ws.className = 'value ' + ((m.ws_status === 'connected') ? 'ok' : 'bad');
        document.getElementById('heartbeat').textContent = fmtTs(m.last_heartbeat_at);
        const c = m.current_candle;
        document.getElementById('candle').textContent = c
          ? `${c.timestamp} O:${c.open} H:${c.high} L:${c.low} C:${c.close}`
          : (m.last_candle_ts || '-');
        document.getElementById('today_signals').textContent = m.today_signals;
        document.getElementById('open_trades').textContent = m.open_trades;
        document.getElementById('closed_trades').textContent = m.closed_trades;
        document.getElementById('win_rate').textContent = ((m.win_rate || 0) * 100).toFixed(1) + '%';
        document.getElementById('running_pnl').textContent = Number(m.running_pnl || 0).toFixed(2);
        document.getElementById('avg_latency').textContent = Number(m.avg_signal_latency_ms || 0).toFixed(1) + ' ms';
        const db = document.getElementById('db_ok');
        db.textContent = m.db_ok ? 'OK' : 'FAIL';
        db.className = 'value ' + (m.db_ok ? 'ok' : 'bad');
        document.getElementById('cpu_pct').textContent = Number(m.cpu_pct || 0).toFixed(1) + '%';
        document.getElementById('mem_pct').textContent = Number(m.mem_pct || 0).toFixed(1) + '%';
        document.getElementById('errors').textContent = (m.recent_errors || []).slice(-10).join('\\n') || 'None';
        drawCurve(m.equity_curve || []);
      } catch (e) {
        document.getElementById('errors').textContent = 'Status fetch failed: ' + e;
      }
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def create_app(
    *,
    metrics: LiveMetrics,
    trade_manager: PaperTradeManager,
    db: PaperSignalDatabase,
) -> FastAPI:
    """Build the FastAPI application bound to shared runtime objects."""
    app = FastAPI(title="SmartMoneyEngine Live Paper", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        # Refresh trading stats + system gauges
        try:
            stats = trade_manager.stats()
            metrics.update_trading_stats(
                today_signals=int(stats["today_signals"]),
                open_trades=int(stats["open_trades"]),
                closed_trades=int(stats["closed_trades"]),
                win_rate=float(stats["win_rate"]),
                running_pnl=float(stats["running_pnl"]),
                equity_curve=list(stats["equity_curve"]),
            )
        except Exception as exc:  # noqa: BLE001
            metrics.record_error(f"stats_failed: {exc}")

        db_ok = True
        try:
            db.recent_signals(limit=1)
        except Exception as exc:  # noqa: BLE001
            db_ok = False
            metrics.record_error(f"db_ping_failed: {exc}")

        cpu_pct = 0.0
        mem_pct = 0.0
        try:
            import psutil

            cpu_pct = float(psutil.cpu_percent(interval=0.0))
            mem_pct = float(psutil.virtual_memory().percent)
        except Exception:  # noqa: BLE001
            pass

        metrics.set_system(db_ok=db_ok, cpu_pct=cpu_pct, mem_pct=mem_pct)
        payload: dict[str, Any] = {
            "metrics": metrics.snapshot(),
            "open_trades": [
                {
                    "signal_id": t.signal_id,
                    "timestamp": t.timestamp,
                    "direction": t.direction,
                    "entry": t.entry,
                    "stop": t.stop,
                }
                for t in trade_manager.list_open()
            ],
            "server_time": time.time(),
        }
        return JSONResponse(payload)

    return app
