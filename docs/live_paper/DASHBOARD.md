# Dashboard

FastAPI app served by uvicorn in a daemon thread.

- UI: `GET /`
- JSON: `GET /api/status`
- Auto-refresh every 2000 ms

## Fields

Market Status, WS Status, Heartbeat, Current Candle, Today's Signals, Open/Closed Trades, Win Rate, Running PnL, Equity Curve (SVG), Avg Latency, DB Status, CPU, Memory, Recent Errors

## Bind

- `LIVE_PAPER_DASHBOARD_HOST` (default `0.0.0.0`)
- `LIVE_PAPER_DASHBOARD_PORT` (default `8080`)
- `LIVE_PAPER_ENABLE_DASHBOARD=true|false`
