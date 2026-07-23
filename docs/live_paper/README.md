# Live Paper Trading

Production-grade **paper-only** live trading stack for SmartMoneyEngine.

- Websocket ticks → 5-minute candles → frozen BUY_V3 / SELL_V6 signals
- SQLite persistence (existing schema)
- SMTP email alerts + FastAPI dashboard
- Missed-candle recovery via FYERS historical REST

## Docs

- [SETUP.md](SETUP.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
- [EMAIL.md](EMAIL.md) — primary notifications
- [TELEGRAM.md](TELEGRAM.md) — optional legacy (default off)
- [DASHBOARD.md](DASHBOARD.md)
- [LOGGING.md](LOGGING.md)
- [RECOVERY.md](RECOVERY.md)

## Quick start

```bash
pip install -r requirements.txt -r requirements-live-paper.txt
cp .env.example .env   # fill FYERS_* and SMTP_*
python -m src.live_paper.email_test   # verify SMTP
python -m src.live_paper
```

Dashboard: http://127.0.0.1:8080
