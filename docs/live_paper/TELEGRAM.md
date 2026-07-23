# Telegram Alerts (optional / legacy)

Telegram is **disabled by default** (`LIVE_PAPER_ENABLE_TELEGRAM=false`) because many corporate networks block `api.telegram.org`. Prefer [EMAIL.md](EMAIL.md) for production alerts.

## Env

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `LIVE_PAPER_ENABLE_TELEGRAM=true|false` (default false)

If token/chat are missing, Telegram stays disabled with a warning (no crash).

## Test

```bash
python -m src.live_paper.telegram_test
```
