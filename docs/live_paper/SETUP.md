# Setup

## Prerequisites

- Python 3.11+
- Valid FYERS app credentials and access token (`FYERS_*` in `.env`)
- SMTP credentials for email alerts (`SMTP_*` in `.env`) — see [EMAIL.md](EMAIL.md)
- Optional: Telegram bot token + chat id (default **off**)

## Install

```bash
pip install -r requirements.txt -r requirements-live-paper.txt
```

## Configure

1. Copy `.env.example` → `.env`
2. Set `FYERS_APP_ID`, `FYERS_SECRET_KEY`, `FYERS_REDIRECT_URI`
3. Authenticate so `data/tokens/fyers_token.json` exists (existing FYERS auth flow)
4. Set `SMTP_HOST`, `SMTP_FROM`, `SMTP_TO` (and usually `SMTP_USER` / `SMTP_PASSWORD`)
5. Confirm `LIVE_PAPER_ENABLE_EMAIL=true` and `LIVE_PAPER_ENABLE_TELEGRAM=false`
6. Confirm `LIVE_PAPER_CAPITAL_MODE=paper` (required)

Optional YAML defaults: `config/live_paper/live_paper.yaml`

## Verify email

```bash
python -m src.live_paper.email_test
```

## Warm-start history

Default CSV: `outputs/pipeline/NIFTY50_5m_pipeline.csv`  
Override with `LIVE_PAPER_HISTORY_CSV`.

## Run

```bash
python -m src.live_paper
```
