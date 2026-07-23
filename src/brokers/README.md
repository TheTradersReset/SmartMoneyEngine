# FYERS Connectivity Test

Verify that SmartMoneyEngine can authenticate with the FYERS API v3 and stream live market ticks over websocket. This path is **connectivity only** — no order placement, signal engine, Telegram, SQLite, or paper trading.

## Files

| Path | Role |
|------|------|
| `src/brokers/fyers_client.py` | Load `.env`, validate token, profile / quotes helpers |
| `src/brokers/websocket_client.py` | Data websocket, `NSE:NIFTY50-INDEX` subscribe, reconnect, tick print |
| `tests/test_fyers_client.py` | Unit tests (mocked; no live API) |
| `tests/test_websocket_client.py` | Unit tests (mocked reconnect / subscribe) |

## Prerequisites

```powershell
.\.venv\Scripts\pip.exe install fyers-apiv3 python-dotenv
```

Copy `.env.example` to `.env` and fill in real values (never commit secrets).

```env
FYERS_APP_ID=your_app_id-100
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=http://127.0.0.1:8000
FYERS_ACCESS_TOKEN=
FYERS_PIN=
```

`FYERS_CLIENT_ID` is accepted as an alias for `FYERS_APP_ID`.  
If `FYERS_ACCESS_TOKEN` is empty, the client also accepts `data/tokens/fyers_token.json` (same format produced by `src/brokers/fyers/auth.py`).

---

## Authentication steps (FYERS API v3)

### 1. Create a FYERS app

1. Log in at [FYERS API Dashboard](https://myapi.fyers.in/dashboard).
2. Create an app and note **App ID** / **Client ID** and **Secret Key**.
3. Set the redirect URI to match `.env`, e.g. `http://127.0.0.1:8000`.
4. Put values into `.env` as `FYERS_APP_ID`, `FYERS_SECRET_KEY`, `FYERS_REDIRECT_URI`.

### 2. Generate an auth code (login URL)

Option A — interactive flow already in the repo:

```powershell
.\.venv\Scripts\python.exe -m src.brokers.fyers.auth
```

This opens the FYERS login page, captures the redirect auth code, exchanges it for a token, and writes `data/tokens/fyers_token.json`.

Option B — manual:

```powershell
.\.venv\Scripts\python.exe -c "from src.brokers.fyers_client import load_credentials, generate_login_url; print(generate_login_url(load_credentials()))"
```

Open the printed URL, log in, and copy the `auth_code` query parameter from the redirect URL.

### 3. Exchange auth code for access token (v3)

If you used Option A, the token file is already saved.

Manual exchange example:

```powershell
.\.venv\Scripts\python.exe -c "from src.brokers.fyers_client import load_credentials, exchange_auth_code, save_access_token; c=load_credentials(); t=exchange_auth_code(c, 'PASTE_AUTH_CODE'); print(save_access_token(t)); print(t.get('access_token','')[:20], '...')"
```

Optionally set `FYERS_ACCESS_TOKEN=<access_token>` in `.env` (raw token only; do not commit it).

### 4. Refresh / re-auth when expired

FYERS access tokens expire (typically daily). When profile/quotes/websocket fail with auth errors:

1. Re-run `.\.venv\Scripts\python.exe -m src.brokers.fyers.auth`, **or**
2. Repeat steps 2–3 and update `FYERS_ACCESS_TOKEN` / `data/tokens/fyers_token.json`.

There is no long-lived refresh token in this connectivity path — re-authenticate when the token expires.

### 5. Validate REST connectivity

```powershell
.\.venv\Scripts\python.exe -m src.brokers.fyers_client
```

Expected: profile printed and (when the market is open) a NIFTY50 quote.

### 6. Stream live ticks

```powershell
.\.venv\Scripts\python.exe -m src.brokers.websocket_client
```

Subscribes to `NSE:NIFTY50-INDEX`, prints `TICK #n: {...}` lines, auto-reconnects with exponential backoff on disconnect. Press **Ctrl+C** for a clean shutdown.

---

## Unit tests (no live API)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fyers_client.py tests/test_websocket_client.py -q --tb=short
```

---

## Notes

- Websocket token format used internally: `APP_ID:ACCESS_TOKEN` (built automatically).
- Structured logs go to the console and `logs/engine.log` via `src.core.logger`.
- Do **not** commit `.env`, token JSON, or real secrets.
