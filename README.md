# SmartMoneyEngine

Quantitative / SMC research and trading toolkit.

## FYERS Connectivity

Live FYERS API v3 auth + websocket tick streaming (connectivity test only).

See **[src/brokers/README.md](src/brokers/README.md)** for:

- `.env` keys and auth steps (app creation → auth code → access token → re-auth)
- Run commands (PowerShell / `.venv`)
- Unit tests

Quick start:

```powershell
.\.venv\Scripts\pip.exe install fyers-apiv3 python-dotenv
.\.venv\Scripts\python.exe -m src.brokers.fyers_client
.\.venv\Scripts\python.exe -m src.brokers.websocket_client
```
