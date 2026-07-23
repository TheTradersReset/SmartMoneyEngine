# Deployment

## Local process

```bash
python -m src.live_paper
```

Graceful shutdown: Ctrl+C / SIGTERM.

## Docker

```bash
docker compose up --build
```

Mounts:

- `.env` (read-only)
- `data/`, `logs/`, `outputs/`
- `config/` (read-only)

Exposes dashboard port **8080**.

## Safety

- `LIVE_PAPER_CAPITAL_MODE` must be `paper`
- Entrypoint never calls place / modify / cancel order APIs
- Only market-data websocket + historical REST for gap recovery
