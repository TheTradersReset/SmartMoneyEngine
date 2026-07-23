# Logging

## Component logs

Directory: `logs/live_paper/`

| File | Component |
|------|-----------|
| websocket.log | ticks / WS status |
| candle.log | candle close / recovery ingest |
| signal.log | accepted signals / latency / outcomes |
| database.log | DB persistence notes |
| email.log | SMTP email send path |
| telegram.log | Telegram send path (optional/legacy) |
| errors.log | operational errors |
| reconnect.log | reconnect + recovery |

Format: `timestamp | LEVEL | component | message`

## Engine log

Main pipeline still writes to `logs/engine.log` via `src.core.logger`.
