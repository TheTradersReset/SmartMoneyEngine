# Real-Time Candle & Signal Pipeline — SmartMoneyEngine

**Status:** Paper signal mode only  
**Capital mode:** `paper_signal` — **NO orders, NO Telegram, NO live capital, NO auto-trading**  
**Stack fingerprint:** `BUY_V3|SELL_V6|fixed_10|60/100/Runner|RegimeThrottle`

---

## 0. Purpose

Convert FYERS live ticks into real-time 5-minute candles and evaluate the **frozen production candidate** on every candle close:

| Component | Locked value |
|-----------|--------------|
| Buy engine | **BUY_V3** (`LDM-BUY-V3`) |
| Sell engine | **SELL_V6** (`LDM-SELL-V6`) |
| Stop | **fixed_10** (±10 points) |
| Targets | **60 / 100 / Runner** |
| Regime | **Regime Throttle ON** |

Forbidden: BUY_V4, SELL_V7, new indicators, new signal engines, models, discovery engines.

---

## 1. Architecture

```text
FYERS WebSocket (ticks)
        │
        ▼
FiveMinuteCandleBuilder  ──► SQLite candles table
        │ (on 5M close)
        ▼
MarketContextService     ──► enriched frames + intel maps
        │
        ├──► BuyV3Engine.evaluate_bar()
        └──► SellV6Engine.evaluate_bar()
        │
        ▼
RegimeThrottle           ──► FULL / HALF / QUARTER / BLOCK
        │
        ▼
Same-bar conflict check  ──► reject both if BUY+SELL same bar
        │
        ▼
SignalObject + SQLite signals / signal_events
        │
        ▼
Structured JSON log (stdout) + logs/engine.log
```

### Module map

| Path | Role |
|------|------|
| `src/brokers/websocket_client.py` | FYERS v3 websocket connectivity + reconnect |
| `src/data/candle_builder.py` | Tick parse + IST 5M OHLCV aggregation |
| `src/pipeline/market_context_service.py` | Replay-compatible context builder |
| `src/signals/buy_v3.py` | BUY_V3 production wrapper |
| `src/signals/sell_v6.py` | SELL_V6 production wrapper |
| `src/signals/regime_throttle.py` | Regime throttle from `regime_detection_audit.json` |
| `src/storage/sqlite.py` | Candles + signals + audit events |
| `src/pipeline/realtime_signal_pipeline.py` | Orchestrator + CLI entry |

---

## 2. Data flow

### 2.1 Tick ingestion

- Source: FYERS data websocket (`NSE:NIFTY50-INDEX`)
- Parser: `parse_fyers_tick()` expects `ltp` / `symbol` fields
- Session filter: NSE cash hours 09:15–15:30 IST (weekdays)

### 2.2 Candle construction

- Bucket alignment: floor timestamp to 5-minute IST boundary
- Close trigger: first tick in a **new** bucket closes the prior candle
- Persist: `candles` table in `data/paper/realtime_signals.db`
- Warm-start: optional history from `outputs/pipeline/NIFTY50_5m_pipeline.csv`

### 2.3 Signal evaluation (candle close only)

On each closed 5M bar (minimum **120 bars** warm context):

1. Append candle to rolling frame
2. Rebuild enriched context (5M / 15M / 1H / 1D intel)
3. Run BUY_V3 fast path + SELL_V6 `evaluate_bar()`
4. Build `SignalObject` with fixed_10 + 60/100/Runner levels
5. Apply regime throttle
6. Reject if same-bar BUY+SELL conflict
7. Persist accepted/rejected signals + audit events

**No orders are placed.** Signals are logged and stored only.

---

## 3. Signal object schema

| Field | Description |
|-------|-------------|
| `timestamp` | Candle close timestamp (IST) |
| `direction` | `BUY` or `SELL` |
| `entry` | Signal entry (bar close) |
| `stop` | Entry ± 10 points |
| `target1` | 60 points from entry |
| `target2` | 100 points from entry |
| `target_structure` | `60/100/Runner` |
| `confidence` | 1.0 when layer5 pass, else 0.5 |
| `regime` | Composite regime key |
| `engine_version` | `BUY_V3` or `SELL_V6` |
| `throttle_level` | FULL / HALF / QUARTER / BLOCK |
| `accepted` | Whether signal passed throttle + conflict checks |
| `rejection_reason` | e.g. `REGIME_BLOCK:...`, `SAME_BAR_CONFLICT` |

### Level math

**BUY:** `stop = entry - 10`, `target1 = entry + 60`, `target2 = entry + 100`  
**SELL:** `stop = entry + 10`, `target1 = entry - 60`, `target2 = entry - 100`

---

## 4. Regime throttle

- Rules loaded from `outputs/research/regime_detection_audit.json` when present
- Fallback: all regimes `FULL` if export missing
- `BLOCK` → signal stored as rejected (`REGIME_BLOCK`)
- `HALF` / `QUARTER` → accepted with weight annotation (paper sizing future work)

---

## 5. SQLite schema

**Database:** `data/paper/realtime_signals.db`

### `candles`

`symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_count`

### `signals`

`timestamp`, `direction`, `engine_version`, `entry`, `stop`, `target1`, `target2`, `target_structure`, `confidence`, `regime`, `throttle_level`, `accepted`, `rejection_reason`, `raw_json`

### `signal_events`

Audit trail: `SIGNAL_GENERATED`, `SIGNAL_ACCEPTED`, `SIGNAL_REJECTED`, `SAME_BAR_CONFLICT`

---

## 6. Structured logging

- Logger: `src/core/logger.py` → `logs/engine.log` + stderr
- Signal emission: JSON line to stdout per signal
- Candle close: INFO with OHLC summary
- Throttle / conflict: WARNING with reason codes

---

## 7. Run commands (Windows PowerShell)

### Prerequisites

```powershell
.\.venv\Scripts\pip.exe install fyers-apiv3 python-dotenv
```

Configure `.env` (see `src/brokers/README.md`):

```text
FYERS_APP_ID=...
FYERS_SECRET_KEY=...
FYERS_REDIRECT_URI=...
FYERS_ACCESS_TOKEN=...
```

### Warm-start historical context (recommended once)

Ensure pipeline CSV exists:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.market_pipeline
```

### Start paper signal pipeline

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.realtime_signal_pipeline
```

Press `Ctrl+C` to stop. Active candle is flushed on shutdown.

### Run unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_candle_builder.py tests/test_realtime_signal_pipeline.py -q --tb=short
```

---

## 8. Market session behaviour

| Phase | Behaviour |
|-------|-----------|
| Pre-market | Websocket may connect; candles outside 09:15–15:30 IST skipped |
| Market open | First tick starts 09:15 bucket |
| Intraday | Evaluate signals on each 5M close |
| Market close | Last bucket flushed on shutdown |
| Reconnect | FYERS SDK reconnect + outer backoff; missed candles backfill via history CSV on restart |
| Holidays | No session ticks → no candles → no signals |

---

## 9. Risk controls (signal layer)

| Control | Implementation |
|---------|----------------|
| Paper only | No order API calls |
| Regime BLOCK | Signal rejected before persistence as accepted |
| Same-bar conflict | Both BUY and SELL rejected on same bar |
| Duplicate bar | Engine `emitted_bars` set prevents re-fire |
| Min context | Requires ≥120 bars before evaluation |

---

## 10. What is NOT included (by design)

- Telegram notifications
- Trade lifecycle / MFE / MAE tracking (see `paper_trading_framework.md`)
- Order placement or capital deployment
- BUY_V4 / SELL_V7 engines

---

## 11. Operational checklist

- [ ] FYERS token valid in `.env`
- [ ] `outputs/pipeline/NIFTY50_5m_pipeline.csv` present for warm-start
- [ ] `outputs/research/regime_detection_audit.json` present (optional; defaults to FULL)
- [ ] `data/paper/` directory writable
- [ ] Run during market hours for live ticks
- [ ] Confirm stdout JSON signals + `logs/engine.log` entries
- [ ] Verify `data/paper/realtime_signals.db` rows after session

---

## 12. Related documents

- `paper_trading_framework.md` — full paper trading system design
- `src/brokers/README.md` — FYERS auth + websocket connectivity
- `outputs/research/live_deployment_validation_framework.json` — promotion gates

---

*Generated for SmartMoneyEngine frozen production candidate: BUY_V3 + SELL_V6 + fixed_10 + 60/100/Runner + Regime Throttle.*
