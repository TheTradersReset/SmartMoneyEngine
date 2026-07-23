# Missed Candle Recovery

After websocket reconnect or stale-tick detection during IST session:

1. Compute missed 5-minute bars since last closed candle
2. If gap > 1 bar, fetch FYERS **5m** history for the gap window only (REST allowed for recovery)
3. Skip timestamps already present in `candles` (UNIQUE)
4. Skip bars that already have accepted signals in `signals`
5. Feed remaining candles through `ingest_closed_candle` (same live path)
6. Engines use `emitted_bars`; trade manager dedupes by `(timestamp, direction)`

Recovery events are written to `reconnect.log` / `candle.log`.
