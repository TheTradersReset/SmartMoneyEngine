# Signal Decision vs Outcome Separation

## 1. Updated execution diagram

```text
Closed Candle (Replay / Live)
        │
        ▼
RealtimeSignalPipeline.ingest_closed_candle
        │
        ├─► BUY_V3 / SELL_V6 evaluate_bar
        │         Layer1 → Layer2 → Layer3 → Layer5
        │         Layer4 realtime plan (structural stop/risk)
        │         ⛔ does NOT require future candles
        │         ⛔ _trade_outcome never blocks emission
        │
        ├─► verdict BUY / SELL when Layer5 passes + Layer4 plan exists
        │
        ├─► to_signal() → persist signals row (outcome=PENDING)
        │         decision timestamp = signal bar time (unchanged forever)
        │
        └─► continue replay / next candles …
                  │
                  ▼
         after FORWARD_BARS (80) elapse
                  │
                  ▼
         evaluate_post_signal_outcome()
                  │  uses existing _trade_outcome(frame, signal_bar, side)
                  ▼
         UPDATE signals SET
            entry, stop, target, risk, reward,
            outcome ∈ {WIN, LOSS, BREAKEVEN},
            holding_bars, outcome_timestamp
         WHERE timestamp = decision_timestamp
           AND direction = BUY|SELL
```

## 2. Files modified

| File | Change |
|------|--------|
| `src/research/smartmoneyengine_v3_implementation_validation_research.py` | SELL `_layer4_execution` emits realtime plan; forward outcome optional |
| `src/research/buy_v2_candidate_validation_research.py` | BUY `_layer4_execution` same separation |
| `src/research/nifty50_liquidity_direction_decision_matrix_research.py` | Document `_trade_outcome` as post-signal only |
| `src/signals/signal_outcome.py` | **New** — classify + post-signal evaluator + Layer4 plan builder |
| `src/pipeline/realtime_signal_pipeline.py` | Pending queue; resolve after `FORWARD_BARS`; persist `PENDING` |
| `src/storage/sqlite.py` | Outcome columns + `update_signal_outcome` + migration |
| `src/storage/async_db_writer.py` | `signal_outcome` write op |
| `tests/test_signal_outcome_separation.py` | **New** unit tests |
| `signal_decision_outcome_separation.md` | This document |

**Not modified (by design):**

- `src/signals/buy_v3.py` / `src/signals/sell_v6.py` (wrappers / strategy surface)
- `src/replay/*` (Replay Engine)
- Layer1–5 strategy gates / formula rules

## 3. Migration impact

Existing SQLite signal DBs gain columns via `ALTER TABLE` on open:

- `risk`, `reward`, `target`, `outcome`, `holding_bars`, `outcome_timestamp`

New signals insert with `outcome='PENDING'` (accepted) or `'REJECTED'`.

Old rows remain readable; outcome fields stay `NULL` until a new run resolves them.

No Replay Engine API change. Campaign/replay CLIs unchanged.

## 4. Why this preserves strategy logic

- Layer1–5 predicates are untouched (events, HTF, VWAP, EMA, volume, Mid-Range, etc.).
- BUY_V3 / SELL_V6 wrappers still require `verdict == BUY|SELL` and the same entry/stop/target paper geometry.
- `_trade_outcome` math (structural stop, MFE/MAE, realized PnL, hit_Nr) is unchanged; it only moved to **after** emission.
- Research frames that already contain future bars still optionally attach `forward_outcome` on Layer4 for offline reports — without making emptiness a hard gate.

## 5. Unit tests

```bash
python -m pytest tests/test_signal_outcome_separation.py -q
```

Coverage:

- Layer4 SELL plan on last bar when `_trade_outcome` returns `{}`
- `evaluate_bar` → `verdict=SELL` → `to_signal()` without forward bars
- Post-signal classification WIN/LOSS/BREAKEVEN
- SQLite update preserves decision `timestamp`, writes `outcome_timestamp`

## 6. Replay verification procedure

1. Point a scratch DB at a fresh path (do not overwrite campaign evidence DBs unless intentional).
2. Replay one known day that previously had `sell_score==100` and Layer5 pass, e.g. `2026-03-05`:

```bash
python -m src.replay.engine --day 2026-03-05 --signal-db data/paper/verify_outcome_sep.db --speed unlimited
```

3. Immediately after the run (or mid-run once signals print):

```sql
SELECT timestamp, direction, accepted, outcome, risk, reward, holding_bars, outcome_timestamp
FROM signals
ORDER BY id;
```

Expect:

- Rows with `direction='SELL'` (and/or `BUY`) appear **at decision time** with `outcome='PENDING'` (accepted path).
- After ≥ `FORWARD_BARS` subsequent candles in the same continuous frame, those rows update to `WIN`/`LOSS`/`BREAKEVEN` with `outcome_timestamp` set and **unchanged** `timestamp`.

4. Confirm decisions still log every candle; signal count > 0 iff Layer5+throttle accept (no longer blocked by empty forward window).

5. Optional: compare Layer5 `reason_codes` on `signal_decisions` against pre-fix audit — gate codes must match; only Layer4 blocking behavior changes.

**Day-by-day note:** Pending outcomes are hydrated from SQLite on each warm-start so a later day can resolve signals once `FORWARD_BARS` exist in the growing frame — Replay Engine code is unchanged.
