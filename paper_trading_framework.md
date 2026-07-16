# FYERS Paper Trading Framework — SmartMoneyEngine

**Status:** Design / runbook (signal generation & tracking only)  
**Capital mode:** `paper` only — **NO live orders, NO production capital, NO auto-trading**  
**Evidence source:** `outputs/research/live_deployment_validation_framework.json`  
**Generated for:** NIFTY50 · 5-minute bars · IST session

---

## 0. Locked stack (non-negotiable)

| Component | Locked value | Model / note |
|-----------|--------------|--------------|
| Buy engine | **BUY_V3** | `LDM-BUY-V3` (`src/research/buy_v3_candidate_validation_research.py`) |
| Sell engine | **SELL_V6** | `LDM-SELL-V6` (`src/research/walk_forward_failure_root_cause_audit_research.py`) |
| Stop | **fixed_10** | Entry ± 10 points |
| Targets | **60 / 100 / Runner** | T1=60, T2=100, runner leg ON; leg weights ⅓ / ⅓ / ⅓ |
| Regime | **Regime Throttle ON** | FULL / HALF / QUARTER / BLOCK (`src/research/regime_detection_audit_research.py`) |
| Conflict policy | **NO_TRADE** | Same-bar opposing BUY+SELL → no paper fill |
| Forbidden | BUY_V4, SELL_V7, new indicators, new signal engines, discovery engines | `DO_NOT_PROMOTE` |

**Stack fingerprint (must appear on every session row):**

```text
BUY_V3|SELL_V6|fixed_10|60/100/Runner|RegimeThrottle
```

**Paper trading verdict (from validation framework):** CONDITIONAL start allowed with kill-switches armed.  
**Real capital readiness:** **NO** until 20/40/60-session gates pass.

**Replay baseline (240d throttled combined — targets, NOT live proof):**

| Metric | Value |
|--------|------:|
| Win rate | 69.01% |
| Profit factor | 5.58 |
| Expectancy | 108.05 pts |
| Max DD | 2424.26 pts |
| Signals / month | 63.89 |

---

## 1. Purpose and scope

Convert the proven **replay** stack into a **live paper** system that:

1. Ingests FYERS market data (REST + WebSocket).
2. Builds real-time 5-minute candles.
3. Emits **BUY_V3** and **SELL_V6** signals only.
4. Applies **Regime Throttle** and same-bar conflict rules.
5. Simulates paper fills (intended price ± tracked slippage) — **never sends live orders**.
6. Logs signals, trades, MFE/MAE, slippage, lifecycle, and audit events to **SQLite**.
7. Notifies operators via **Telegram**.
8. Feeds a monitoring dashboard and 20 / 40 / 60 session promotion gates.

This document is the primary deliverable. Implementation packages listed under Deployment Architecture are **proposed** wrappers around existing repo modules; do not invent conflicting engine IDs.

---

## 2. Core requirements

### 2.1 Live market data ingestion (FYERS API + WebSocket)

| Concern | Design |
|---------|--------|
| Auth | Reuse `src/brokers/fyers/auth.py` + `src/brokers/fyers/config.py` (`.env`: `FYERS_APP_ID`, `FYERS_SECRET_KEY`, `FYERS_REDIRECT_URI`) |
| Token | `data/tokens/fyers_token.json` via `src/brokers/fyers/client.py` (`FyersClient`) |
| History backfill | `src/brokers/fyers/historical.py` → `data/historical/` (IST `Asia/Kolkata`) |
| Live ticks | Proposed `src/paper/fyers_ws.py` — FYERS WebSocket subscribe on NIFTY50 index symbol; **paper-only** (no order channel) |
| REST health | Periodic `quotes` / history checksum against WS candle builder |

**Rules:**

- Paper mode flag `capital_mode=paper` must be set before any WS connect.
- Order / place / modify / cancel APIs must be **hard-disabled** in paper config (`enable_live_orders=false`).
- On auth failure: halt session, Telegram alert, no discretionary re-entry.

### 2.2 Real-time 5-minute candle builder

Proposed: `src/paper/candle_builder.py`

- Aggregate ticks into OHLCV buckets aligned to IST 5-minute boundaries (09:15, 09:20, … 15:25).
- Emit `bar_close` only when the bucket is complete (or session forced-close).
- Persist incomplete bar state for reconnect recovery.
- Validate against FYERS 5m history on reconnect (missed-candle recovery — §3.8).
- Feed closed bars into the same feature path used by research replay (pipeline-compatible columns: timestamp, open, high, low, close, volume).

Reference historical path already in repo: `outputs/pipeline/NIFTY50_5m_pipeline.csv` (replay), `data/historical/` (FYERS downloads).

### 2.3 BUY_V3 signal generation

| Item | Value |
|------|-------|
| Engine | BUY_V3 only |
| Model ID | `LDM-BUY-V3` |
| Research source | `src/research/buy_v3_candidate_validation_research.py` |
| Paper wrapper | Proposed `src/paper/engines/buy_v3_live.py` — call existing BUY_V3 candidate logic on closed 5m bars; **no V4** |

On each closed bar: evaluate BUY_V3 → if fire, emit signal event with intended entry = bar close, stop = entry−10, T1/T2/runner levels.

### 2.4 SELL_V6 signal generation

| Item | Value |
|------|-------|
| Engine | SELL_V6 only |
| Model ID | `LDM-SELL-V6` |
| Research sources | `src/research/sell_v6_replay_validation_research.py`, walk-forward root-cause audit |
| Paper wrapper | Proposed `src/paper/engines/sell_v6_live.py` — **no V7** |

Same bar clock as BUY_V3. Intended entry = bar close; stop = entry+10 for shorts.

### 2.5 Regime Throttle integration

Source: `src/research/regime_detection_audit_research.py`

| Level | Weight | Paper action |
|-------|-------:|--------------|
| FULL | 1.0 | Accept signal at full paper size |
| HALF | 0.5 | Accept at half size |
| QUARTER | 0.25 | Accept at quarter size |
| BLOCK | 0.0 | **Reject** — log `throttle_action=BLOCK`, **zero fills** |

- Throttle primarily gates **SELL_V6** by labeled regime (BUY may continue per playbook unless portfolio kill-switch fires).
- Any SELL fill in BLOCK regime → **KS-3** hard halt (see Risk Control Layer).
- Proposed module: `src/paper/regime_throttle.py` wrapping the audited throttle map (export: `outputs/research/regime_detection_audit.json`).

### 2.6 Signal logging

Every signal (accepted or rejected) writes to:

1. SQLite `signals` table (§6)
2. Append-only JSONL journal: `outputs/paper/journals/signals_YYYYMMDD.jsonl`
3. Optional mirror under `logs/paper_signals.log`

Logged fields: timestamp, side, engine, version, bar_ts, intended_entry, SL, T1, T2, runner, regime, throttle_action, conflict_flag, accept/reject + reason.

### 2.7 SQLite trade database

Path: `data/paper/paper_trading.db`  
Schema: §6. WAL mode, daily backup to `outputs/paper/backups/`.

### 2.8 Telegram notifications

Proposed: `src/paper/telegram_notifier.py`  
Config: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` in `.env` (never commit).  
Message catalog: §8.

### 2.9 Trade lifecycle tracking

Reuse concepts from `src/signals/setup_lifecycle_engine.py` for stage vocabulary; paper trade lifecycle states:

```text
SIGNAL → ACCEPTED | REJECTED → OPEN → T1_HIT → T2_HIT → RUNNER → CLOSED
                              ↘ STOPPED / SESSION_FLAT / KILL_FLAT
```

Proposed: `src/paper/trade_lifecycle.py`

### 2.10 MFE tracking

On every tick / bar update while position open:

- Long: `mfe = max(mfe, high - entry_fill)`
- Short: `mfe = max(mfe, entry_fill - low)`

Persist peak MFE on trade row; used for capture % and target accuracy.

### 2.11 MAE tracking

- Long: `mae = max(mae, entry_fill - low)`
- Short: `mae = max(mae, high - entry_fill)`

Stop accuracy gate uses MAE ≥ 10 (fixed_10).

### 2.12 Slippage tracking

**Definition** (from validation framework):

```text
slippage_pts = sum(|fill_price - intended_price|) for entry and each exit leg
```

| Gate | Threshold |
|------|----------:|
| Median entry+exit (pass) | ≤ 5.0 pts |
| P90 trade slippage (pass) | ≤ 8.0 pts |
| Kill (median rolling) | > 10.0 pts |

Paper broker: intended = signal bar close / target / stop; fill = simulated or FYERS paper fill if used — still **no live capital**.

### 2.13 Same-bar conflict tracking

If BUY_V3 and SELL_V6 both fire on the same 5m bar:

- Policy: **NO_TRADE**
- Log conflict; **zero fills**
- Compliance must be **100%**
- Alert if conflict rate > 12% of union signal bars; soft→hard if > 20% over 10 sessions (KS-9)

---

## 3. Market session management

All times **Asia/Kolkata (IST)**. Symbol: NIFTY50. Timeframe: 5M.

### 3.1 Market open

| Step | Action |
|------|--------|
| T−30m (≈08:45) | Pre-flight: token valid, SQLite healthy, Telegram ping, kill-switches armed, stack fingerprint assert |
| T−5m (≈09:10) | WS connect; seed candle builder from FYERS 5m history (last N bars) |
| 09:15 | Session OPEN; `sessions` row created (`session_id=YYYYMMDD-NIFTY50-5M`) |
| First closed bar | Engines evaluate; no overnight position carry unless explicitly configured (default: flat overnight) |

### 3.2 Market close

| Step | Action |
|------|--------|
| 15:25–15:30 | Last bars; no new entries after configured cutoff (recommend 15:15 last entry) |
| EOD | Flatten any open paper runner if playbook requires session flat; compute daily rollup |
| Post | Write session rollup; backup DB; Telegram session summary |

### 3.3 Pre-market

- Holiday check (§3.6)
- Refresh FYERS token if near expiry (`src/brokers/fyers/auth.py`)
- Load regime throttle map snapshot
- Verify `enable_live_orders=false`
- Clear previous-day incomplete bars unless recovery needed

### 3.4 Post-market

- Archive journals to `outputs/paper/sessions/YYYYMMDD/`
- Run daily metrics job (WR, PF, expectancy, slippage, capture)
- Update promotion scorecard counters (sessions completed)
- No WS reconnect after session marked CLOSED unless operator force-restart

### 3.5 Connection recovery

On disconnect / checksum failure (**KS-12**):

1. Cancel any *paper* working orders / mark uncertain state.
2. Flat if position state uncertain (paper flat preferred over ghost position).
3. Telegram `CONNECTION_FAILURE`.
4. Attempt reconnect with exponential backoff (e.g. 1s, 2s, 5s, 10s, 30s; max 5 attempts then halt).
5. **No discretionary re-entry** mid-bar after uncertain gap.

### 3.6 WebSocket reconnect

Proposed `src/paper/fyers_ws.py` reconnect loop:

1. Re-auth if needed.
2. Resubscribe symbol.
3. Request REST 5m candles covering gap → **missed candle recovery**.
4. Replay closed bars through engines in order (deterministic); do not double-open trades (idempotent `signal_id` / `bar_ts` unique constraint).
5. Resume live ticks only after gap filled and checksum OK.

### 3.7 Holiday handling

- Maintain NSE holiday calendar file: `config/paper/nse_holidays.json` (proposed).
- If holiday or special session: skip session creation; Telegram info; exit 0.
- Half-days: use shortened close; last-entry cutoff adjusted.

### 3.8 Session restart

Operator / watchdog restart mid-day:

1. Load open trades from SQLite.
2. Rebuild candle state from REST.
3. Reconcile MFE/MAE from bar path since entry.
4. Resume only if `kill_switch` not active and stack fingerprint matches.

### 3.9 Missed candle recovery

1. Detect gap: expected 5m stamps vs received.
2. Fill via `FyersClient` / `src/brokers/fyers/historical.py` for the gap window.
3. Process each recovered closed bar through BUY_V3 / SELL_V6 / throttle / conflict **in timestamp order**.
4. Log `event_type=MISSED_CANDLE_RECOVERY` with bar count.
5. If gap > configurable max (e.g. 6 bars) during open position: prefer flat + halt (data integrity).

---

## 4. Risk control layer

Thresholds from `outputs/research/live_deployment_validation_framework.json` → `risk_controls`.

### 4.1 Kill switches (armed in paper)

| ID | Trigger | Action | Severity |
|----|---------|--------|----------|
| **KS-1** | Portfolio daily loss **> 593.79 pts** | Flat all sleeves; halt new entries for session | HARD |
| **KS-2** | Median trade slippage **> 10.0 pt** over any rolling **5** sessions | Halt promotion; shadow-only; re-calibrate | HARD |
| **KS-3** | Any SELL fill in labeled **BLOCK** regime | Halt SELL sleeve; RCA before resume | HARD |
| **KS-4** | Same-bar conflict **fill** (policy breach) | Halt combined engine; fix router | HARD |
| **KS-5** | Consecutive losses **≥ 7** (paper) / ≥5 real | Pause new entries 1 full session | HARD |
| **KS-6** | Rolling 20-session combined PF **< 1.2** | Demote tier / paper-only; no scale-up | HARD |
| **KS-7** | Rolling 20-session SELL PF **< 1.5** | Stop SELL; BUY half size pending review | HARD |
| **KS-8** | Missed entry rate **> 15%** over 10 sessions | Halt until order path fixed | HARD |
| **KS-9** | Conflict rate **> 20%** of union bars over 10 sessions | Review clocks; no capital increase | SOFT→HARD if PF degrading |
| **KS-10** | Stack drift (V4/V7, non-fixed_10, non-60/100/Runner) | Immediate halt; reset LOCKED_STACK | HARD |
| **KS-11** | Intraday DD **> 400 pts** (paper) or **> 4%** capital (real) | Flat + session halt | HARD |
| **KS-12** | Broker/API disconnect or bar-feed checksum failure | Cancel working; flat if uncertain; no discretionary re-entry | HARD |

### 4.2 Daily loss kill switch

- Limit: **593.79 points** portfolio (KS-1).
- Measured on realized paper PnL for `session_date`.
- Breach → Telegram `KILL_SWITCH`, `daily_loss_limit_breached=true`.

### 4.3 Consecutive loss kill switch

- Paper: **7** consecutive closed losses (KS-5).
- Counter resets on a winning closed trade.
- Real-capital tiers (future): 50K→6, 1L/2L→5 — **not used while paper-only**.

### 4.4 Slippage kill switch

| Metric | Pass | Kill |
|--------|-----:|-----:|
| Median pts | ≤ 5.0 | rolling-5 median > 10.0 |
| P90 pts | ≤ 8.0 | alert / block promotion |

### 4.5 Regime BLOCK enforcement

- Throttle map must yield `throttle_action=BLOCK` and **zero fills** for blocked SELL.
- `throttle_violations` must remain **0**.
- Shadow compare vs labeled regimes each session.

### 4.6 Same-bar conflict protection

- Router: if both sides fire → reject both; `conflict_flag=true`.
- Policy compliance **100%** mandatory.
- Alert threshold: conflict rate > **12%**.

### 4.7 Signal cooldown

Proposed defaults (paper config):

- Per-side cooldown: **1 bar** after reject/conflict (avoid spam).
- After stop-out: **1 bar** before new entry same side.
- After KS-5 pause: **1 full session**.

### 4.8 Circuit breaker

Trip when any HARD KS fires or when:

- Feed silence > **60s** during market hours, or
- > **3** reconnect failures in 10 minutes.

State: `CIRCUIT_OPEN` → no new signals accepted until operator reset + checklist.

### 4.9 Data feed failure protection

- Heartbeat on WS + REST quote poll.
- Checksum: last closed bar OHLC vs FYERS history.
- On failure: KS-12 path; do not invent bars.

### 4.10 Drawdown caps (paper)

| Cap | Points |
|-----|-------:|
| Soft | 800 |
| Hard kill | 1200 |
| Intraday paper proxy | 400 |

---

## 5. Audit trail

Every material event appends to SQLite `events` and JSONL.

| Event | Required fields |
|-------|-----------------|
| Signal Generated | signal_id, side, engine, bar_ts, intended prices, regime |
| Signal Accepted | signal_id, throttle_action, size_weight |
| Signal Rejected | signal_id, reason (`BLOCK`, `CONFLICT`, `COOLDOWN`, `KILL_SWITCH`, `CIRCUIT`, `SESSION_CUTOFF`, …) |
| Trade Open | trade_id, entry_intended, entry_fill, entry_slippage |
| Target / Stop / Runner | leg, fill, slippage, ts |
| Trade Close | exit, pnl, mfe, mae, capture_pct, duration |
| Regime | regime_label, throttle_action |
| MFE / MAE | running + final |
| PnL | points (+ optional INR null in paper) |
| Slippage | per leg + trade total |
| Kill / Connect | ks_id, detail |

Retention: online DB continuous; daily export `outputs/paper/audit/YYYYMMDD_events.jsonl`.

---

## 6. Trade database design (SQLite)

**File:** `data/paper/paper_trading.db`

### 6.1 Table: `sessions`

| Column | Type | Notes |
|--------|------|-------|
| session_id | TEXT PK | `YYYYMMDD-NIFTY50-5M` |
| session_date | TEXT | ISO date |
| stack_fingerprint | TEXT | locked fingerprint |
| capital_mode | TEXT | `paper` |
| lots | INTEGER | paper lots (typically 1) |
| signals_buy_v3 | INTEGER | |
| signals_sell_v6 | INTEGER | |
| blocked_sell | INTEGER | |
| trades_opened | INTEGER | |
| trades_closed | INTEGER | |
| same_bar_conflicts | INTEGER | |
| missed_entries | INTEGER | |
| partial_fill_events | INTEGER | |
| pnl_points | REAL | |
| pnl_inr | REAL NULL | null in paper |
| win_rate_pct | REAL | |
| profit_factor_session | REAL NULL | |
| max_adverse_excursion_session_pts | REAL | |
| drawdown_pts | REAL | |
| median_slippage_pts | REAL | |
| median_execution_delay_ms | REAL | |
| capture_efficiency_pct | REAL | |
| throttle_violations | INTEGER | |
| daily_loss_limit_breached | INTEGER | 0/1 |
| kill_switch_fired | TEXT | reason or empty |
| notes | TEXT | |
| created_at | TEXT | |
| closed_at | TEXT NULL | |

### 6.2 Table: `signals`

| Column | Type | Notes |
|--------|------|-------|
| signal_id | TEXT PK | |
| session_id | TEXT FK | |
| ts | TEXT | decision time |
| bar_ts | TEXT | 5m bar close |
| side | TEXT | BUY / SELL |
| signal_version | TEXT | BUY_V3 / SELL_V6 |
| model_id | TEXT | LDM-BUY-V3 / LDM-SELL-V6 |
| entry_intended | REAL | |
| sl | REAL | fixed_10 |
| t1 | REAL | ±60 |
| t2 | REAL | ±100 |
| runner | INTEGER | 1 |
| regime | TEXT | |
| throttle_action | TEXT | FULL/HALF/QUARTER/BLOCK/N/A |
| conflict_flag | INTEGER | |
| status | TEXT | GENERATED/ACCEPTED/REJECTED |
| reject_reason | TEXT NULL | |
| UNIQUE(bar_ts, signal_version) | | idempotency |

### 6.3 Table: `trades`

| Column | Type | Notes |
|--------|------|-------|
| trade_id | TEXT PK | |
| session_id | TEXT FK | |
| signal_id | TEXT FK | |
| ts_open | TEXT | |
| side | TEXT | BUY / SELL |
| signal_version | TEXT | BUY_V3 / SELL_V6 |
| entry | REAL | fill |
| entry_intended | REAL | |
| sl | REAL | |
| t1 | REAL | |
| t2 | REAL | |
| runner | INTEGER | |
| exit | REAL NULL | VWAP of legs or final |
| pnl | REAL NULL | points |
| mfe | REAL | |
| mae | REAL | |
| regime | TEXT | |
| slippage | REAL | entry+exits |
| trade_duration_sec | INTEGER NULL | |
| lifecycle_outcome | TEXT | T1/T2/RUNNER/STOP/FLAT/KILL/… |
| throttle_action | TEXT | |
| conflict_flag | INTEGER | |
| capture_efficiency_pct | REAL NULL | |
| status | TEXT | OPEN/CLOSED |

### 6.4 Table: `trade_legs`

| Column | Type | Notes |
|--------|------|-------|
| leg_id | TEXT PK | |
| trade_id | TEXT FK | |
| leg | TEXT | t1 / t2 / runner / stop |
| intended_qty_pct | REAL | 33.33 |
| fill_price | REAL | |
| fill_qty_pct | REAL | |
| slippage_pts | REAL | |
| fill_ts | TEXT | |

### 6.5 Table: `events`

| Column | Type | Notes |
|--------|------|-------|
| event_id | INTEGER PK AUTOINCREMENT | |
| ts | TEXT | |
| session_id | TEXT | |
| trade_id | TEXT NULL | |
| signal_id | TEXT NULL | |
| event_type | TEXT | see Audit Trail |
| payload_json | TEXT | |

### 6.6 Table: `mfe_mae_samples` (optional high-res)

| Column | Type |
|--------|------|
| trade_id | TEXT |
| ts | TEXT |
| mfe | REAL |
| mae | REAL |
| last_price | REAL |

### 6.7 Indexes

```sql
CREATE INDEX idx_signals_session ON signals(session_id);
CREATE INDEX idx_signals_bar ON signals(bar_ts);
CREATE INDEX idx_trades_session ON trades(session_id);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_open_ts ON trades(ts_open);
CREATE INDEX idx_events_session_ts ON events(session_id, ts);
CREATE INDEX idx_legs_trade ON trade_legs(trade_id);
```

### 6.8 Daily rollup required fields

`session_date`, `pnl_points`, `median_slippage_pts`, `missed_entry_rate_pct`, `same_bar_conflict_count`, `throttle_violations`, `capture_efficiency_pct`, `kill_switch_fired`.

---

## 7. Monitoring dashboard

Proposed: extend `src/dashboard/` → `src/paper/dashboard.py` (read-only on SQLite).  
Views / panels:

| Panel | Content |
|-------|---------|
| Today's Signals | BUY_V3 / SELL_V6 emitted, accepted, rejected + reasons |
| Open Trades | side, entry, SL, T1/T2, MFE, MAE, regime, duration |
| Closed Trades | outcome, PnL, slippage, capture |
| WR | session + rolling 20 |
| PF | session + rolling 20 |
| Expectancy | pts/trade |
| MFE / MAE | distributions, averages |
| Regime Distribution | counts by label + throttle actions |
| Slippage Stats | median, p90, max; kill proximity |
| Target Achievement | T1 when MFE≥60, T2 when MFE≥100 |
| Trade Lifecycle Stats | outcome mix (stop / T1 / T2 / runner / flat) |
| Kill-switch board | armed / tripped / last reason |
| Stack fingerprint | must show locked stack |

Artifacts also land in `outputs/paper/dashboard/` as HTML/JSON snapshots each EOD.

---

## 8. Telegram integration

| Alert | When | Payload (min) |
|-------|------|----------------|
| Signal | Accepted signal | side, engine, entry, SL, T1, T2, regime, throttle |
| Trade Open | Paper position opened | trade_id, fill, slippage |
| Target Hit | T1 or T2 fill | leg, price, remaining size |
| Runner Updates | Trail / partial / mark | mfe, mae, unrealized |
| Trade Close | Flat | pnl, mfe, mae, capture, outcome |
| Kill Switch | Any KS-* | id, trigger, action |
| Connection Failure | WS/API/checksum | KS-12 detail |

Rules:

- Rate-limit identical alerts (e.g. 1/min) except kill-switches (always immediate).
- Prefix every message: `[PAPER ONLY][BUY_V3|SELL_V6]`.
- Never include secrets / tokens.

---

## 9. Paper trading validation framework

### 9.1 Measurement definitions (summary)

| Metric | Pass target |
|--------|-------------|
| Slippage median | ≤ 5.0 pts (kill > 10 rolling-5) |
| Slippage p90 | ≤ 8.0 |
| Decision→submit delay median | ≤ 2000 ms |
| Submit→fill median | ≤ 5000 ms |
| P95 total delay | ≤ 15000 ms |
| Missed entry rate | ≤ 8% (hard kill > 15% / 10 sessions) |
| Conflict compliance | 100% NO_TRADE |
| Conflict rate alert | > 12% |
| Partial fill rate | ≤ 10%; size shortfall ≤ 5% |
| T1 fill when MFE≥60 | ≥ 90% |
| T2 fill when MFE≥100 | ≥ 85% |
| Stop fill when MAE≥10 | ≥ 95%; median stop slip ≤ 3; false stop ≤ 5% |
| Capture efficiency | 34–44% band (P20 tighter 34–41% vs ~37.66% replay) |

### 9.2 Track — 20 sessions (paper gate)

**Purpose:** Prove execution telemetry + playbook compliance before any real capital.

Per-session gates (every day):

- Stack fingerprint = LOCKED_STACK (no V4/V7)
- Session log schema complete
- BLOCK regime SELLs → zero fills
- Conflicts → NO_TRADE zero fills
- Median slippage recorded
- Daily loss ≤ 593.79 pts
- Kill-switch status reviewed

**End-of-phase gates (P20):**

| ID | Gate |
|----|------|
| P20-1 | ≥18/20 sessions non-negative combined PnL (or ≤2 documented anomalies) |
| P20-2 | Median slippage ≤ 5.0 |
| P20-3 | Throttle BLOCK violations == 0 |
| P20-4 | Conflict compliance == 100% |
| P20-5 | Daily loss breaches == 0 |
| P20-6 | Capture 34.0–41.0% |
| P20-7 | Sample: BUY ≥15, SELL ≥25 |
| P20-8 | Combined PF ≥ 1.5 AND WR ≥ 55% |
| P20-9 | No integrity kill-switch |
| P20-10 | No V4/V7 / stack drift |

**Gate to ₹50K:** all P20-* + written risk sign-off.  
**Current:** paper sessions completed = **0** → promotion **NO**.

### 9.3 Track — 40 sessions

Sample floors: BUY≥30, SELL≥50, combined≥80.  
PF ≥ 1.8, WR ≥ 58%, expectancy ≥ 40 pts.  
Rolling SELL PF≥1.5 at checkpoints; slippage/conflict/throttle/capture gates hold; recovery ≤10 trading days; independent audit sign-off.

### 9.4 Track — 60 sessions

Required for ₹1L → ₹2L path: PF≥2.0, WR≥60%, expectancy≥50, DD caps, slippage median≤5 / p90≤8, independent audit, confidence proxy ≥75%.

### 9.5 Metrics to measure every session / rollup

WR, PF, Expectancy, Capture %, Avg RR, Max DD, Recovery Factor, Regime Performance, Slippage (median/p90).

---

## 10. Promotion framework

Aligns with `promotion_criteria` in live deployment validation JSON.  
**Live capital is out of scope for this paper system** — criteria documented so paper evidence maps cleanly later.

### 10.1 Paper → ₹50K

| Criterion | Value |
|-----------|------:|
| Min sessions | 20 |
| Min PF | 1.5 |
| Min WR | 55% |
| Min expectancy | 30 pts |
| Max DD | 800 pts / 8% capital |
| Max median slippage | 5.0 |
| Max consecutive losses | 6 (at tier) |
| Max daily loss | 593.79 pts |
| Lots max | 1 |
| Verdict now | **NO** |

Evidence: P20 checklist, fill CSV + signal journal join, throttle 100%, conflict log, slippage dist, no stack drift, risk sign-off.

### 10.2 ₹50K → ₹1L

| Criterion | Value |
|-----------|------:|
| Min sessions | 40 (incl. ≥20 at 50K) |
| Min PF | 1.8 |
| Min WR | 58% |
| Min expectancy | 40 |
| Max DD | 1200 pts / 8% |
| BUY WR on ≥30 trades | ≥ 48% |
| SELL rolling-20 PF | ≥ 1.5 |
| Verdict now | **NO** |

### 10.3 ₹1L → ₹2L

| Criterion | Value |
|-----------|------:|
| Min sessions | 60 (incl. ≥20 at 1L) |
| Min PF | 2.0 |
| Min WR | 60% |
| Min expectancy | 50 |
| Max DD | 1500 pts / 8% |
| Slippage | median≤5, p90≤8 |
| Lots max | 2 |
| Verdict now | **NO** |

**Note:** Engine research PRODUCTION_GATES (WR≥65, PF≥2) remain research gates; live promotions use the **conservative** floors above until live samples justify otherwise.

---

## 11. Deployment architecture

### 11.1 Recommended project architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     PAPER ONLY RUNTIME                          │
│  enable_live_orders = false · capital_mode = paper              │
└─────────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌──────────────────────────┐
│ FYERS Auth/Client   │              │ FYERS WebSocket (ticks)  │
│ src/brokers/fyers/* │              │ src/paper/fyers_ws.py    │
└─────────┬───────────┘              └────────────┬─────────────┘
          │ REST history                          │
          ▼                                       ▼
┌───────────────────────────────────────────────────────────────┐
│              5m Candle Builder · Session Manager                │
│         src/paper/candle_builder.py · session_manager.py        │
└─────────────────────────────┬─────────────────────────────────┘
                              │ closed bars
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   BUY_V3 live         SELL_V6 live        Regime Throttle
   (LDM-BUY-V3)        (LDM-SELL-V6)       FULL/HALF/QUARTER/BLOCK
          └───────────────────┬───────────────────┘
                              ▼
                    Conflict Router (NO_TRADE)
                              ▼
                    Risk / Kill-Switch Layer
                              ▼
              Paper Broker (simulate fills only)
                              ▼
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
     SQLite DB          Telegram Alerts      Dashboard (RO)
  data/paper/*.db     telegram_notifier     src/paper/dashboard
```

### 11.2 Folder structure (existing + proposed)

```text
SmartMoneyEngine/
├── paper_trading_framework.md          ← this document
├── .env                                ← FYERS_* , TELEGRAM_* (secrets)
├── config/paper/                       ← proposed
│   ├── paper_trading.yaml
│   ├── nse_holidays.json
│   └── regime_throttle_map.json        ← snapshot from research export
├── data/
│   ├── tokens/fyers_token.json         ← existing
│   ├── historical/                     ← existing FYERS history
│   └── paper/
│       ├── paper_trading.db
│       └── backups/
├── src/
│   ├── brokers/fyers/                  ← existing (auth, client, historical, config)
│   ├── research/                       ← BUY_V3 / SELL_V6 / regime throttle sources
│   ├── signals/                        ← lifecycle / decision helpers
│   ├── dashboard/                      ← existing stub
│   └── paper/                          ← proposed paper runtime package
│       ├── __init__.py
│       ├── main.py                     ← entry: paper session runner
│       ├── candle_builder.py
│       ├── fyers_ws.py
│       ├── session_manager.py
│       ├── engines/buy_v3_live.py
│       ├── engines/sell_v6_live.py
│       ├── regime_throttle.py
│       ├── conflict_router.py
│       ├── risk_controls.py
│       ├── paper_broker.py             ← NO live orders
│       ├── trade_lifecycle.py
│       ├── db.py
│       ├── telegram_notifier.py
│       ├── dashboard.py
│       └── recovery.py
├── outputs/
│   ├── research/live_deployment_validation_framework.json
│   └── paper/
│       ├── journals/
│       ├── sessions/
│       ├── audit/
│       └── dashboard/
├── logs/
│   └── engine.log
└── tests/
    └── test_paper_*                    ← proposed unit tests for router/risk/db
```

### 11.3 Required files (minimum to run paper later)

| File | Role |
|------|------|
| `config/paper/paper_trading.yaml` | Stack lock, session times, KS thresholds, cooldowns |
| `config/paper/regime_throttle_map.json` | Audited throttle map |
| `config/paper/nse_holidays.json` | Holiday calendar |
| `src/paper/main.py` | Orchestrator |
| `data/tokens/fyers_token.json` | FYERS access |
| `data/paper/paper_trading.db` | Created on first run |
| `.env` | FYERS + Telegram |

### 11.4 Config file sketch (`config/paper/paper_trading.yaml`)

```yaml
mode: paper
enable_live_orders: false
symbol: NIFTY50
timeframe: 5M
stack:
  buy: BUY_V3
  sell: SELL_V6
  stop: fixed_10
  targets: [60, 100, runner]
  regime_throttle: true
  fingerprint: "BUY_V3|SELL_V6|fixed_10|60/100/Runner|RegimeThrottle"
session:
  timezone: Asia/Kolkata
  open: "09:15"
  close: "15:30"
  last_entry: "15:15"
risk:
  daily_loss_limit_pts: 593.79
  consecutive_loss_limit: 7
  slippage_median_pass: 5.0
  slippage_p90_pass: 8.0
  slippage_kill: 10.0
  intraday_dd_pts: 400
  paper_dd_soft: 800
  paper_dd_hard: 1200
conflict_policy: NO_TRADE
database: data/paper/paper_trading.db
```

### 11.5 FYERS design

| Layer | Module |
|-------|--------|
| Config | `src/brokers/fyers/config.py` |
| OAuth | `src/brokers/fyers/auth.py` → `data/tokens/fyers_token.json` |
| REST | `src/brokers/fyers/client.py` (`FyersClient`) |
| History | `src/brokers/fyers/historical.py` |
| Live WS | proposed `src/paper/fyers_ws.py` (data only) |
| Paper broker | proposed `src/paper/paper_broker.py` — simulates fills from bar OHLC; **must not call place_order** |

### 11.6 Telegram design

- Bot token + chat id from env.
- Template renderer with `[PAPER ONLY]` prefix.
- Severity: INFO (signal), WARN (slippage drift), CRITICAL (kill / disconnect).

### 11.7 Deployment steps (paper)

1. Confirm git workspace; **do not enable live orders**.
2. Populate `.env` (FYERS + Telegram).
3. Run FYERS auth → token file present.
4. Copy throttle map from `outputs/research/regime_detection_audit.json` into `config/paper/regime_throttle_map.json`.
5. Create `config/paper/paper_trading.yaml` with locked stack + KS thresholds.
6. Initialize SQLite schema (`src/paper/db.py`).
7. Dry-run: connect WS in market hours with engines in **shadow** (log signals, no paper fills) for 1 session.
8. Enable paper fills; arm kill-switches; Telegram test message.
9. Operate for 20 sessions under §9.2 checklist.
10. Archive artifacts under `outputs/paper/sessions/`.

### 11.8 Runbook (daily)

| When | Action |
|------|--------|
| Pre-market | Holiday check; token; DB backup; Telegram ping; fingerprint assert |
| Open | Start `python -m src.paper.main` (or equivalent) |
| Intraday | Watch dashboard + Telegram; do not override BLOCK/conflict |
| On alert | Follow KS action table; no discretionary re-entry after KS-12 |
| Close | Confirm flat/session policy; review rollup; commit artifacts (optional ops copy — not git unless asked) |
| Weekly | Review rolling PF/WR/slippage vs P20/P40 gates |

### 11.9 Monitoring architecture

- Primary: SQLite + dashboard RO queries.
- Secondary: Telegram CRITICAL channel.
- Tertiary: `logs/engine.log` + paper journals.
- EOD JSON snapshot for promotion scorecard.

### 11.10 Recovery procedures

| Scenario | Procedure |
|----------|-----------|
| WS drop | §3.5–3.6; missed candle recovery; KS-12 if uncertain |
| Process crash | Restart → load OPEN trades → reconcile bars → resume or flat |
| Bad bar / checksum | Halt; restore from FYERS history; mark event; no backfilled fills without idempotent IDs |
| Accidental stack drift | KS-10: halt; reset yaml to LOCKED_STACK; invalidate session for promotion |
| Kill-switch trip | Flat; document; operator reset only after checklist |

### 11.11 Promotion procedures

1. Export 20/40/60 session scorecard from DB.
2. Join signal journal + paper fills.
3. Verify P20/P40 gates against §9–§10.
4. **Do not** enable INR tiers from this paper runtime until written sign-off.
5. Current capital readiness: **₹50K / ₹1L / ₹2L = NO**.

---

## 12. Final answer — checklists

### 12.1 Recommended architecture

**Paper-only FYERS data path** → 5m candle builder → **BUY_V3 + SELL_V6** → **Regime Throttle** → **NO_TRADE conflict router** → kill-switch risk layer → **paper broker (no live orders)** → SQLite + Telegram + dashboard.  
Reuse existing `src/brokers/fyers/*` and research engine IDs `LDM-BUY-V3` / `LDM-SELL-V6`. Do not introduce BUY_V4 / SELL_V7.

### 12.2 Deployment checklist

- [ ] `enable_live_orders=false` locked in config
- [ ] FYERS `.env` + `data/tokens/fyers_token.json` valid
- [ ] Stack fingerprint matches LOCKED_STACK
- [ ] Regime throttle map loaded from audit export
- [ ] SQLite schema created at `data/paper/paper_trading.db`
- [ ] Telegram bot test message received
- [ ] Holiday calendar present
- [ ] Kill-switches KS-1…KS-12 coded/armed
- [ ] Shadow session (optional) then paper fills enabled
- [ ] Backup path `data/paper/backups/` writable

### 12.3 Paper trading checklist

- [ ] Each session writes full schema (sessions/signals/trades/events)
- [ ] BLOCK SELLs → zero fills
- [ ] Conflicts → NO_TRADE
- [ ] MFE/MAE updated while open
- [ ] Slippage logged entry+legs
- [ ] Daily loss ≤ 593.79 pts
- [ ] Journals under `outputs/paper/`
- [ ] Track toward **20** sessions before any capital discussion

### 12.4 Operational checklist

- [ ] Pre-market token + health
- [ ] Open/close per IST schedule
- [ ] Reconnect + missed candle recovery tested
- [ ] EOD rollup + Telegram summary
- [ ] Weekly rolling PF/WR/slippage review
- [ ] No manual “force fill” on BLOCK/conflict bars

### 12.5 Risk checklist

- [ ] Daily loss KS-1 armed (593.79)
- [ ] Slippage KS-2 armed (10 pt kill)
- [ ] BLOCK fill KS-3 zero tolerance
- [ ] Conflict fill KS-4 zero tolerance
- [ ] Consecutive loss KS-5 (7 paper)
- [ ] Feed failure KS-12
- [ ] Stack drift KS-10
- [ ] Circuit breaker on repeated disconnects
- [ ] Paper DD soft/hard 800 / 1200

### 12.6 Go-live checklist (paper go-live — not capital)

Paper “go-live” means **first supervised paper session**, not INR deployment.

- [ ] All deployment + risk checklists green
- [ ] Operator present first 3 sessions
- [ ] Kill-switch drill documented (simulated daily-loss halt)
- [ ] Confirm **no** FYERS place-order calls in logs
- [ ] Promotion scorecard initialized at session 0
- [ ] Explicit acknowledgment: real capital **NO** until P20/P40 pass

---

## 13. One-line operating policy

**PAPER ONLY:** run **BUY_V3 + SELL_V6 + fixed_10 + 60/100/Runner + Regime Throttle** on FYERS 5m data with full telemetry and kill-switches; **never** send live orders or promote V4/V7; real capital remains **NO** until validation gates pass.

---

*Evidence anchors: `outputs/research/live_deployment_validation_framework.json`, `src/brokers/fyers/`, `src/research/buy_v3_candidate_validation_research.py`, `src/research/regime_detection_audit_research.py`, `src/research/sell_v6_replay_validation_research.py`.*
