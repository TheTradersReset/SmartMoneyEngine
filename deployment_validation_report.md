# Live Paper Deployment Validation Report

- **Date/time IST:** 2026-07-22T12:29:48.686059+05:30
- **Environment summary:** .env=yes; market_session_open=True; history_csv_exists=True; token_exists=True; credentials={'FYERS_APP_ID': 'SET', 'FYERS_SECRET_KEY': 'SET', 'FYERS_REDIRECT_URI': 'SET', 'TELEGRAM_BOT_TOKEN': 'MISSING', 'TELEGRAM_CHAT_ID': 'MISSING'}
- **Overall verdict:** CONDITIONAL
- **Go / No-Go:** **NO-GO**
- **Go/No-Go reason:** Telegram credentials missing: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env

## Executive summary
- Token: VALID (exp IST 2026-07-23T06:00:00+05:30); refresh present; ensure_ok
- WS smoke (~60s): tick_count=50 reconnect_attempts=5
- Live paper smoke (~10-12 min): max_tick=#1458; candle_closed=1015; telegram_hits=0; latency_avg_ms=639.0
- Telegram: **FAIL** — missing .env keys: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- Candle builder pytest: PASS (2 passed)

## Step 0 — Environment: PASS
- .env exists: True
- Credential presence (no values): FYERS_APP_ID=SET, FYERS_SECRET_KEY=SET, FYERS_REDIRECT_URI=SET, TELEGRAM_BOT_TOKEN=MISSING, TELEGRAM_CHAT_ID=MISSING
- Optional: FYERS_PIN=MISSING
- LIVE_PAPER_* env keys: ['LIVE_PAPER_CAPITAL_MODE', 'LIVE_PAPER_DASHBOARD_HOST', 'LIVE_PAPER_DASHBOARD_PORT', 'LIVE_PAPER_ENABLE_DASHBOARD', 'LIVE_PAPER_ENABLE_TELEGRAM']
- live_paper.yaml exists: True
- fyers_token.json: {"exists": true, "has_access_token_field": true, "jwt": {"decoded": true, "exp_epoch": 1784766600, "exp_ist": "2026-07-23T06:00:00+05:30", "expired": false, "seconds_to_exp": 63011}}
- history CSV path=C:\Users\bhargupt\OneDrive - Nokia\Mindtree_Data\Docs\Project\DOCS\MLOps\TheTradersReset\SmartMoneyEngine\outputs\pipeline\NIFTY50_5m_pipeline.csv exists=True
- IST now: 2026-07-22T12:29:48.686059+05:30 weekday=Wednesday
- Market session 09:15-15:30 IST weekday: True

## Step 1 — FYERS WebSocket: PASS
- code: use_sdk_reconnect default True present=True
- code: outer reconnect loop + backoff present=True
- ensure_valid_access_token(allow_interactive_oauth=False): OK
- REST fyers_client exit=0 stdout_lines=5
- WS probe ticks=20 heartbeat=True last_tick_at_set=True

## Step 2 — Candle Builder: PASS
- _floor_to_bar OK -> 2026-07-22T09:15:00+05:30
- Synthetic two-bucket: 1 close on roll + 1 flush OK
- Same-bucket ticks -> one candle on flush OK
- Session alignment evidence: {'SESSION_OPEN': False, 'SESSION_CLOSE': False, 'BAR_MINUTES_5': True}
- MissedCandleRecovery exists; detect_missed_candles=4
- pytest test_candle_builder exit=0 tail=..                                                                       [100%]
2 passed in 0.06s

## Step 3 — Strategy Execution: PARTIAL
- LivePaperPipeline subclass + super call=True
- Engines buy=BuyV3Engine sell=SellV6Engine ok=True
- warm_start_from_frame bars=200
- Synthetic candle wall_ms=101.2 latencies=1
- mark_emitted twice same bar size=1 contains=True
- Latency stats: {'n': 1, 'avg': 101.2339000008069, 'max': 101.2339000008069, 'p95': 101.2339000008069}
- **Reason:** Only 0-1 latency samples ? p95 not statistically meaningful
- **Recommended fix:** Collect multi-candle latency during live smoke for meaningful p95.

## Step 4 — Telegram: FAIL
- TELEGRAM_BOT_TOKEN=MISSING
- TELEGRAM_CHAT_ID=MISSING
- format_signal_alert dry-run OK=True
- TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing ? config incomplete
- Real signal delivery deferred ? cannot send without credentials
- **Reason:** TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing ? config incomplete
- **Recommended fix:** Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env; re-run python -m src.live_paper.telegram_test.

## Step 5 — SQLite: PASS
- UNIQUE(symbol,timestamp) after 2 inserts count=1 (expect 1)
- insert_signal id=1 update_outcome rows=1 outcome=WIN
- Concurrent writers errors=0 detail=none
- Query counts: {'candles': 42, 'signals': 1}
- journal_mode=delete

## Step 6 — Dashboard: PASS
- GET / status=200
- GET /api/status status=200 keys={'market': True, 'ws': True, 'heartbeat': True, 'signals': True, 'trades': True, 'latency': True, 'cpu': True, 'mem': True, 'errors': True}
- HTML setInterval auto-refresh=True

## Step 7 — Failure Tests: PASS
- WS reconnect: backoff=[1.0, 2.0, 4.0] reset/stop=True outer_loop=True
- request_stop on from_env client: OK
- Telegram network fail -> False no crash: True
- Telegram disabled/missing -> False no exception: True
- SQLite exclusive lock observed: OperationalError: database is locked
- AsyncDbWriter enqueue/flush/close OK
- Dashboard start/stop/start on :18080 -> [True, True]

## Step 8 — Live Paper Session: PARTIAL
- market_open=True token_ok=True fyers_creds=True
- Starting smoke live session for 180s (NOT a full trading day).
- Smoke session returncode=1 log_bytes=153224 candle~2 signal~7 errors~0
- Honest scope: smoke session only ? NOT a complete trading session.
- **Reason:** Smoke session only. Full-day validation deferred.
- **Recommended fix:** Run full session 09:15-15:30 IST after Telegram is configured; capture logs/live_paper/.

## External live smoke (manual ~10-12 min)
- Log: outputs/deployment_validation/live_smoke.log (encoding=utf-16, bytes=23349092)
- max_tick_number: 1458
- candle_closed_count: 1015 (includes history warm-start closes)
- telegram_keyword_hits: 0
- latency_avg_ms: 639.0 (n=2030); latency_max_ms: 4719.09
- error_lines_approx: 0

## Metrics collected
- WS probe ticks: 20
- WS smoke ticks: 50 (reconnect_attempts=5)
- Strategy latency stats: {'n': 1, 'avg': 101.2339000008069, 'max': 101.2339000008069, 'p95': 101.2339000008069}
- Telegram TEST latency_ms: None
- Validation step8 duration_sec: 180 ran=True max_tick=274

## Risks / blockers
- **BLOCKER:** Missing .env keys: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- Optional: FYERS_PIN MISSING (refresh path may need it later)
- Full-day live session not completed (smoke only)
- Strategy latency p95 not statistically meaningful (n=1 synthetic)

## Sign-off checklist
- [x] FYERS credentials + valid token + WS ticks
- [x] Candle builder tests green
- [x] Strategy engines BuyV3/SellV6 wired
- [ ] Telegram configured + TEST ok
- [x] SQLite insert/outcome/unique
- [x] Dashboard / and /api/status
- [x] Failure handling simulations
- [ ] Live session evidence (full day)

_Evidence: outputs/deployment_validation/evidence.json; WS smoke: outputs/deployment_validation/ws_smoke.py; Live smoke: outputs/deployment_validation/live_smoke.log_
