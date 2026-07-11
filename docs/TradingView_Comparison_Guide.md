# TradingView Comparison Guide

**Project:** SmartMoneyEngine  
**Scope:** Visual and numerical comparison of engine output against TradingView ICT/SMC indicators  
**Audience:** Quant validation engineers

---

## Purpose

TradingView is the **reference visual environment** for validating SmartMoneyEngine output. This guide defines how to set up charts, which indicators to use, how to compare each module, and how to triage discrepancies.

Engine validation is **not** a pixel-perfect match exercise. The goal is to confirm that institutional logic is directionally correct, causally sound, and within documented tolerance bands.

---

## Golden Rules

1. **Same symbol, same timeframe, same session** — mismatched settings cause false failures.
2. **Compare definitions, not colours** — verify rules (close vs wick, lookback, tolerance).
3. **Log every discrepancy** with a reason code (see § Discrepancy Triage).
4. **Validate swings first** — all downstream modules inherit swing errors.
5. **Batch vs live** — engine runs batch-confirmed swings; TV pivots may update in real time. Compare on **closed historical bars only**.

---

## Reference Chart Setup

### Step 1 — Base Chart

| Setting | Requirement |
|---------|-------------|
| Symbol | Same as CSV source (e.g. `NSE:NIFTY`, `BINANCE:BTCUSDT`) |
| Timeframe | Exact match to CSV (1H, 4H, 1D, etc.) |
| Session | Match exchange session used in CSV data |
| Chart type | Candlestick |
| Price source | OHLC (not Heikin Ashi, not Renko) |

### Step 2 — Date Alignment

1. Open CSV `Date` column and TV chart crosshair on same bar.
2. Verify `Open`, `High`, `Low`, `Close` match TV within tick tolerance.
3. If OHLC mismatch → **stop** and fix data feed before SMC comparison.

### Step 3 — Export Engine Output

```powershell
python -m src.validation.engine_validation --csv tests/sample_data/swing_test.csv
```

Open `validation_output.csv` alongside TradingView.

### Step 4 — Recommended TradingView Indicators

| Module | TV Indicator Options | Notes |
|--------|---------------------|-------|
| Swings | Pivot Points High/Low, LuxAlgo Swing Highs/Lows | Set left/right = engine `lookback` (default 2) |
| Structure | ICT Market Structure (community), Smart Money Concepts | Match HH/HL/LH/LL definitions |
| Trend | Manual from structure shading | Few native "Trend" columns |
| BOS/CHOCH | ICT Structure / BOS & CHOCH indicators | Check close vs wick setting |
| FVG | Fair Value Gap, ICT FVG | Default three-candle wick gap |
| Order Block | ICT Order Block, OB Finder | Expect count differences due to filters |
| Liquidity | Equal Highs/Lows, Liquidity levels | Manual equality vs 0.10% tolerance |

> Community ICT scripts vary widely. Document the **exact script name and settings** used for each validation run.

---

## Comparison Workflow

```
1. Align OHLC        →  must pass before continuing
2. Compare Swings    →  foundation for all structure
3. Compare Structure →  HH/HL/LH/LL at swing points
4. Compare Trend     →  direction on sample segments
5. Compare BOS/CHOCH →  event bar dates and prices
6. Compare FVG       →  zone boundaries
7. Compare OB        →  zone + mitigation bar
8. Compare Liquidity →  pool level + sweep bar
9. Log discrepancies →  reason codes
10. Sign off         →  SMC_Validation_Checklist.md
```

---

## Module Comparison Procedures

### SwingDetector

**Engine rule:** Bar `i` is swing high if `High[i]` strictly exceeds `lookback` highs on each side (default lookback = 2).

**TradingView setup:**

1. Add **Pivot Points High/Low** (built-in) or equivalent.
2. Set **Left Bars** = 2, **Right Bars** = 2 (match engine lookback).
3. Disable dynamic/streaming pivot updates for historical review.

**Comparison procedure:**

1. List all engine rows where `Swing_High` is not null.
2. For each, locate the same datetime on TV.
3. Confirm TV shows a pivot high at that bar.
4. Repeat for `Swing_Low`.

**Tolerance:** Price must match exactly (same OHLC source). Date must match exactly.

**Acceptance:** ≥ 80% pivot agreement on reference dataset.

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV pivot, engine NaN | TV uses `>=` vs engine strict `>` |
| Engine pivot, TV none | Different lookback; TV indicator still updating |
| Offset by 1 bar | Timezone / session boundary mismatch |
| Price matches, date differs | CSV date is bar open; TV is bar close |

---

### MarketStructure

**Engine rule:** HH/LH at consecutive swing highs; HL/LL at consecutive swing lows.

**TradingView setup:**

1. Use ICT structure indicator that labels swings (not every candle).
2. Alternatively, **manual markup** on agreed swing points from previous step.

**Comparison procedure:**

1. Only compare at rows where both engine and TV agree on swing location.
2. For each pair of consecutive engine swing highs:
   - If second > first → TV should show HH (or bullish structure continuation).
   - If second < first → TV should show LH.
3. Same for swing lows → HL / LL.

**Acceptance:** 100% agreement at agreed swing indices (structure is deterministic from swings).

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| Label on first swing | TV labels initial swing; engine skips |
| Label on equal swing | TV forces LH; engine leaves NaN |
| Structure on break | TV uses close break; engine uses swing point |

---

### TrendEngine

**Engine rule:** BULLISH = HH+HL confirmed; BEARISH = LH+LL; SIDEWAYS = mixed.

**TradingView setup:**

- Most TV scripts show trend as background colour or label — no standard column export.
- **Manual segment review** recommended.

**Comparison procedure:**

1. Select 3 visually bullish segments on TV (clear HH/HL staircase).
2. Verify engine `Trend == BULLISH` over same date range (after confirmation bar).
3. Select 3 bearish segments → `BEARISH`.
4. Select 2 choppy ranges → `SIDEWAYS`.

**Acceptance:** Direction match on all manually selected segments.

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV bullish, engine sideways | Mixed HL/LH evidence in engine |
| Flip timing differs | TV flips on single counter-label |
| Strength differs | Engine-specific 0–3 score |

---

### BreakOfStructure (BOS)

**Engine rule:**

- Bullish BOS: `Close` breaks above last HH; prior close was at/below HH.
- Bearish BOS: `Close` breaks below last LL; prior close was at/above LL.
- No same-bar repaint with new HH/LL.

**TradingView setup:**

1. ICT indicator with **BOS** enabled.
2. Confirm indicator setting: **close break** (not wick).
3. Draw horizontal line at last HH manually as independent check.

**Comparison procedure:**

1. For each engine `Bullish_BOS` row, mark TV bar where close first exceeds last HH.
2. Compare dates — should match exactly if same close rule.
3. Verify BOS price = close on both engine and TV.
4. Confirm no duplicate BOS on subsequent bars staying above level.

**Acceptance:** Event bar date match ≥ 90%; price = close on match bars.

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV BOS earlier | TV uses wick (High) break |
| Engine BOS, TV none | HH not yet confirmed in engine swing chain |
| Different HH reference | TV uses most recent swing high vs structure HH |

---

### ChangeOfCharacter (CHOCH)

**Engine rule:**

- Bullish CHOCH: bearish bias + close above last LH.
- Bearish CHOCH: bullish bias + close below last HL.

**TradingView setup:**

- ICT CHOCH indicator or manual LH/HL break identification.

**Comparison procedure:**

1. Identify last LH before bullish reversal on TV.
2. Find first close above that LH → compare to `Bullish_CHOCH` row.
3. Verify engine did not fire on LH formation bar (no-repaint).
4. Repeat for bearish CHOCH with HL.

**Acceptance:** Event bar match ≥ 85% (CHOCH definitions vary most across TV scripts).

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV shows CHOCH, engine BOS | Different continuation vs reversal classification |
| Timing offset | Bias definition differs |
| CHOCH on wick | TV wick break vs engine close |

---

### FairValueGap (FVG)

**Engine rule:**

- Bullish: `Low[candle3] > High[candle1]` → gap between `High[candle1]` and `Low[candle3]`.
- Bearish: `High[candle3] < Low[candle1]`.
- Label on third candle.

**TradingView setup:**

1. Add standard **Fair Value Gap** indicator.
2. Disable minimum gap size filter if configurable (engine has no min size).

**Comparison procedure:**

1. For each engine FVG row, locate third candle on TV.
2. Compare top/bottom values within tick tolerance.
3. Verify TV gap exists at same three-candle sequence.

**Acceptance:** ≥ 90% boundary overlap on matched gaps.

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV fewer gaps | TV min gap size / ATR filter |
| Boundary offset | TV uses body gap variant |
| Anchor candle differs | TV marks on candle 2 vs engine candle 3 |

---

### OrderBlockDetector

**Engine rule:**

- Bullish OB = last bearish candle before impulsive bullish move to BOS.
- Zone = origin candle high/low.
- Mitigation when price trades through zone after BOS.

**TradingView setup:**

- ICT Order Block indicator (note displacement filter settings).

**Comparison procedure:**

1. For each engine OB, locate origin candle date on TV.
2. Verify candle polarity (bearish for bullish OB).
3. Compare zone boundaries (high/low).
4. Confirm BOS occurs after OB candle.
5. Find first bar where low < OB low (bullish) → compare mitigation.

**Acceptance:** Zone overlap ≥ 70% on matched OBs; polarity 100% correct.

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV more OBs | TV less strict displacement filter |
| Engine OB, TV none | Body ratio / equal-level rejection |
| Zone width differs | TV uses last candle in series vs single origin |
| Mitigation timing | TV uses touch of mean threshold vs full zone |

---

### LiquidityDetector

**Engine rule:**

- EQH/EQL from swings within 0.10% tolerance.
- Pool active from second touch.
- Buy sweep: high above pool, close below. Sell sweep: low below, close above.

**TradingView setup:**

- Manual horizontal lines at equal highs/lows from **swing points**.
- Optional: liquidity sweep indicator.

**Comparison procedure:**

1. Identify engine equal-high clusters → mark swings on TV.
2. Verify swings visually equal within tolerance.
3. Draw pool level at max(highs) for buy-side.
4. Pool active from second touch bar forward.
5. Find sweep bar: wick through, close reject → compare to engine sweep column.

**Acceptance:** Pool level within 0.10%; sweep bar match ≥ 80%.

**Common TV differences:**

| Difference | Reason Code |
|------------|-------------|
| TV pool, engine none | Swings too far apart for tolerance |
| Engine pool, TV none | TV uses raw OHLC equality not swings |
| Sweep timing differs | TV uses wick only without close reject |
| Pool persists after sweep | Engine v1 forward-fill (document, not TV bug) |

---

## Side-by-Side Comparison Template

Use this table for each validation session:

| Date | Module | Engine Value | TV Value | Match? | Reason Code |
|------|--------|--------------|----------|--------|-------------|
| 2024-01-15 | Swing_High | 4520.5 | 4520.5 | ✓ | — |
| 2024-01-18 | Bullish_BOS | 4550.0 | 4550.0 | ✓ | — |
| 2024-01-20 | Bullish_OB_Low | 4480.0 | 4475.0 | ✗ | OB-FILTER |

---

## Discrepancy Triage

### Reason Codes

| Code | Meaning | Action |
|------|---------|--------|
| `DATA-MIS` | OHLC mismatch between CSV and TV | Fix data source |
| `TZ-OFF` | Date offset (timezone/session) | Normalize timestamps |
| `LB-DIFF` | Lookback / pivot length mismatch | Align lookback settings |
| `CLOSE-WICK` | Close break vs wick break | Document expected delta |
| `TOL-LIQ` | Liquidity tolerance difference | Adjust tolerance or accept |
| `FVG-FILTER` | TV gap size filter | Document TV settings |
| `OB-FILTER` | OB displacement/body filter | Expected engine strictness |
| `SWING-BATCH` | Batch-confirmed vs live pivot | Compare historical bars only |
| `TV-SCRIPT` | Community script non-standard logic | Change TV script or document |
| `ENG-CAVEAT` | Known engine v1 limitation | Track in checklist caveats |

### Severity Classification

| Severity | Description | Release Impact |
|----------|-------------|----------------|
| **Critical** | Wrong event type (BOS labelled as CHOCH), repaint detected | Block release |
| **Major** | Event missed or >2 bar offset on clear pattern | Investigate before release |
| **Minor** | ≤1 bar offset, tolerance boundary, count difference with filters | Document and track |
| **Cosmetic** | Label placement, colour, non-price metadata | No impact |

---

## Recommended Reference Datasets

| Dataset | Path | Purpose |
|---------|------|---------|
| Swing test | `tests/sample_data/swing_test.csv` | Integration smoke test |
| Nifty sample | `data/sample/nifty_sample.csv` | Real instrument validation |
| Trending segment | Custom export (20+ bars, clear trend) | BOS/OB validation |
| Ranging segment | Custom export (mixed structure) | CHOCH/SIDEWAYS validation |
| Equal highs | Custom export (double top pattern) | Liquidity validation |

---

## Validation Session Checklist

| # | Step | Done |
|---|------|------|
| 1 | Chart symbol/timeframe matches CSV | ☐ |
| 2 | OHLC spot-check (≥ 5 random bars) | ☐ |
| 3 | TV indicator names and settings recorded | ☐ |
| 4 | Engine output CSV loaded | ☐ |
| 5 | Swings compared | ☐ |
| 6 | Structure compared at agreed swings | ☐ |
| 7 | BOS/CHOCH events compared | ☐ |
| 8 | FVG boundaries compared | ☐ |
| 9 | Order blocks compared (if present) | ☐ |
| 10 | Liquidity pools compared (if present) | ☐ |
| 11 | Discrepancy log completed | ☐ |
| 12 | Severity assigned to each discrepancy | ☐ |
| 13 | Sign-off in SMC_Validation_Checklist.md | ☐ |

---

## What TradingView Cannot Validate

These engine properties require **code review or automated tests**, not chart comparison:

- No same-bar repaint (BOS/CHOCH causal ordering)
- No duplicate BOS/CHOCH events
- Column preservation through pipeline
- No duplicate column names
- Order block position < BOS position (structural invariant)
- Input column contract enforcement

Use `engine_validation.py` and unit tests for these guarantees.

---

## Related Documents

- `SMC_Validation_Checklist.md` — master acceptance checklist
- `Module_Validation_Guide.md` — detailed per-module validation steps
