# SMC Validation Checklist

**Project:** SmartMoneyEngine  
**Scope:** End-to-end Smart Money Concepts (SMC/ICT) foundation pipeline  
**Version:** Foundation v1  
**Audience:** Quant validation engineers, QA, release reviewers

---

## Purpose

This document is the **master gate checklist** for validating the complete SmartMoneyEngine SMC pipeline before any downstream strategy, signal, or live-trading work is approved.

Use it together with:

- `Module_Validation_Guide.md` — detailed per-module procedures
- `TradingView_Comparison_Guide.md` — visual/chart comparison methodology

---

## Pipeline Under Validation

```
CSV (MarketData)
    ↓
SwingDetector
    ↓
MarketStructure
    ↓
TrendEngine
    ↓
BreakOfStructure
    ↓
ChangeOfCharacter
    ↓
FairValueGap
    ↓
OrderBlockDetector
    ↓
LiquidityDetector
    ↓
validation_output.csv
```

**Canonical runner:** `python -m src.validation.engine_validation`

---

## Pre-Flight Checks (Before Pipeline)

| # | Check | Pass |
|---|-------|------|
| 1 | CSV contains `Date`, `Open`, `High`, `Low`, `Close`, `Volume` | ☐ |
| 2 | No missing OHLCV values in source CSV | ☐ |
| 3 | Dates sorted ascending, no duplicates | ☐ |
| 4 | OHLC integrity: `High >= Open/Close`, `Low <= Open/Close`, `High >= Low` | ☐ |
| 5 | Volume > 0 on all rows | ☐ |
| 6 | Minimum row count ≥ 5 (swing lookback=2 needs ≥ 5 rows) | ☐ |
| 7 | Timeframe documented (e.g. 1H, 4H, D) for TradingView comparison | ☐ |
| 8 | No duplicate column names in source frame | ☐ |

---

## Stage Gate Checklist

After **every** stage, verify all items before proceeding.

### Gate 0 — MarketData (CSV Load)

| # | Check | Pass |
|---|-------|------|
| 1 | Row count matches source CSV | ☐ |
| 2 | All six base columns present | ☐ |
| 3 | dtypes correct (`Date` datetime, OHLCV numeric) | ☐ |
| 4 | No columns removed from prior state | ☐ |
| 5 | Index reset and aligned (0…N-1 or consistent DatetimeIndex) | ☐ |

---

### Gate 1 — SwingDetector

| # | Check | Pass |
|---|-------|------|
| 1 | `Swing_High`, `Swing_Low` columns added | ☐ |
| 2 | Non-swing rows are `NaN` (not `0`, not `False`) | ☐ |
| 3 | Swing values equal actual `High` / `Low` at labelled rows | ☐ |
| 4 | Edge rows (first/last `lookback` bars) are `NaN` | ☐ |
| 5 | Strict comparison: swing high > neighbours; swing low < neighbours | ☐ |
| 6 | Prior CSV columns preserved | ☐ |
| 7 | No duplicate column names introduced | ☐ |

**Spot-check:** Manually verify 3 swing highs and 3 swing lows against TradingView pivot settings (see comparison guide).

---

### Gate 2 — MarketStructure

| # | Check | Pass |
|---|-------|------|
| 1 | `HH`, `HL`, `LH`, `LL` columns added | ☐ |
| 2 | Labels appear only on swing rows (same index as `Swing_High` / `Swing_Low`) | ☐ |
| 3 | First swing high/low of each series is unlabelled (`NaN`) | ☐ |
| 4 | HH only when current swing high > previous swing high | ☐ |
| 5 | LH only when current swing high < previous swing high | ☐ |
| 6 | HL only when current swing low > previous swing low | ☐ |
| 7 | LL only when current swing low < previous swing low | ☐ |
| 8 | Equal consecutive swings produce no label (both `NaN`) | ☐ |
| 9 | Swing columns unchanged | ☐ |

---

### Gate 3 — TrendEngine

| # | Check | Pass |
|---|-------|------|
| 1 | `Trend`, `Trend_Strength` columns added | ☐ |
| 2 | `Trend` values ∈ `{BULLISH, BEARISH, SIDEWAYS}` only | ☐ |
| 3 | `Trend_Strength` values ∈ `{0, 1, 2, 3}` | ☐ |
| 4 | BULLISH requires both HH and HL evidence (not a single label) | ☐ |
| 5 | BEARISH requires both LH and LL evidence | ☐ |
| 6 | Mixed HH/LH or HL/LL without full flip → SIDEWAYS | ☐ |
| 7 | Trend is forward-filled between structure events (every row populated) | ☐ |
| 8 | Trend flip requires opposing structure confirmation (LH+LL or HH+HL) | ☐ |
| 9 | Structure columns unchanged | ☐ |

---

### Gate 4 — BreakOfStructure

| # | Check | Pass |
|---|-------|------|
| 1 | `Bullish_BOS`, `Bearish_BOS` columns added | ☐ |
| 2 | Bullish BOS: `Close` > last HH, prior `Close` ≤ last HH | ☐ |
| 3 | Bearish BOS: `Close` < last LL, prior `Close` ≥ last LL | ☐ |
| 4 | **No same-bar repaint:** HH/LL labelled on bar T not used for BOS on bar T | ☐ |
| 5 | No duplicate BOS while price remains beyond level | ☐ |
| 6 | BOS value equals confirming `Close` | ☐ |
| 7 | Non-event rows are `NaN` | ☐ |

---

### Gate 5 — ChangeOfCharacter

| # | Check | Pass |
|---|-------|------|
| 1 | `Bullish_CHOCH`, `Bearish_CHOCH` columns added | ☐ |
| 2 | Bullish CHOCH only after bearish bias (LH or LL present) | ☐ |
| 3 | Bearish CHOCH only after bullish bias (HH or HL present) | ☐ |
| 4 | Bullish CHOCH: `Close` > last LH, prior `Close` ≤ last LH | ☐ |
| 5 | Bearish CHOCH: `Close` < last HL, prior `Close` ≥ last HL | ☐ |
| 6 | **No same-bar repaint:** LH/HL on bar T not used for CHOCH on bar T | ☐ |
| 7 | No duplicate CHOCH while beyond level | ☐ |
| 8 | CHOCH value equals confirming `Close` | ☐ |

---

### Gate 6 — FairValueGap

| # | Check | Pass |
|---|-------|------|
| 1 | Four FVG columns added (`Bullish_FVG_Top/Bottom`, `Bearish_FVG_Top/Bottom`) | ☐ |
| 2 | Bullish FVG: candle-3 `Low` > candle-1 `High` | ☐ |
| 3 | Bearish FVG: candle-3 `High` < candle-1 `Low` | ☐ |
| 4 | Gap boundaries stored on **third candle** of pattern | ☐ |
| 5 | Bullish bottom = candle-1 high; bullish top = candle-3 low | ☐ |
| 6 | Bearish top = candle-1 low; bearish bottom = candle-3 high | ☐ |
| 7 | Non-gap rows are `NaN` | ☐ |
| 8 | Minimum 3 rows required; pipeline handles short datasets gracefully | ☐ |

---

### Gate 7 — OrderBlockDetector

| # | Check | Pass |
|---|-------|------|
| 1 | Six OB columns added (high/low + mitigated for each side) | ☐ |
| 2 | Bullish OB: last **bearish** candle before bullish displacement ending at BOS | ☐ |
| 3 | Bearish OB: last **bullish** candle before bearish displacement ending at BOS | ☐ |
| 4 | **Order block position < BOS position** (OB always before BOS) | ☐ |
| 5 | OB zone uses origin candle `High`/`Low` | ☐ |
| 6 | Weak/doji origin candles rejected (body ratio filter) | ☐ |
| 7 | Overlapping duplicate OBs of same direction suppressed | ☐ |
| 8 | Bullish mitigation: `Low` < OB low after BOS bar | ☐ |
| 9 | Bearish mitigation: `High` > OB high after BOS bar | ☐ |
| 10 | `order_blocks` records align with column projections | ☐ |

---

### Gate 8 — LiquidityDetector

| # | Check | Pass |
|---|-------|------|
| 1 | Seven liquidity columns added | ☐ |
| 2 | EQH/EQL formed only from `Swing_High` / `Swing_Low` (not raw OHLC) | ☐ |
| 3 | Cluster requires ≥ 2 touches within tolerance (default 0.10%) | ☐ |
| 4 | Buy-side pool at max equal-high level; sell-side at min equal-low | ☐ |
| 5 | Pool projection starts on **second touch** (confirmation bar), not first | ☐ |
| 6 | Strength: 2 touches → 1, 3 → 2, 4+ → 3 | ☐ |
| 7 | Buy sweep: `High` > pool AND `Close` < pool | ☐ |
| 8 | Sell sweep: `Low` < pool AND `Close` > pool | ☐ |
| 9 | Sweep recorded once per pool | ☐ |
| 10 | `liquidity_pools` records align with detected sweeps | ☐ |

---

## Post-Pipeline Integration Checks

| # | Check | Pass |
|---|-------|------|
| 1 | All prior columns still present (no column deletion) | ☐ |
| 2 | No duplicate column names in final frame | ☐ |
| 3 | Final row count equals source row count | ☐ |
| 4 | `validation_output.csv` written successfully | ☐ |
| 5 | Column count = 6 base + 2 swing + 4 structure + 2 trend + 2 BOS + 2 CHOCH + 4 FVG + 6 OB + 7 liquidity = **35** | ☐ |
| 6 | Missing-value counts documented (sparse SMC columns expected) | ☐ |
| 7 | Pipeline completes without exception | ☐ |
| 8 | Total execution time recorded | ☐ |

---

## Final Foundation Acceptance Checklist

Sign off only when **all** items pass or have an approved documented exception.

### Causal Integrity

| # | Criterion | Pass |
|---|-----------|------|
| ✓ | **No repaint** — BOS/CHOCH do not use structure labels from the same bar | ☐ |
| ✓ | **No future leak** — swing detection uses symmetric lookback (confirmed only after right-side bars exist in batch mode) | ☐ |
| ✓ | **No duplicate BOS** — one event per level break sequence | ☐ |
| ✓ | **No duplicate CHOCH** — one event per level break sequence | ☐ |
| ✓ | **Chronological processing** — all detectors scan rows in index order | ☐ |

### Structural Consistency

| # | Criterion | Pass |
|---|-----------|------|
| ✓ | **Swing → Structure alignment** — HH/LH only on swing highs; HL/LL only on swing lows | ☐ |
| ✓ | **Trend matches structure** — BULLISH/BEARISH/SIDEWAYS consistent with HH/HL/LH/LL sequence | ☐ |
| ✓ | **BOS uses correct references** — HH for bullish, LL for bearish | ☐ |
| ✓ | **CHOCH uses correct references** — LH after bearish bias, HL after bullish bias | ☐ |

### Institutional Pattern Integrity

| # | Criterion | Pass |
|---|-----------|------|
| ✓ | **Order block before BOS** — OB position strictly less than BOS position | ☐ |
| ✓ | **Order block polarity correct** — bearish candle → bullish OB; bullish candle → bearish OB | ☐ |
| ✓ | **FVG three-candle rule** — wick gap between candle 1 and candle 3 | ☐ |
| ✓ | **Liquidity from swings only** — no EQH/EQL from non-swing prices | ☐ |
| ✓ | **Liquidity activation timing** — pool not active before second touch | ☐ |
| ✓ | **Sweep rejection** — wick through level with close back across level | ☐ |

### Data Contract Integrity

| # | Criterion | Pass |
|---|-----------|------|
| ✓ | **Column preservation** — no detector deletes prior columns | ☐ |
| ✓ | **No duplicate columns** — unique column names throughout | ☐ |
| ✓ | **Required inputs present** — each stage receives columns it requires | ☐ |
| ✓ | **Expected outputs created** — each stage adds its declared columns | ☐ |
| ✓ | **Sparse event columns** — non-events stored as `NaN`, not zero | ☐ |

### TradingView Alignment (Sample Set)

| # | Criterion | Pass |
|---|-----------|------|
| ✓ | ≥ 80% swing point agreement on reference chart (same TF, same lookback) | ☐ |
| ✓ | Structure labels match engine on agreed swing points | ☐ |
| ✓ | FVG zones overlap TradingView ICT FVG indicator boundaries | ☐ |
| ✓ | Documented discrepancies logged with reason code | ☐ |

### Known Foundation v1 Caveats (Track, Do Not Block Release Unless Critical)

These are **documented current behaviours** to track in validation logs. They do not fail the pipeline but may differ from ideal ICT live-trading semantics:

| # | Caveat | Tracked |
|---|--------|---------|
| ⚠ | Liquidity pool columns **continue forward-filling after sweep** (pool level not cleared post-sweep in columns) | ☐ |
| ⚠ | Liquidity clustering uses **full-batch swing history** (not incremental/live confirmation) | ☐ |
| ⚠ | Overlapping liquidity pools may **overwrite** column values silently | ☐ |
| ⚠ | Swing detection is **batch-confirmed** (requires future bars within lookback window) | ☐ |
| ⚠ | Order block count may be **lower than TradingView** due to displacement/body filters | ☐ |

---

## Validation Sign-Off Template

```
Dataset         :
Timeframe       :
Validator       :
Date            :
Engine version  :
CSV rows        :
Pipeline status : PASS / FAIL
TV comparison   : PASS / PARTIAL / NOT RUN
Exceptions      :
Approver        :
```

---

## Quick Reference — Column Inventory

| Stage | Columns Added |
|-------|---------------|
| CSV | `Date`, `Open`, `High`, `Low`, `Close`, `Volume` |
| SwingDetector | `Swing_High`, `Swing_Low` |
| MarketStructure | `HH`, `HL`, `LH`, `LL` |
| TrendEngine | `Trend`, `Trend_Strength` |
| BreakOfStructure | `Bullish_BOS`, `Bearish_BOS` |
| ChangeOfCharacter | `Bullish_CHOCH`, `Bearish_CHOCH` |
| FairValueGap | `Bullish_FVG_Top`, `Bullish_FVG_Bottom`, `Bearish_FVG_Top`, `Bearish_FVG_Bottom` |
| OrderBlockDetector | `Bullish_OB_High`, `Bullish_OB_Low`, `Bearish_OB_High`, `Bearish_OB_Low`, `Bullish_OB_Mitigated`, `Bearish_OB_Mitigated` |
| LiquidityDetector | `Equal_High`, `Equal_Low`, `Buy_Side_Liquidity`, `Sell_Side_Liquidity`, `Buy_Liquidity_Sweep`, `Sell_Liquidity_Sweep`, `Liquidity_Strength` |

---

## Related Documents

- `Module_Validation_Guide.md` — step-by-step module validation
- `TradingView_Comparison_Guide.md` — chart comparison procedures
- `src/validation/engine_validation.py` — automated integration runner
