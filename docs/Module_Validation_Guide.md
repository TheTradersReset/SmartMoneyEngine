# Module Validation Guide

**Project:** SmartMoneyEngine  
**Scope:** Per-module validation procedures for the SMC foundation pipeline  
**Audience:** Quant validation engineers, QA analysts

---

## How to Use This Guide

1. Load a reference CSV and run the pipeline stage-by-stage (or use `engine_validation.py`).
2. For each module below, execute **Validation Steps** in order.
3. Compare results against **Expected Behaviour** and **Acceptance Criteria**.
4. Log discrepancies using the reason codes in **Common Failure Cases**.
5. Cross-check visually using `TradingView_Comparison_Guide.md`.

Each module section contains:

| Section | Purpose |
|---------|---------|
| Purpose | Why the module exists |
| Inputs | Required upstream columns |
| Outputs | Columns and records produced |
| Validation Steps | Ordered verification procedure |
| Expected Behaviour | Deterministic rules from implementation |
| Common Failure Cases | Typical bugs and misconfigurations |
| ICT Behaviour | Institutional Smart Money Concepts reference |
| TradingView Comparison Method | How to verify on charts |
| Acceptance Criteria | Pass/fail gates |

---

## Pipeline Context

```
CSV → SwingDetector → MarketStructure → TrendEngine → BreakOfStructure
    → ChangeOfCharacter → FairValueGap → OrderBlockDetector → LiquidityDetector
```

---

# 1. SwingDetector

## Purpose

Identify **confirmed swing highs and swing lows** from OHLC data. Swings are the foundation for all downstream structure, liquidity, and (indirectly) institutional pattern analysis.

## Inputs

| Column | Required |
|--------|----------|
| `High` | Yes |
| `Low` | Yes |

**Parameters:** `lookback=2` (default) — candles compared on each side.

**Minimum rows:** `lookback * 2 + 1` (= 5 with default lookback).

## Outputs

| Column | Description |
|--------|-------------|
| `Swing_High` | High price at swing-high candles; `NaN` elsewhere |
| `Swing_Low` | Low price at swing-low candles; `NaN` elsewhere |

## Validation Steps

1. Confirm `Swing_High` and `Swing_Low` exist after detection.
2. Count non-null swing highs/lows; verify count is reasonable for dataset length.
3. For each non-null `Swing_High` at index `i`:
   - Verify `Swing_High[i] == High[i]`.
   - Verify `High[i] > High[i-1]` … `High[i-lookback]` (strict).
   - Verify `High[i] > High[i+1]` … `High[i+lookback]` (strict).
4. Repeat for `Swing_Low` with `<` comparison against neighbouring lows.
5. Confirm indices `0…lookback-1` and `N-lookback…N-1` are `NaN` (edge exclusion).
6. Verify no prior columns were removed or renamed.

## Expected Behaviour

- **Strict inequality:** equal highs/lows with neighbours disqualify the bar.
- **Price storage:** actual price values, not booleans or flags.
- **Edge exclusion:** insufficient neighbours → no swing label.
- **Independent detection:** swing highs and swing lows evaluated separately (a bar can theoretically be both in rare equal-wick cases, but strict rules usually prevent overlap).
- **Batch confirmation:** a swing at bar `i` requires `lookback` future bars — in live mode this implies delayed confirmation.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| False positive swing | Label on non-extreme bar | Lookback mismatch vs TradingView pivot length |
| False negative swing | Missing obvious pivot | Strict `>` vs `>=`; TV uses different pivot rule |
| Zeros instead of NaN | Non-events show `0.0` | Wrong output dtype / fill logic |
| Swings at edges | Labels in first/last bars | Edge guard not applied |
| Too many swings | Noisy labels every few bars | Lookback too small |

## ICT Behaviour

ICT market structure begins with **protected swing points** — liquidity rests above swing highs and below swing lows. Swings must be:

- Objectively defined (same inputs → same outputs)
- Non-repainting once confirmed (in live trading, after right-side bars close)
- Stable across re-runs on the same historical data

## TradingView Comparison Method

See `TradingView_Comparison_Guide.md` § Swing Points. Summary:

- Use **Pivot Points High/Low** or an ICT structure indicator with equivalent left/right bars = `lookback`.
- Match timeframe and session exactly.
- Compare dates/prices of each engine swing to nearest TV pivot within 0.01% tolerance.

## Acceptance Criteria

- [ ] All swing labels satisfy strict neighbour comparison for configured `lookback`
- [ ] Edge rows are `NaN`
- [ ] Values equal source `High`/`Low` at labelled rows
- [ ] ≥ 80% agreement with TradingView pivots on reference dataset
- [ ] No column deletion from input frame

---

# 2. MarketStructure

## Purpose

Classify consecutive swing points into **HH, HL, LH, LL** labels — the vocabulary of institutional market structure.

## Inputs

| Column | Required |
|--------|----------|
| `Swing_High` | Yes |
| `Swing_Low` | Yes |

## Outputs

| Column | Description |
|--------|-------------|
| `HH` | Higher High — current swing high > previous swing high |
| `LH` | Lower High — current swing high < previous swing high |
| `HL` | Higher Low — current swing low > previous swing low |
| `LL` | Lower Low — current swing low < previous swing low |

Labels store the swing **price** at the classified candle; all other rows are `NaN`.

## Validation Steps

1. Extract chronological non-null swing highs; walk pairs:
   - Second > first → `HH` on second index
   - Second < first → `LH` on second index
   - Second == first → both `NaN` (no label)
2. Repeat for swing lows → `HL` / `LL`.
3. Confirm first swing of each type has no structure label.
4. Verify labels appear **only** at swing indices (never on non-swing rows).
5. Cross-check: every `HH`/`LH` row has non-null `Swing_High`; every `HL`/`LL` row has non-null `Swing_Low`.

## Expected Behaviour

- Sequential comparison of **consecutive swings of the same type** (not alternating high/low).
- First swing high and first swing low are unclassified.
- Equal consecutive swings produce no label.
- HH/LH never appear on swing-low rows; HL/LL never on swing-high rows.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| Label on first swing | HH/LH/HL/LL on earliest swing | Missing "first swing skip" logic |
| Label on non-swing row | Structure without swing | Join misalignment |
| Both HH and LH set | Same index labelled twice | Should be mutually exclusive |
| Wrong classification | HH when price lower | Comparing wrong predecessor |
| Alternating comparison | HL derived from prior swing high | Incorrect swing series used |

## ICT Behaviour

- **Bullish structure:** HH + HL sequence (higher highs and higher lows)
- **Bearish structure:** LH + LL sequence (lower highs and lower lows)
- **CHOCH precursor:** LH after HH sequence (bearish shift) or HL after LL sequence (bullish shift)
- Structure labels describe **swing points**, not every candle close

## TradingView Comparison Method

- Overlay HH/HL/LH/LL markers from an ICT Market Structure indicator.
- Align on **same swing points** first; structure labels should match at agreed swings.
- Document TV indicators that label on close break vs swing point (engine uses swing points only).

## Acceptance Criteria

- [ ] First swing of each type unlabelled
- [ ] 100% internal consistency: labels derivable from swing series alone
- [ ] No label when consecutive swings equal
- [ ] Mutual exclusivity: HH xor LH per swing high; HL xor LL per swing low
- [ ] Swing columns unchanged

---

# 3. TrendEngine

## Purpose

Derive a **stable trend regime** (`BULLISH`, `BEARISH`, `SIDEWAYS`) and **strength score** (0–3) from classified structure labels.

## Inputs

| Column | Required |
|--------|----------|
| `HH` | Yes |
| `HL` | Yes |
| `LH` | Yes |
| `LL` | Yes |

## Outputs

| Column | Description |
|--------|-------------|
| `Trend` | `BULLISH`, `BEARISH`, or `SIDEWAYS` on every row |
| `Trend_Strength` | Integer 0–3 on every row |

**Strength mapping:**

| Value | Meaning |
|-------|---------|
| 0 | No structure events processed yet |
| 1 | Weak (sideways or minimal confirmation) |
| 2 | Medium (≥1 primary + ≥1 secondary label in direction) |
| 3 | Strong (≥2 primary + ≥2 secondary labels in direction) |

## Validation Steps

1. Confirm every row has non-null `Trend` and `Trend_Strength`.
2. Walk structure events chronologically; manually track expected trend state.
3. Verify BULLISH entry requires HH **and** HL (from sideways: both seen, no opposing LH/LL mix).
4. Verify BEARISH entry requires LH **and** LL.
5. Verify trend flip from BULLISH → BEARISH requires both LH and LL while in bullish trend.
6. Verify flip from BEARISH → BULLISH requires both HH and HL while in bearish trend.
7. Verify mixed structure (e.g. HH and LH both present in sideways phase) → SIDEWAYS.
8. Check strength increments as additional HH/HL (or LH/LL) labels accumulate.

## Expected Behaviour

- Trend updates only on structure event rows; value forward-fills to subsequent rows.
- Isolated counter-label does not flip trend (requires paired opposing confirmation).
- SIDEWAYS strength always = 1 (once events exist).
- Strength 0 only before any structure label processed.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| Premature BULLISH | Single HH sets bullish | Missing HL requirement |
| Stuck SIDEWAYS | Clear trend not detected | Mixed evidence guard too strict |
| Flip on single LH | Bearish after one lower high | Missing LL confirmation |
| Strength always 3 | Over-scoring | Count logic error |
| NaN trend rows | Gaps in trend column | Forward-fill missing |

## ICT Behaviour

ICT trend is inferred from **order flow direction** visible in swing structure:

- Bullish: buyers defending higher lows, making higher highs
- Bearish: sellers defending lower highs, making lower lows
- Consolidation: overlapping / mixed structure

Trend should **lag** the first counter-structure signal until full opposing confirmation (CHOCH/BOS semantics downstream depend on this stability).

## TradingView Comparison Method

- TV rarely exposes a single "Trend" column — compare against visual structure shading or custom ICT trend indicators.
- Validate manually on 3 clear bullish, 3 clear bearish, and 2 ranging segments.
- Strength is engine-specific; compare direction only unless TV indicator exposes equivalent scoring.

## Acceptance Criteria

- [ ] Trend populated on 100% of rows
- [ ] No flip without paired opposing structure confirmation
- [ ] BULLISH/BEARISH consistent with dominant HH/HL or LH/LL sequence
- [ ] SIDEWAYS correctly assigned in mixed/choppy segments
- [ ] Strength monotonically reflects accumulating structure in active trend

---

# 4. BreakOfStructure (BOS)

## Purpose

Detect **Break of Structure** — continuation signals where price **closes** beyond the most recent protected structural level:

- **Bullish BOS:** close breaks above last **HH**
- **Bearish BOS:** close breaks below last **LL**

## Inputs

| Column | Required |
|--------|----------|
| `HH`, `HL`, `LH`, `LL` | Yes |
| `Close` | Yes |

## Outputs

| Column | Description |
|--------|-------------|
| `Bullish_BOS` | Close price at bullish break; `NaN` elsewhere |
| `Bearish_BOS` | Close price at bearish break; `NaN` elsewhere |

## Validation Steps

1. Scan rows chronologically maintaining `last_hh` and `last_ll`.
2. **Before** updating levels from current row, evaluate close against prior levels.
3. Bullish BOS at row `i` iff:
   - `Close[i] > last_hh`
   - `Close[i-1] <= last_hh` (or first row exception)
4. Bearish BOS symmetric with `last_ll`.
5. Confirm no BOS on same bar where HH/LL is first assigned (no-repaint).
6. Confirm no second BOS while closes remain beyond level without reset.
7. BOS value must equal `Close` at event row.

## Expected Behaviour

- Uses **close**, not wick, for break confirmation.
- HH/LL updated **after** BOS check on each row (causal ordering).
- One BOS event per level breach sequence.
- BOS indicates **continuation** in prevailing structure direction.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| Same-bar repaint | BOS when HH forms same candle | Level updated before check |
| Duplicate BOS | Multiple labels above same HH | Missing prior-close guard |
| Wick-only break | BOS without close beyond level | Using High/Low instead of Close |
| Wrong reference | Bullish BOS vs HL | Incorrect level tracked |
| Missing BOS | Obvious break not flagged | HH not yet established |

## ICT Behaviour

- **BOS = continuation** — smart money pushing trend forward
- Must be discrete, non-repainting events
- Close confirmation filters liquidity sweeps that fail to hold
- Bullish BOS in bearish structure may instead be classified as CHOCH (different module)

## TradingView Comparison Method

- Use ICT BOS indicator or manual horizontal line at last HH/LL.
- Draw line at last confirmed HH; first close above = bullish BOS bar.
- Account for TV indicators that use wick breaks (engine uses close only — document delta).

## Acceptance Criteria

- [ ] Close-only confirmation
- [ ] No same-bar HH/LL reference repaint
- [ ] No duplicate events per level
- [ ] Bullish BOS references HH only; bearish references LL only
- [ ] Event price equals Close

---

# 5. ChangeOfCharacter (CHOCH)

## Purpose

Detect **Change of Character** — early **reversal** signals:

- **Bullish CHOCH:** after bearish structure, close breaks above last **LH**
- **Bearish CHOCH:** after bullish structure, close breaks below last **HL**

## Inputs

| Column | Required |
|--------|----------|
| `HH`, `HL`, `LH`, `LL` | Yes |
| `Close` | Yes |

## Outputs

| Column | Description |
|--------|-------------|
| `Bullish_CHOCH` | Close at bullish reversal; `NaN` elsewhere |
| `Bearish_CHOCH` | Close at bearish reversal; `NaN` elsewhere |

## Validation Steps

1. Track `structure_bias`: BULLISH if HH or HL on row; BEARISH if LH or LL; else carry forward.
2. Track `last_lh` and `last_hl` (updated after check, like BOS).
3. Bullish CHOCH requires:
   - `structure_bias == BEARISH`
   - `Close[i] > last_lh`
   - `Close[i-1] <= last_lh`
4. Bearish CHOCH requires:
   - `structure_bias == BULLISH`
   - `Close[i] < last_hl`
   - `Close[i-1] >= last_hl`
5. Verify no CHOCH on bar where LH/HL first appears (no-repaint).
6. Verify CHOCH precedes or coincides with trend flip (not after full BOS continuation).

## Expected Behaviour

- CHOCH is the **first structural warning** of reversal against current bias.
- Bias from labels on current row applied **after** CHOCH evaluation.
- Mutually independent bullish/bearish columns (both can be NaN; never both set same row).
- Close-only confirmation with prior-close guard.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| CHOCH in wrong bias | Bullish CHOCH during HH sequence | Bias tracking error |
| Same-bar LH repaint | CHOCH on LH formation bar | Level used before prior bar close |
| CHOCH confused with BOS | Label on HH break | Wrong reference level |
| Duplicate CHOCH | Multiple breaks same LH | Missing dedup guard |
| Missing CHOCH | Reversal not flagged | Bias not yet bearish/bullish |

## ICT Behaviour

- **CHOCH = first sign of reversal** (character change)
- **BOS = continuation** after structure already aligned
- CHOCH typically appears **before** full trend engine flip
- LH break in downtrend → bullish CHOCH; HL break in uptrend → bearish CHOCH

## TradingView Comparison Method

- Mark last LH in bearish leg; first close above = bullish CHOCH.
- Mark last HL in bullish leg; first close below = bearish CHOCH.
- Compare event bar dates; allow ±1 bar if TV uses wick vs close.

## Acceptance Criteria

- [ ] Correct bias gating (bearish for bullish CHOCH; bullish for bearish CHOCH)
- [ ] No same-bar LH/HL repaint
- [ ] Close-only with prior-close deduplication
- [ ] CHOCH references LH (bullish) or HL (bearish) only
- [ ] Events are sparse (`NaN` dominant)

---

# 6. FairValueGap (FVG)

## Purpose

Detect **Fair Value Gaps** — three-candle imbalances where outer wicks do not overlap, indicating inefficient price delivery.

## Inputs

| Column | Required |
|--------|----------|
| `High` | Yes |
| `Low` | Yes |

**Minimum rows:** 3

## Outputs

| Column | Description |
|--------|-------------|
| `Bullish_FVG_Top` | Upper boundary (candle-3 low) |
| `Bullish_FVG_Bottom` | Lower boundary (candle-1 high) |
| `Bearish_FVG_Top` | Upper boundary (candle-1 low) |
| `Bearish_FVG_Bottom` | Lower boundary (candle-3 high) |

All values on **third candle** of pattern; `NaN` elsewhere.

## Validation Steps

1. For each index `i >= 2`, define candles `i-2`, `i-1`, `i`.
2. Bullish FVG if `Low[i] > High[i-2]`:
   - Bottom = `High[i-2]`, Top = `Low[i]`
3. Bearish FVG if `High[i] < Low[i-2]`:
   - Top = `Low[i-2]`, Bottom = `High[i]`
4. Verify top > bottom for bullish; top > bottom for bearish zone semantics.
5. Middle candle (`i-1`) is the impulse candle (not stored separately).
6. Both bullish and bearish can be NaN on same row (normal).

## Expected Behaviour

- Pure OHLC geometry — no structure dependency.
- Wick-based gap (High/Low), not body-based.
- Label anchored on third candle (completion bar).
- Multiple FVGs can exist in dataset; no deduplication.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| Body gap used | FVG where wicks overlap | Using open/close instead of high/low |
| Wrong anchor candle | Gap on middle candle | Index off-by-one |
| Inverted boundaries | Top < bottom | Swap top/bottom assignment |
| Missing gap | Visible imbalance not flagged | Strict inequality vs TV tolerance |
| Extra gaps | Too many FVGs | TV requires minimum gap size filter |

## ICT Behaviour

- FVG = **imbalance** / inefficiency; price often revisits to mitigate
- Bullish FVG acts as potential support; bearish as resistance
- ICT teaches three-candle FVG with non-overlapping outer wicks (engine matches)
- Some practitioners filter by displacement/context (engine does not — expect more FVGs)

## TradingView Comparison Method

- Use "Fair Value Gap" or ICT FVG indicator with default settings.
- Compare top/bottom prices at each gap within tick tolerance.
- Match third-candle anchor date.
- Document TV filters (min size, CE/ICT variant) if counts differ.

## Acceptance Criteria

- [ ] Three-candle wick rule correctly applied
- [ ] Boundaries match candle 1 and candle 3 wicks
- [ ] Label on third candle only
- [ ] ≥ 90% boundary overlap with TradingView on reference gaps
- [ ] No dependency on structure columns

---

# 7. OrderBlockDetector

## Purpose

Detect **Order Blocks (OB)** — the final opposing candle before an impulsive displacement that produces a confirmed BOS. Institutional zones where unfilled orders may remain.

## Inputs

| Column | Required |
|--------|----------|
| `Open`, `High`, `Low`, `Close` | Yes |
| `Bullish_BOS`, `Bearish_BOS` | Yes |

**Default parameters:**

| Parameter | Default |
|-----------|---------|
| `rolling_window` | 14 |
| `min_body_ratio` | 0.35 |
| `min_displacement_body_ratio` | 0.50 |
| `min_displacement_multiplier` | 1.25 |
| `equal_level_tolerance_ratio` | 0.05 |
| `overlap_threshold` | 0.50 |

## Outputs

| Column | Description |
|--------|-------------|
| `Bullish_OB_High`, `Bullish_OB_Low` | Bullish OB zone |
| `Bearish_OB_High`, `Bearish_OB_Low` | Bearish OB zone |
| `Bullish_OB_Mitigated`, `Bearish_OB_Mitigated` | Boolean mitigation flags |
| `order_blocks` | Tuple of `OrderBlockRecord` (programmatic access) |

## Validation Steps

1. For each non-null `Bullish_BOS` at position `b`:
   - Trace displacement leg backward from `b`.
   - Find last **bearish** candle (`Close < Open`) before displacement.
   - Verify OB position `< b`.
   - Verify displacement candle passes impulsive body/range filters.
2. Repeat for bearish BOS with last **bullish** origin candle.
3. Verify OB zone = origin candle high/low.
4. Check weak/doji origins rejected (`min_body_ratio`).
5. Check overlapping same-direction OBs suppressed (`overlap_threshold`).
6. Mitigation scan starts at `bos_position + 1`:
   - Bullish mitigated when any `Low < OB low`
   - Bearish mitigated when any `High > OB high`
7. Cross-check `order_blocks` records with column projections.

## Expected Behaviour

- OB **requires** valid BOS — no BOS → no OB.
- Origin polarity inverted relative to displacement direction.
- Equal-high/low origin rejection near neighbouring bars.
- Mitigation is binary; first touch through zone edge triggers flag.
- Duplicate overlapping blocks of same direction rejected.

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| OB after BOS | OB position ≥ BOS | Origin search wrong direction |
| Wrong polarity | Bullish OB on bullish candle | Origin predicate inverted |
| Too many OBs | Label on every BOS | Filters too permissive |
| Too few OBs | Missing obvious OB | Displacement filter too strict |
| Early mitigation | Mitigated on BOS bar | Scan starts before bos+1 |
| No mitigation | Price clearly traded through zone | Edge comparison uses wrong bound |

## ICT Behaviour

- **Bullish OB:** last down-close candle before up displacement breaking structure
- **Bearish OB:** last up-close candle before down displacement
- OB is a **zone** (full candle range), not a single price
- Mitigation = price returns to fill the inefficiency left by institutional move
- Not every large candle is an OB — context (BOS displacement) required

## TradingView Comparison Method

- Manually mark last opposing candle before displacement leg on TV.
- Compare zone boundaries (high/low of origin candle).
- Expect fewer engine OBs than discretionary TV markup due to automated filters.
- Validate mitigation bar: first subsequent candle trading through OB edge.

## Acceptance Criteria

- [ ] OB position strictly before BOS position
- [ ] Correct origin candle polarity
- [ ] Impulsive displacement filter applied at BOS bar
- [ ] Overlapping duplicates suppressed
- [ ] Mitigation logic fires on first valid touch after BOS
- [ ] Records match column output

---

# 8. LiquidityDetector

## Purpose

Detect **institutional liquidity pools** from equal highs (buy-side) and equal lows (sell-side), track **strength**, and identify **sweeps** (wick through, close reject).

## Inputs

| Column | Required |
|--------|----------|
| `Swing_High`, `Swing_Low` | Yes |
| `High`, `Low`, `Close` | Yes |

**Default:** `tolerance_ratio = 0.001` (0.10% midpoint tolerance).

## Outputs

| Column | Description |
|--------|-------------|
| `Equal_High`, `Equal_Low` | Cluster level at touch swings |
| `Buy_Side_Liquidity` | Active BSL pool level (forward-filled from confirmation) |
| `Sell_Side_Liquidity` | Active SSL pool level |
| `Buy_Liquidity_Sweep`, `Sell_Liquidity_Sweep` | Sweep price when detected |
| `Liquidity_Strength` | 1–3 strength score |
| `liquidity_pools` | Tuple of `LiquidityPoolRecord` |

## Validation Steps

1. Extract swing points chronologically.
2. Cluster swings within tolerance (midpoint-relative distance).
3. Clusters with < 2 touches → no pool.
4. EQH/EQL marked at each touch in valid cluster.
5. Pool level: max price for buy-side; min for sell-side.
6. `Buy_Side_Liquidity` / `Sell_Side_Liquidity` start at **second touch** index.
7. Strength: 2 touches → 1; 3 → 2; 4+ → 3.
8. Buy sweep: `High > pool` AND `Close < pool` (first occurrence per pool).
9. Sell sweep: `Low < pool` AND `Close > pool`.
10. Verify `liquidity_pools[n].swept` matches sweep columns.

## Expected Behaviour

- Liquidity derived **only** from confirmed swings (not raw equal OHLC).
- Pool inactive before cluster confirmation (second touch).
- One sweep event per pool maximum.
- Forward-fill of pool level from confirmation bar onward.
- **Current v1:** pool columns **not cleared** after sweep (see caveats).

## Common Failure Cases

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| Pool from raw highs | EQH without swing | Using OHLC instead of swings |
| Pool on first touch | BSL active too early | Confirmation timing wrong |
| No cluster | Swings visually equal but not clustered | Tolerance too tight |
| False cluster | Unrelated swings grouped | Tolerance too wide |
| Sweep on close hold | Sweep when price stays beyond level | Missing close rejection check |
| Missing sweep | Obvious wick-and-reject not flagged | Pool not active yet |
| Pool persists after sweep | BSL still shown post-sweep | Known v1 forward-fill behaviour |

## ICT Behaviour

- **Buy-side liquidity (BSL)** rests above equal highs — stop losses of shorts
- **Sell-side liquidity (SSL)** rests below equal lows — stop losses of longs
- **Sweep** = run stops then reject (manipulation / stop hunt)
- Strength increases with touch count (more resting orders)
- After sweep, liquidity is considered **taken** — ideal systems deactivate pool

## TradingView Comparison Method

- Mark equal highs/lows from **same swing points** as engine.
- Draw horizontal at pool level from second touch forward.
- Identify sweep bars manually: wick beyond, close back inside.
- Compare tolerance: TV manual equality vs 0.10% engine tolerance.

## Acceptance Criteria

- [ ] Clusters from swings only, minimum 2 touches
- [ ] Activation on second touch, not first
- [ ] Correct strength scoring
- [ ] Sweep = wick through + close rejection
- [ ] One sweep per pool in records
- [ ] Document if pool columns persist after sweep (known v1 caveat)

---

# Cross-Module Integration Rules

| Rule | Validation |
|------|------------|
| Column monotonicity | Column count only increases through pipeline |
| Index alignment | All columns share same index length |
| Swing → Structure | Structure labels ⊆ swing indices |
| Structure → Trend | Trend reacts only to structure events |
| BOS → OB | Every OB references existing BOS at later position |
| Swings → Liquidity | EQH/EQL values ⊆ swing prices |
| Sparse events | BOS, CHOCH, FVG, OB event columns mostly NaN |
| Dense state | Trend, Trend_Strength populated every row |

---

# Validation Evidence Log Template

```
Module          :
Dataset         :
Timeframe       :
Validator       :
Date            :
Result          : PASS / FAIL / PARTIAL
Steps executed  :
Discrepancies   :
TV comparison   :
Evidence files  : validation_output.csv, screenshots/
Sign-off        :
```

---

## Related Documents

- `SMC_Validation_Checklist.md` — master gate checklist
- `TradingView_Comparison_Guide.md` — chart comparison procedures
