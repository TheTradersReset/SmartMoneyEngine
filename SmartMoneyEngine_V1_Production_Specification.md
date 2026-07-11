# SmartMoneyEngine V1 — Production Specification

**Status:** Frozen design (research synthesis only)  
**Version:** 1.0  
**Date:** 2026-07-06  
**Scope:** NIFTY50, BANKNIFTY, FINNIFTY · 5M / 15M / 1H  
**Method:** No new research. Derived exclusively from completed research exports under `outputs/research/`.

---

## 1. Executive Summary

SmartMoneyEngine V1 is a **Tier-2 institutional momentum system** with optional quality layering. The core signal is a completed structural sequence — **Displacement → CHOCH → BOS → FVG Reclaim** — validated on 502 NIFTY50 signals (41.63% WR, PF 2.63, expectancy 102.48 pts, ~42 signals/month).

V1 production deploys **Raw Tier-2** (no HTF/MI hard gate), **BOS Close entry**, **Structural Swing SL**, and a **three-target exit ladder** culminating in **opposite-liquidity / structure trail** after 1R. Confidence is driven by the **Institutional Quality Score (0–100)** with research-backed boosters and penalties.

---

## 2. Research Inputs (Frozen)

| Research Module | Export | Primary Contribution |
|---|---|---|
| Tier-2 Production Validation | `tier2_production_validation.json` | Raw Tier-2 recommended over filtered variants |
| Tiered Signal Framework | `tiered_signal_framework.json` | Tier-1 / Tier-2 / Tier-3 definitions |
| Institutional Quality Score | `institutional_quality_validation.json` | Score formula, threshold ≥ 70 |
| Sequence Entry Timing | `sequence_entry_timing_validation.json` | Displacement Close = best sequence entry |
| Tier-2 Entry Optimization | `tier2_entry_optimization.json` | BOS Close = best full-universe entry |
| Tier-2 Exit Optimization | `tier2_exit_optimization.json` | Model E: trail structure after 1R |
| Trade Construction | `trade_construction_validation.json` | SL + target architecture |
| Tier-2 Regime Classification | `tier2_regime_classification.json` | Regime tags and prioritization |
| Tier-2 Composite Edge | `tier2_composite_edge_validation.json` | Winning trait combinations |
| Tier-2 Winner/Loser Comparison | `tier2_winner_loser_comparison.json` | Confidence boosters / penalties |
| Institutional Signal Construction | `institutional_signal_construction.json` | Pre-move narrative architecture |
| Institutional Confirmation Candle | `institutional_confirmation_candle.json` | Trigger-candle quality scoring |
| Institutional Trigger Validation | `institutional_trigger_validation.json` | Trigger matrix & false triggers |
| Institutional Momentum Origin | `institutional_momentum_origin.json` | Pre-expansion market profile |
| Support / Resistance Pressure | `support_resistance_pressure.json` | Level tests, breaks, bounces |
| Major Level Strength | `major_level_strength.json` | Level strength tiers |
| Liquidity Sweep Tradeability | `liquidity_sweep_tradeability.json` | Sweep quality context |
| Liquidity Move Reconstruction | `liquidity_move_reconstruction.json` | Move sequencing |
| Robust Filter / Production Stack | `robust_filter_report.json`, `production_stack_analysis.json` | Optional filter stack reference |

---

## 3. Signal Tiers

### Tier 2 — V1 Production Base (Mandatory Sequence)

All four events must occur **in order** before the BOS bar close:

1. **Displacement** — medium or strong body in signal direction  
2. **CHOCH** — change of character in signal direction  
3. **BOS** — break of structure in signal direction  
4. **FVG Reclaim** — price reclaims fair value gap in signal direction  

Validated: 502 signals · 41.63% WR · PF 2.63 · expectancy 102.48 pts.

### Tier 1 — Premium Upgrade (Optional Classification)

Adds **Liquidity Sweep** before displacement:

`Liquidity Sweep → Displacement → CHOCH → BOS → FVG Reclaim`

Validated: 60 signals · 41.67% WR · PF 2.60 · **expectancy 146.14 pts** (~5/month).

Use Tier 1 as a **confidence booster**, not a hard gate, to preserve signal frequency.

---

## 4. What Conditions Generate BUY?

A **BUY** is generated when **all mandatory gates pass** (Section 6) and the following **bullish Tier-2 sequence** completes on the active timeframe:

| Step | Condition |
|---|---|
| 1 | **Sell-side or both-side liquidity sweep** occurs in the pre-move window (institutional signal construction: +18.8% lift for both-side sweep on large moves) |
| 2 | **Bullish displacement** candle (medium or strong preferred) |
| 3 | **Bullish CHOCH** prints |
| 4 | **Bullish BOS** closes above broken structure |
| 5 | **Bullish FVG reclaim** confirmed on or before BOS bar |
| 6 | Price context aligns with **support-side interaction**: near support, discount zone, or post-failed-breakdown (support/resistance + momentum origin) |

**High-probability BUY trigger model** (from trigger validation + confirmation candle research):

```
Level Retest x2–3
+ Failed Breakdown
+ Lower Wick Sweep >60%
+ Bullish Engulfing (or Hammer)
+ Strong/Medium Displacement
+ Consolidation → Range Expansion state
```

Example matrix entry: **Level Retest x3 + Level:Moderate + Consolidation** → 100% probability of 100+/200+/300+ move (17–25 samples, indices universe).

**Narrative-frequency BUY pattern** (most common before 100+ bullish moves):

```
Weak/Moderate Level + CHOCH/BOS + RSI oversold + Discount Zone + Engulfing + Hammer
```

---

## 5. What Conditions Generate SELL?

A **SELL** is generated when **all mandatory gates pass** and the following **bearish Tier-2 sequence** completes:

| Step | Condition |
|---|---|
| 1 | **Buy-side or both-side liquidity sweep** in pre-move window |
| 2 | **Bearish displacement** (medium or strong preferred) |
| 3 | **Bearish CHOCH** prints |
| 4 | **Bearish BOS** closes below broken structure |
| 5 | **Bearish FVG reclaim** confirmed |
| 6 | Price context aligns with **resistance-side interaction**: near resistance, premium zone, or post-failed-breakout |

**High-probability SELL trigger model:**

```
Level Retest x2–3
+ Failed Breakout
+ Upper Wick Sweep >60%
+ Bearish Engulfing (or Shooting Star)
+ Strong Displacement
+ Premium Zone + RSI >70
```

**Top SELL confirmation candle** (confirmation candle research):

```
Strong Body + Close Top 20% + Level:Weak + Gap Down + Zone:Premium + RSI 60–70
```

Institutional signal construction magnitude discriminators for SELL: **RSI >70**, **both-side sweep**.

---

## 6. Mandatory vs Optional Filters

### 6.1 Mandatory Filters (Signal cannot fire without these)

| # | Filter | Source |
|---|---|---|
| M1 | **Tier-2 sequence complete** (Displacement, CHOCH, BOS, FVG Reclaim) | Tier-2 production validation |
| M2 | **Valid structural swing SL** calculable below/above entry (min distance > 0, within max risk cap) | Trade construction validation |
| M3 | **BOS direction** matches signal side (bullish BOS → BUY, bearish BOS → SELL) | Tier-2 definition |
| M4 | **Symbol ∈ {NIFTY50, BANKNIFTY, FINNIFTY}** | Research universe |
| M5 | **Timeframe ∈ {5M, 15M, 1H}** | Research universe |
| M6 | **Quality Score ≥ 20** (hard reject 0–20 bucket: 0% WR, negative expectancy) | Institutional quality validation |
| M7 | **No active invalidation** (Section 11) | Cross-research |

> **Note:** Raw Tier-2 is recommended **without** HTF Alignment or Market Intelligence ≥ 65 as mandatory gates — filtered variants reduced balance score (`tier2_production_validation.json`).

### 6.2 Optional Filters (Improve confidence; do not block base signal)

| # | Filter | Effect | Source |
|---|---|---|---|
| O1 | **Quality Score ≥ 70** | Production-grade subset: PF 4.3, expectancy 136.52, DD 457 vs 1547 | Quality validation |
| O2 | **Tier-1 liquidity sweep present** | +expectancy vs Tier-2 baseline | Tiered framework |
| O3 | **Regime = Liquidity Reversal** | Highest regime expectancy (447.4 pts, WR 80%) | Regime classification |
| O4 | **Regime = Trend Continuation** | Best production regime score (PF 3.62, DD 590) | Regime classification |
| O5 | **RSI < 40 (BUY) / RSI > 70 (SELL)** | Composite edge: PF 3.04 / directional lift | Composite edge + signal construction |
| O6 | **Near Support (BUY) / Near Resistance (SELL)** | PF 3.11 near support; +9.6% winner edge | Composite edge + winner/loser |
| O7 | **Midday session (11:00–14:00 IST)** | Top composite trait | Composite edge |
| O8 | **Strong displacement** | +5.6% winner edge vs medium | Winner/loser comparison |
| O9 | **CHOCH→BOS slow (90–240 min)** | +14.4% winner edge | Winner/loser comparison |
| O10 | **Level strength = Strong** (score ≥ 50) | 22.8% bounce / 19.7% rejection rate | Major level strength |
| O11 | **Confirmation candle score ≥ 50** | Trigger quality alignment | Confirmation candle research |
| O12 | **Trigger model match** (Section 4/5 patterns) | Matrix probability boost | Trigger validation |

---

## 7. Confidence Score Formula

V1 uses a **0–100 Institutional Confidence Score** combining three research-backed components:

```
Confidence = clamp(0, 100,
    0.50 × QualityScore
  + 0.25 × ConfirmationCandleScore
  + 0.25 × ContextScore
)
```

### 7.1 Quality Score (0–100) — Base 50%

From `institutional_quality_validation.json`:

| Component | Points | Condition |
|---|---|---|
| Strong Displacement | 20 | Displacement strength = Strong |
| CHOCH→BOS Timing | 20 | 90–240 minutes (inclusive lower, exclusive upper at 240) |
| FVG Retests | 20 | Exactly 1 FVG retest before BOS |
| FVG Freshness | 20 | FVG age 6–15 bars inclusive |
| Swing Distance | 20 | Distance from swing high/low < 20 points |

**Production threshold:** ≥ 70 recommended for deployment priority (PF 4.3 vs 2.63 baseline).

### 7.2 Confirmation Candle Score (0–100) — 25%

From `institutional_confirmation_candle.json` (top-20% vs bottom-20% magnitude cohort):

Weighted sum of matched characteristics ÷ sum of available positive lifts, scaled to 100.

**BUY boosters:** Gap Down (+18.6%), Gap Up (+10.5%), RSI >70 (+9.0%), Lower Wick Sweep (+6.2%), Sell-side sweep (+10.0%).  
**SELL boosters:** Level Weak (+14.4%), Gap Up (+12.1%), Buy-side sweep (+8.4%).

### 7.3 Context Score (0–100) — 25%

| Factor | BUY | SELL | Weight |
|---|---|---|---|
| Regime match | Liquidity Reversal / Trend Continuation | Same | 30 |
| Market location | Near Support | Near Resistance | 25 |
| RSI band | < 40 | > 70 | 20 |
| Level strength | Strong / Moderate | Strong / Moderate | 15 |
| Trigger model matched | Yes (matrix entry) | Yes | 10 |

### 7.4 Confidence Bands

| Band | Range | Action |
|---|---|---|
| **Reject** | 0–19 | Do not trade (quality bucket 0–20 = 0% WR) |
| **Low** | 20–49 | Log only / paper trade |
| **Standard** | 50–69 | Tier-2 production eligible |
| **High** | 70–84 | Priority deployment |
| **Institutional** | 85–100 | Full size + Tier-1 upgrade if present |

---

## 8. Entry Calculation

### V1 Standard Entry (Production Default)

**Method:** BOS Close Entry  
**Price:** Close of the BOS confirmation candle  
**Source:** `tier2_entry_optimization.json` — 502/502 triggers, 0 missed moves, expectancy 102.48

```
Entry_BUY  = Close(BOS_bullish_bar)
Entry_SELL = Close(BOS_bearish_bar)
```

### V1 Premium Entry (Sequence-Optimized Alternative)

**Method:** Displacement Close  
**Source:** `sequence_entry_timing_validation.json` — best composite score (123.5), WR 66.67%, PF 4.19

```
Entry = Close(Displacement_bar)
```

Use when Tier-1 sweep sequence is present and Quality Score ≥ 70.

### Entry Timing Rule

- Enter at **BOS bar close** (standard) or **displacement bar close** (premium)  
- Trigger-to-expansion distance: **1 bar** median (trigger validation)  
- Do **not** use FVG 50% or liquidity retest as primary entry (lower expectancy, high missed-move rate)

---

## 9. Stop Loss Calculation

**Model:** Structural Swing SL (unanimous across entry, exit, and trade construction research)

```
BUY:  SL = min(Low) over swing window before BOS, minus buffer
SELL: SL = max(High) over swing window before BOS, plus buffer

Buffer = max(0.15 × ATR(14), 5 points)
Risk  = |Entry − SL|
```

**Validation:** Best accuracy pairing with BOS Close across 502 signals. Alternative ATR SL rejected for V1 (lower net profit in trade construction matrix).

**Max risk cap:** Reject signal if `Risk > 2.0 × ATR(14)` or `Risk > symbol-specific max points`.

---

## 10. Target Calculation

RR reachability baseline (`tier2_exit_optimization.json`): **46% hit 1R · 24% hit 2R · 10% hit 3R**.

### Target-1 (T1) — 1R Partial

```
T1 = Entry + Risk × 1.0   (BUY)
T1 = Entry − Risk × 1.0   (SELL)
```

**Action:** Exit **50%** of position at T1 (exit model C/E foundation).

### Target-2 (T2) — 2R Partial

```
T2 = Entry + Risk × 2.0   (BUY)
T2 = Entry − Risk × 2.0   (SELL)
```

**Action:** Exit **25%** of remaining at T2 (33% model D alternative).

### Target-3 (T3) — Opposite Liquidity / Structure Trail

**Primary price target:**

```
T3 = Nearest opposite-side liquidity pool
     (Buy-side liquidity for SELL, Sell-side liquidity for BUY)
```

**Source:** `trade_construction_validation.json` — Opposite Liquidity Pool: net 17,580 pts (best net profit).

**After T1 hit:** Move SL to **breakeven**, then **trail last 25%** using swing structure (Exit Model E — best profit: net 16,996 pts, PF 1.51, expectancy 33.86).

```
Trail_SL_BUY  = last confirmed Higher Low after 1R
Trail_SL_SELL = last confirmed Lower High after 1R
```

---

## 11. Signal Invalidation Conditions

A signal is **invalid** (never entered) if any condition is true at evaluation time:

| ID | Condition | Source |
|---|---|---|
| I1 | Quality Score < 20 | Quality validation (0% WR bucket) |
| I2 | Structural SL cannot be placed or risk = 0 | Trade construction |
| I3 | Tier-2 sequence incomplete or out of order | Tier-2 definition |
| I4 | **Weak level** + trading **into** level (BUY into resistance / SELL into support with Strong level expecting bounce/rejection > 20%) | Major level strength |
| I5 | BOS candle is counter-direction | Tier-2 definition |
| I6 | FVG not reclaimed at BOS bar | Tier-2 definition |
| I7 | Trigger matches **false trigger model** with >70% false-trigger rate and move expectancy < 150 pts | Trigger validation |
| I8 | Regime = Session Breakout **and** Quality Score < 50 | Regime classification (rank 5, PF 1.54) |

---

## 12. Signal Cancellation Conditions

A signal is **cancelled** (was valid, now withdrawn) if any condition occurs **before entry fill**:

| ID | Condition |
|---|---|
| C1 | Price closes back through BOS level in opposite direction |
| C2 | Opposing CHOCH prints before entry |
| C3 | FVG reclaim fails (price fully invalidates FVG) |
| C4 | New swing forms that expands SL beyond max risk cap |
| C5 | Liquidity sweep reversed (e.g., sell-side sweep then immediate buy-side sweep against direction) |
| C6 | Signal age > 3 bars on 5M / 2 bars on 15M / 1 bar on 1H without entry trigger |
| C7 | Market halt / session close within 15 minutes |

After entry: manage via SL/T1/T2/T3 — cancellation rules no longer apply; exit rules take over.

---

## 13. Confidence Increase Conditions

| Condition | Boost | Source |
|---|---|---|
| Quality Score ≥ 70 | +15 to +25 effective | Quality validation |
| Tier-1 full sweep sequence | +10 | Tiered framework |
| Regime = Liquidity Reversal | +15 | Regime classification |
| Regime = Trend Continuation | +10 | Regime classification |
| RSI < 40 (BUY) / > 70 (SELL) | +8 | Composite edge |
| Near Support (BUY) / Near Resistance (SELL) | +8 | Winner/loser |
| Strong displacement | +6 | Winner/loser |
| CHOCH→BOS slow (90–240 min) | +8 | Winner/loser |
| Both-side + directional sweep | +10 | Signal construction |
| Bullish/Bearish divergence aligned | +8 | Signal construction |
| Trigger matrix match (100% prob bucket) | +12 | Trigger validation |
| Level Retest x2–3 + failed break | +10 | Trigger validation |
| Lower/Upper wick sweep >60% + engulfing | +8 | Confirmation candle |
| 4-trait composite: RSI<40 + Midday + Strong Disp + Slow CHOCH-BOS | Tier upgrade to Institutional band | Composite edge (PF 13.82, WR 63%) |

---

## 14. Confidence Decrease Conditions

| Condition | Penalty | Source |
|---|---|---|
| Quality Score 20–40 | −10 | Quality validation |
| No liquidity sweep (when expecting Tier-1 quality) | −8 | Signal construction (−33% lift for no sweep) |
| Near Resistance (BUY) / Near Support (SELL) | −12 | Winner/loser (−16.8% edge near resistance for winners) |
| MI Score 65–79 | −10 | Winner/loser (more common in losers) |
| CHOCH→BOS fast (< 30 min) | −8 | Winner/loser |
| CHOCH→BOS moderate (30–90 min) | −6 | Winner/loser |
| Medium-only displacement (no strong) | −5 | Winner/loser |
| Narrative confidence < 50 | −8 | Winner/loser |
| Regime = Session Breakout | −10 | Regime classification |
| Discount zone on large-magnitude BUY expectation mismatch | −5 | Signal construction (magnitude cohort) |
| False trigger pattern match | −15 | Trigger validation |
| Confirmation: weak body + close mid + no volume expansion | −6 | Confirmation candle |

---

## 15. Regime Classification (Production Tags)

Priority order (`tier2_regime_classification.json`):

1. Session Breakout  
2. HTF Reversal  
3. Liquidity Reversal  
4. Range Expansion  
5. Trend Continuation (default when LTF aligns)

| Regime | V1 Action | Key Stats |
|---|---|---|
| **Liquidity Reversal** | Highest confidence boost | WR 80%, PF 8.54, Exp 447 |
| **Trend Continuation** | Recommended production focus | WR 44%, PF 3.62, Exp 107 |
| **Range Expansion** | Standard | WR 39%, PF 3.17, Exp 108 |
| **HTF Reversal** | Standard (most frequent) | WR 43%, PF 2.67, Exp 96 |
| **Session Breakout** | Confidence penalty | WR 30%, PF 1.54, Exp 67 |

---

## 16. Level & Liquidity Context Rules

From support/resistance + major level strength research:

| Level Tier | Bounce | Rejection | V1 Use |
|---|---|---|---|
| **Strong** (≥ 50) | 22.8% | 19.7% | Best for bounce/rejection trades |
| **Moderate** (30–49) | 16.6% | 14.3% | Default institutional zone |
| **Weak** (< 30) | 1.3% | 2.6% | Breakout/breakdown bias (97%+ break rate) |

**Fresh/Retested levels** (~1–2 tests) dominate break outcomes. Prefer **1–2 level tests** before trigger; exhaustion (3+ failed breaks) increases reversal probability.

Liquidity sweep standalone trades: **not V1 primary** (overall sweep tradeability negative expectancy −6.45). Sweeps are **sequence context**, not standalone entries.

---

## 17. V1 Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET DATA (5M / 15M / 1H)                  │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER-2 SEQUENCE DETECTOR                                       │
│  Displacement → CHOCH → BOS → FVG Reclaim                       │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  MANDATORY GATES (M1–M7)                                        │
│  Sequence ✓ · SL valid ✓ · Score ≥ 20 ✓ · No invalidation ✓    │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONTEXT ENRICHMENT                                             │
│  Level · Liquidity · Regime · Trigger · Confirmation Candle     │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONFIDENCE SCORE (0–100)                                       │
│  50% Quality + 25% Candle + 25% Context ± boosters/penalties    │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
   Score ≥ 50                                  Score < 50
   PRODUCTION SIGNAL                           LOG / SKIP
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTION                                                      │
│  Entry: BOS Close (standard) | Displacement Close (premium)     │
│  SL:    Structural Swing                                        │
│  T1:    1R (50% exit) → move SL to breakeven                    │
│  T2:    2R (25% exit)                                           │
│  T3:    Opposite liquidity + structure trail (25% remainder)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 18. BUY / SELL Summary Cards

### BUY Signal = 

```
Tier-2 Bullish Sequence (Disp → CHOCH → BOS → FVG Reclaim)
+ Structural SL valid
+ Quality Score ≥ 20
+ [Optional] Sell-side/Both-side sweep
+ [Optional] Near Support · RSI < 40 · Strong Displacement
+ [Optional] Trigger: Level Retest + Failed Breakdown + Lower Wick Sweep + Engulfing
→ Entry @ BOS Close
→ SL @ Structural Swing Low − buffer
→ T1 @ 1R | T2 @ 2R | T3 @ Opposite Liquidity + trail
```

### SELL Signal = 

```
Tier-2 Bearish Sequence (Disp → CHOCH → BOS → FVG Reclaim)
+ Structural SL valid
+ Quality Score ≥ 20
+ [Optional] Buy-side/Both-side sweep
+ [Optional] Near Resistance · RSI > 70 · Premium Zone
+ [Optional] Trigger: Level Retest + Failed Breakout + Upper Wick Sweep + Engulfing
→ Entry @ BOS Close
→ SL @ Structural Swing High + buffer
→ T1 @ 1R | T2 @ 2R | T3 @ Opposite Liquidity + trail
```

---

## 19. Implementation Notes

1. **Research only was used to author this spec.** No production code changes are implied by this document.  
2. **Tier-2 remains the frequency anchor** (~42 signals/month NIFTY50). Quality ≥ 70 reduces to ~43 signals/year but with PF 4.3 — use as sizing tier, not hard gate, unless frequency reduction is acceptable.  
3. **Sequence entry (Displacement Close)** and **Tier-2 entry (BOS Close)** serve different trade-offs: higher accuracy vs full coverage. V1 standardizes on BOS Close; premium tier uses Displacement Close.  
4. **Exit Model E** (trail after 1R) is the validated production exit; fixed 1R-only exit underperforms (expectancy 8.99 vs 33.86).  
5. All thresholds reference NIFTY50 365-day research (2025-07-03 → 2026-07-03). Re-validate quarterly.

---

## 20. Document Control

| Field | Value |
|---|---|
| Document | SmartMoneyEngine_V1_Production_Specification.md |
| Version | 1.0 |
| Status | Frozen |
| Next Review | After live paper-trading calibration |
| Owner | SmartMoneyEngine Research |

---

*This specification synthesizes completed research exports. It does not constitute trading advice.*
