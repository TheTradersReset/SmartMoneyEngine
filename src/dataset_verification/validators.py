"""Read-only validation checks for historical OHLCV datasets."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.dataset_verification.calendar import (
    BAR_MINUTES,
    EXPECTED_BARS_PER_SESSION,
    IST,
    LAST_BAR_START,
    NSE_HOLIDAYS,
    OUTLIER_RETURN_PCT,
    SESSION_CLOSE,
    SESSION_OPEN,
)


def _issue(check: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"check": check, "severity": severity, "message": message}
    payload.update(extra)
    return payload


def validate_dataset(
    frame: pd.DataFrame,
    *,
    symbol: str = "UNKNOWN",
    resolution: str = "5",
    source: str | None = None,
) -> dict[str, Any]:
    """
    Run all verification checks. Never mutates input data.

    Returns a structured report with per-check results, issues, fingerprints, and coverage.
    """
    working = frame.copy()
    issues: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    if working.empty:
        empty_report = {
            "meta": {
                "symbol": symbol,
                "resolution": resolution,
                "source": source,
                "bar_count": 0,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "checks": {"empty_dataset": {"status": "FAIL", "detail": "No bars loaded"}},
            "issues": [_issue("empty_dataset", "critical", "Dataset contains zero bars")],
            "coverage": {},
            "integrity": {
                "checksum": None,
                "dataset_hash": None,
                "dataset_fingerprint": None,
            },
        }
        return empty_report

    timestamps: pd.Series = working["timestamp"]

    # 1. Timestamp continuity / 12 unexpected gaps / 2 missing (session-aware)
    expected_delta = timedelta(minutes=BAR_MINUTES)
    continuity_gaps: list[dict[str, Any]] = []
    for prev, curr in zip(timestamps.iloc[:-1], timestamps.iloc[1:]):
        delta = curr - prev
        if delta == expected_delta:
            continue
        # Session boundary / overnight is expected when crossing days or lunch is N/A for NSE cash (continuous)
        if prev.date() != curr.date():
            continuity_gaps.append(
                {
                    "from": prev.isoformat(),
                    "to": curr.isoformat(),
                    "delta_minutes": delta.total_seconds() / 60.0,
                    "kind": "day_boundary",
                },
            )
            continue
        continuity_gaps.append(
            {
                "from": prev.isoformat(),
                "to": curr.isoformat(),
                "delta_minutes": delta.total_seconds() / 60.0,
                "kind": "intraday_gap",
            },
        )

    intraday_gaps = [g for g in continuity_gaps if g["kind"] == "intraday_gap"]
    checks["timestamp_continuity"] = {
        "status": "PASS" if not intraday_gaps else "FAIL",
        "intraday_gap_count": len(intraday_gaps),
        "day_boundary_count": sum(1 for g in continuity_gaps if g["kind"] == "day_boundary"),
    }
    for gap in intraday_gaps[:50]:
        issues.append(
            _issue(
                "timestamp_continuity",
                "critical",
                f"Unexpected intraday gap {gap['delta_minutes']}m",
                **gap,
            ),
        )
        issues.append(
            _issue(
                "unexpected_time_gaps",
                "critical",
                f"Unexpected time gap of {gap['delta_minutes']} minutes",
                **gap,
            ),
        )

    # 2 & 13. Missing candles + expected bar count (per trading day)
    by_day = working.groupby(timestamps.dt.date, sort=True)
    missing_days: list[dict[str, Any]] = []
    expected_total = 0
    for day, group in by_day:
        count = len(group)
        if day.weekday() >= 5 or day in NSE_HOLIDAYS:
            continue
        expected_total += EXPECTED_BARS_PER_SESSION
        if count < EXPECTED_BARS_PER_SESSION:
            missing = EXPECTED_BARS_PER_SESSION - count
            missing_days.append({"date": day.isoformat(), "bars": count, "missing": missing})
            issues.append(
                _issue(
                    "missing_candles",
                    "critical",
                    f"{missing} missing bars on {day.isoformat()} (have {count}, expected {EXPECTED_BARS_PER_SESSION})",
                    date=day.isoformat(),
                    bars=count,
                    missing=missing,
                ),
            )
        elif count > EXPECTED_BARS_PER_SESSION:
            issues.append(
                _issue(
                    "expected_bar_count",
                    "warning",
                    f"Extra bars on {day.isoformat()}: {count} > {EXPECTED_BARS_PER_SESSION}",
                    date=day.isoformat(),
                    bars=count,
                ),
            )

    checks["missing_candles"] = {
        "status": "PASS" if not missing_days else "FAIL",
        "days_with_missing": len(missing_days),
        "samples": missing_days[:20],
    }
    checks["expected_bar_count"] = {
        "status": "PASS" if not missing_days else "FAIL",
        "expected_bars_per_session": EXPECTED_BARS_PER_SESSION,
        "trading_days_checked": sum(
            1 for day, _ in by_day if day.weekday() < 5 and day not in NSE_HOLIDAYS
        ),
        "expected_total_bars_approx": expected_total,
        "actual_bars": int(len(working)),
    }

    # 3. Duplicate candles
    dup_mask = timestamps.duplicated(keep=False)
    dup_count = int(timestamps.duplicated().sum())
    checks["duplicate_candles"] = {"status": "PASS" if dup_count == 0 else "FAIL", "count": dup_count}
    if dup_count:
        issues.append(_issue("duplicate_candles", "critical", f"{dup_count} duplicate timestamps"))

    # 4. Weekend detection
    weekend_rows = working[timestamps.dt.weekday >= 5]
    checks["weekend_detection"] = {
        "status": "PASS" if weekend_rows.empty else "FAIL",
        "count": int(len(weekend_rows)),
    }
    if not weekend_rows.empty:
        issues.append(
            _issue("weekend_detection", "critical", f"{len(weekend_rows)} bars fall on weekends"),
        )

    # 5. Holiday validation
    holiday_rows = working[timestamps.dt.date.map(lambda d: d in NSE_HOLIDAYS)]
    checks["holiday_validation"] = {
        "status": "PASS" if holiday_rows.empty else "WARNING",
        "count": int(len(holiday_rows)),
        "note": "Bars on known NSE holiday dates (calendar may need annual refresh)",
    }
    if not holiday_rows.empty:
        issues.append(
            _issue(
                "holiday_validation",
                "warning",
                f"{len(holiday_rows)} bars on known NSE holiday dates",
            ),
        )

    # 6. Trading session validation
    clocks = timestamps.dt.time
    outside = working[(clocks < SESSION_OPEN) | (clocks > LAST_BAR_START)]
    checks["trading_session_validation"] = {
        "status": "PASS" if outside.empty else "FAIL",
        "outside_session_count": int(len(outside)),
        "session_open": SESSION_OPEN.isoformat(),
        "last_bar_start": LAST_BAR_START.isoformat(),
        "session_close": SESSION_CLOSE.isoformat(),
    }
    if not outside.empty:
        issues.append(
            _issue(
                "trading_session_validation",
                "critical",
                f"{len(outside)} bars outside NSE cash session bucket window",
            ),
        )

    # 7. Timezone validation
    tz_ok = True
    tz_name = None
    try:
        tz_name = str(timestamps.dt.tz)
        tz_ok = timestamps.dt.tz is not None and str(timestamps.dt.tz) in {"Asia/Kolkata", "IST"}
        # zoneinfo may render as UTC+05:30
        if timestamps.dt.tz is not None:
            offsets = {ts.utcoffset() for ts in timestamps.head(100)}
            tz_ok = offsets == {timedelta(hours=5, minutes=30)}
    except Exception:
        tz_ok = False
    checks["timezone_validation"] = {
        "status": "PASS" if tz_ok else "FAIL",
        "timezone": tz_name,
        "required": "Asia/Kolkata (UTC+05:30)",
    }
    if not tz_ok:
        issues.append(_issue("timezone_validation", "critical", "Timestamps are not IST (UTC+05:30)"))

    # 8. OHLC consistency
    ohlc_bad = (
        (working["high"] < working["open"])
        | (working["high"] < working["close"])
        | (working["low"] > working["open"])
        | (working["low"] > working["close"])
        | (working["high"] < working["low"])
        | working[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    ohlc_count = int(ohlc_bad.sum())
    checks["ohlc_consistency"] = {"status": "PASS" if ohlc_count == 0 else "FAIL", "count": ohlc_count}
    if ohlc_count:
        issues.append(_issue("ohlc_consistency", "critical", f"{ohlc_count} OHLC consistency violations"))

    # 9. Negative prices
    neg_price = (
        (working["open"] <= 0)
        | (working["high"] <= 0)
        | (working["low"] <= 0)
        | (working["close"] <= 0)
    )
    neg_price_n = int(neg_price.sum())
    checks["negative_prices"] = {"status": "PASS" if neg_price_n == 0 else "FAIL", "count": neg_price_n}
    if neg_price_n:
        issues.append(_issue("negative_prices", "critical", f"{neg_price_n} non-positive prices"))

    # 10. Negative volume
    neg_vol = working["volume"] < 0
    neg_vol_n = int(neg_vol.sum())
    checks["negative_volume"] = {"status": "PASS" if neg_vol_n == 0 else "FAIL", "count": neg_vol_n}
    if neg_vol_n:
        issues.append(_issue("negative_volume", "critical", f"{neg_vol_n} negative volume rows"))

    # 11. Outlier detection
    rets = working["close"].pct_change().abs() * 100.0
    outlier_mask = rets > OUTLIER_RETURN_PCT
    outlier_n = int(outlier_mask.fillna(False).sum())
    checks["outlier_detection"] = {
        "status": "PASS" if outlier_n == 0 else "WARNING",
        "count": outlier_n,
        "threshold_pct": OUTLIER_RETURN_PCT,
    }
    if outlier_n:
        issues.append(
            _issue(
                "outlier_detection",
                "warning",
                f"{outlier_n} bars with |return| > {OUTLIER_RETURN_PCT}%",
            ),
        )

    checks["unexpected_time_gaps"] = {
        "status": "PASS" if not intraday_gaps else "FAIL",
        "count": len(intraday_gaps),
        "samples": intraday_gaps[:20],
    }

    # 14–16 checksum / hash / fingerprint
    digest = _content_digest(working)
    fingerprint = _fingerprint(symbol=symbol, resolution=resolution, frame=working, digest=digest)
    checks["dataset_checksum"] = {"status": "PASS", "sha256": digest}
    checks["dataset_hash"] = {"status": "PASS", "sha256": digest}
    checks["dataset_fingerprint"] = {"status": "PASS", "fingerprint": fingerprint}

    # 17 coverage
    day_counts = Counter(timestamps.dt.date.tolist())
    trading_days = [d for d in day_counts if d.weekday() < 5 and d not in NSE_HOLIDAYS]
    coverage = {
        "symbol": symbol,
        "resolution": resolution,
        "bar_count": int(len(working)),
        "first_timestamp": timestamps.iloc[0].isoformat(),
        "last_timestamp": timestamps.iloc[-1].isoformat(),
        "calendar_days": int(timestamps.dt.date.nunique()),
        "trading_days_present": len(trading_days),
        "weekend_bars": int(len(weekend_rows)),
        "holiday_bars": int(len(holiday_rows)),
        "full_session_days": sum(1 for d in trading_days if day_counts[d] == EXPECTED_BARS_PER_SESSION),
        "partial_session_days": sum(1 for d in trading_days if day_counts[d] < EXPECTED_BARS_PER_SESSION),
    }
    checks["coverage_report"] = {"status": "PASS", "coverage": coverage}

    return {
        "meta": {
            "symbol": symbol,
            "resolution": resolution,
            "source": source,
            "bar_count": int(len(working)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "checks": checks,
        "issues": issues,
        "coverage": coverage,
        "integrity": {
            "checksum": digest,
            "dataset_hash": digest,
            "dataset_fingerprint": fingerprint,
        },
    }


def _content_digest(frame: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    for row in frame.itertuples(index=False):
        line = (
            f"{row.timestamp.isoformat()}|{row.open}|{row.high}|{row.low}|"
            f"{row.close}|{row.volume}"
        )
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _fingerprint(*, symbol: str, resolution: str, frame: pd.DataFrame, digest: str) -> str:
    first = frame["timestamp"].iloc[0].isoformat()
    last = frame["timestamp"].iloc[-1].isoformat()
    material = f"{symbol}|{resolution}|{len(frame)}|{first}|{last}|{digest}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
