"""Dataset health score — advisory only; never modifies data."""

from __future__ import annotations

from typing import Any


def score_health(validation_report: dict[str, Any]) -> dict[str, Any]:
    """
    Compute Dataset Health Score from validation issues.

    100 → Ready for Replay
    95  → Minor warnings
    <90 → Replay not recommended
    """
    issues = validation_report.get("issues") or []
    critical = sum(1 for i in issues if i.get("severity") == "critical")
    warnings = sum(1 for i in issues if i.get("severity") == "warning")

    score = 100
    # Each critical finding type cluster penalizes heavily; cap floor at 0.
    score -= min(70, critical * 5)
    score -= min(20, warnings * 2)
    score = max(0, score)

    if score >= 100 and critical == 0 and warnings == 0:
        verdict = "Ready for Replay"
        band = "READY"
    elif score >= 95:
        verdict = "Minor warnings"
        band = "WARN"
    else:
        verdict = "Replay not recommended"
        band = "BLOCK"

    return {
        "health_score": score,
        "verdict": verdict,
        "band": band,
        "critical_issues": critical,
        "warning_issues": warnings,
        "thresholds": {
            "ready": 100,
            "minor_warnings": 95,
            "not_recommended_below": 90,
        },
    }
