"""NSE session constants and holiday calendar for dataset verification."""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
BAR_MINUTES = 5
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
# Last regular 5m bucket start in cash session.
LAST_BAR_START = time(15, 25)

# Expected bars per full trading day for 5-minute NSE cash session.
EXPECTED_BARS_PER_SESSION = 75

# Known NSE holidays (extend as needed). Weekends handled separately.
NSE_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2025
        date(2025, 2, 26),  # Mahashivratri
        date(2025, 3, 14),  # Holi
        date(2025, 3, 31),  # Id-Ul-Fitr (approx / declared)
        date(2025, 4, 10),  # Mahavir Jayanti
        date(2025, 4, 14),  # Dr Ambedkar Jayanti / Good Friday cluster
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 1),  # Maharashtra Day
        date(2025, 8, 15),  # Independence Day
        date(2025, 8, 27),  # Ganesh Chaturthi
        date(2025, 10, 2),  # Gandhi Jayanti
        date(2025, 10, 21),  # Diwali Laxmi Pujan (muhurat day may vary)
        date(2025, 10, 22),  # Diwali-Balipratipada
        date(2025, 11, 5),  # Guru Nanak Jayanti
        date(2025, 12, 25),  # Christmas
        # 2026
        date(2026, 1, 26),  # Republic Day
        date(2026, 3, 3),  # Holi
        date(2026, 3, 26),  # Ram Navami / Id cluster (verify annually)
        date(2026, 3, 31),  # Ramzan Id (verify annually)
        date(2026, 4, 3),  # Good Friday
        date(2026, 4, 14),  # Dr Ambedkar Jayanti
        date(2026, 5, 1),  # Maharashtra Day
        date(2026, 8, 15),  # Independence Day
        date(2026, 10, 2),  # Gandhi Jayanti
        date(2026, 10, 20),  # Diwali (verify annually)
        date(2026, 11, 24),  # Guru Nanak Jayanti (verify annually)
        date(2026, 12, 25),  # Christmas
    }
)

OUTLIER_RETURN_PCT = 3.0  # absolute close-to-close move threshold (%)
