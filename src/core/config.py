from pathlib import Path

# ============================================
# Project Paths
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

LOG_DIR = BASE_DIR / "logs"

OUTPUT_DIR = BASE_DIR / "outputs"

CONFIG_DIR = BASE_DIR / "config"

DOCS_DIR = BASE_DIR / "docs"


# ============================================
# Trading Settings
# ============================================

DEFAULT_TIMEFRAME = "5m"

DEFAULT_SYMBOL = "NIFTY"

DEFAULT_CAPITAL = 50000

RISK_PER_TRADE = 2


# ============================================
# Logger Settings
# ============================================

LOG_LEVEL = "INFO"