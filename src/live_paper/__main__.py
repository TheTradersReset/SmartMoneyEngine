"""CLI entry: ``python -m src.live_paper``."""

from __future__ import annotations

import sys

from src.live_paper.service import main


if __name__ == "__main__":
    sys.exit(main())
