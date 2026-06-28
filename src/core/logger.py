import logging
from pathlib import Path

# Create Logs Directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log File
LOG_FILE = LOG_DIR / "engine.log"

# Logger Configuration
logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

    handlers=[

        logging.FileHandler(LOG_FILE),

        logging.StreamHandler()

    ]

)

logger = logging.getLogger("SmartMoneyEngine")