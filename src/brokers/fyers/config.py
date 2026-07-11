"""
Fyers broker configuration loaded from environment variables.

Reads ``.env`` via ``python-dotenv`` and exposes a validated ``Config``
dataclass for use by authentication and API modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

REQUIRED_VARIABLES: tuple[str, ...] = (
    "FYERS_APP_ID",
    "FYERS_SECRET_KEY",
    "FYERS_REDIRECT_URI",
)


class ConfigurationError(Exception):
    """Raised when required Fyers configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """
    Validated Fyers application configuration.

    Attributes
    ----------
    app_id : str
        Fyers application identifier (``FYERS_APP_ID``).
    secret_key : str
        Fyers application secret (``FYERS_SECRET_KEY``).
    redirect_uri : str
        OAuth redirect URI registered with Fyers (``FYERS_REDIRECT_URI``).
    """

    app_id: str
    secret_key: str
    redirect_uri: str


def _resolve_env_path(env_path: Path | None) -> Path:
    """
    Resolve the ``.env`` file path.

    Parameters
    ----------
    env_path : Path | None
        Explicit path override. When ``None``, the project-root ``.env`` is used.

    Returns
    -------
    Path
        Resolved environment file path.
    """
    return env_path if env_path is not None else DEFAULT_ENV_PATH


def _read_required_variables() -> dict[str, str]:
    """
    Read and validate required Fyers environment variables.

    Returns
    -------
    dict[str, str]
        Mapping of variable name to non-empty string value.

    Raises
    ------
    ConfigurationError
        If one or more required variables are missing or blank.
    """
    values: dict[str, str] = {}
    missing: list[str] = []

    for name in REQUIRED_VARIABLES:
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            missing.append(name)
            continue
        values[name] = raw_value.strip()

    if missing:
        missing_list = ", ".join(missing)
        raise ConfigurationError(
            "Missing required Fyers configuration variables: "
            f"{missing_list}. "
            f"Define them in {DEFAULT_ENV_PATH} or the process environment."
        )

    return values


def load_config(env_path: Path | None = None) -> Config:
    """
    Load and validate Fyers configuration from ``.env``.

    Parameters
    ----------
    env_path : Path | None, optional
        Path to an environment file. Defaults to the project-root ``.env``.

    Returns
    -------
    Config
        Validated configuration instance.

    Raises
    ------
    ConfigurationError
        If the environment file is missing required variables.
    FileNotFoundError
        If the resolved ``.env`` path does not exist.
    """
    resolved_path = _resolve_env_path(env_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Environment file not found: {resolved_path}")

    logger.info("Loading Fyers configuration from %s", resolved_path)
    load_dotenv(resolved_path, override=False)

    variables = _read_required_variables()
    config = Config(
        app_id=variables["FYERS_APP_ID"],
        secret_key=variables["FYERS_SECRET_KEY"],
        redirect_uri=variables["FYERS_REDIRECT_URI"],
    )

    logger.info(
        "Fyers configuration loaded successfully (app_id=%s, redirect_uri=%s)",
        config.app_id,
        config.redirect_uri,
    )

    return config
