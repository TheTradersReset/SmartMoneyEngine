"""
FYERS API v3 connectivity client for SmartMoneyEngine.

Loads credentials from ``.env``, authenticates / validates an access token,
and exposes lightweight profile / quotes helpers. Connectivity only — no
order placement.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "data" / "tokens" / "fyers_token.json"

REQUIRED_APP_VARS: tuple[str, ...] = (
    "FYERS_APP_ID",
    "FYERS_SECRET_KEY",
    "FYERS_REDIRECT_URI",
)

DEFAULT_QUOTE_SYMBOL = "NSE:NIFTY50-INDEX"
OAUTH_RESPONSE_TYPE = "code"
OAUTH_GRANT_TYPE = "authorization_code"
OAUTH_STATE = "smartmoneyengine"


class FyersConnectivityError(Exception):
    """Base exception for FYERS connectivity failures."""


class FyersCredentialError(FyersConnectivityError):
    """Raised when required credentials are missing or blank."""


class FyersTokenError(FyersConnectivityError):
    """Raised when an access token is missing or invalid."""


class FyersAuthError(FyersConnectivityError):
    """Raised when OAuth / token exchange fails."""


class FyersAPIError(FyersConnectivityError):
    """Raised when the FYERS API returns an error response."""

    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response


@dataclass(frozen=True)
class FyersCredentials:
    """Validated FYERS application credentials loaded from the environment."""

    app_id: str
    secret_key: str
    redirect_uri: str
    access_token: str | None = None
    pin: str | None = None


def load_dotenv_file(env_path: Path | None = None) -> Path:
    """
    Load environment variables from a ``.env`` file.

    Parameters
    ----------
    env_path : Path | None
        Explicit ``.env`` path. Defaults to the project-root ``.env``.

    Returns
    -------
    Path
        Resolved path that was loaded (or attempted).

    Raises
    ------
    FileNotFoundError
        If the resolved ``.env`` file does not exist.
    """
    resolved = env_path if env_path is not None else DEFAULT_ENV_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Environment file not found: {resolved}")
    load_dotenv(resolved, override=False)
    logger.info("Loaded environment from %s", resolved)
    return resolved


def _read_required(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise FyersCredentialError(
            f"Missing required FYERS credential: {name}. "
            f"Define it in {DEFAULT_ENV_PATH} or the process environment."
        )
    return value.strip()


def load_credentials(env_path: Path | None = None) -> FyersCredentials:
    """
    Load and validate FYERS credentials from ``.env``.

    Required: ``FYERS_APP_ID``, ``FYERS_SECRET_KEY``, ``FYERS_REDIRECT_URI``.
    Optional: ``FYERS_ACCESS_TOKEN``, ``FYERS_PIN``, ``FYERS_CLIENT_ID``
    (alias for app id when ``FYERS_APP_ID`` is unset).
    """
    load_dotenv_file(env_path)

    # Allow FYERS_CLIENT_ID as an alias for FYERS_APP_ID.
    if not (os.getenv("FYERS_APP_ID") or "").strip():
        client_id = (os.getenv("FYERS_CLIENT_ID") or "").strip()
        if client_id:
            os.environ["FYERS_APP_ID"] = client_id

    app_id = _read_required("FYERS_APP_ID")
    secret_key = _read_required("FYERS_SECRET_KEY")
    redirect_uri = _read_required("FYERS_REDIRECT_URI")

    access_token_raw = os.getenv("FYERS_ACCESS_TOKEN")
    access_token = access_token_raw.strip() if access_token_raw and access_token_raw.strip() else None
    pin_raw = os.getenv("FYERS_PIN")
    pin = pin_raw.strip() if pin_raw and pin_raw.strip() else None

    logger.info(
        "FYERS credentials loaded (app_id=%s, redirect_uri=%s, access_token=%s)",
        app_id,
        redirect_uri,
        "set" if access_token else "unset",
    )
    return FyersCredentials(
        app_id=app_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        access_token=access_token,
        pin=pin,
    )


def load_access_token(
    credentials: FyersCredentials | None = None,
    token_path: Path | None = None,
) -> str:
    """
    Resolve an access token from env credentials or ``fyers_token.json``.

    Preference order:
    1. ``FYERS_ACCESS_TOKEN`` on ``credentials``
    2. ``access_token`` field in the token JSON file
    """
    if credentials is not None and credentials.access_token:
        logger.info("Using FYERS_ACCESS_TOKEN from environment.")
        return credentials.access_token

    destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH
    if not destination.exists():
        raise FyersTokenError(
            "No FYERS access token found. Set FYERS_ACCESS_TOKEN in .env "
            f"or create {destination} via OAuth (see src/brokers/README.md)."
        )

    logger.info("Loading FYERS access token from %s", destination)
    try:
        with destination.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise FyersTokenError(f"Invalid JSON in token file: {destination}") from exc

    if not isinstance(payload, dict):
        raise FyersTokenError("Token file must contain a JSON object.")

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise FyersTokenError(f"'access_token' missing or empty in {destination}")

    return access_token.strip()


def format_ws_access_token(app_id: str, access_token: str) -> str:
    """
    Build the FYERS websocket access token string ``app_id:access_token``.

    If ``access_token`` already contains a colon (pre-formatted), it is returned
    unchanged.
    """
    token = access_token.strip()
    if ":" in token:
        return token
    return f"{app_id.strip()}:{token}"


def generate_login_url(credentials: FyersCredentials) -> str:
    """Generate the FYERS v3 OAuth login URL."""
    from fyers_apiv3 import fyersModel

    session = fyersModel.SessionModel(
        client_id=credentials.app_id,
        secret_key=credentials.secret_key,
        redirect_uri=credentials.redirect_uri,
        response_type=OAUTH_RESPONSE_TYPE,
        grant_type=OAUTH_GRANT_TYPE,
        state=OAUTH_STATE,
    )
    return session.generate_authcode()


def exchange_auth_code(credentials: FyersCredentials, auth_code: str) -> dict[str, Any]:
    """
    Exchange an OAuth authorization code for an access token (FYERS v3).

    Parameters
    ----------
    credentials : FyersCredentials
        App credentials.
    auth_code : str
        Authorization code from the redirect callback.

    Returns
    -------
    dict[str, Any]
        Token response from FYERS.
    """
    from fyers_apiv3 import fyersModel

    if not auth_code or not auth_code.strip():
        raise FyersAuthError("Authorization code is empty.")

    session = fyersModel.SessionModel(
        client_id=credentials.app_id,
        secret_key=credentials.secret_key,
        redirect_uri=credentials.redirect_uri,
        response_type=OAUTH_RESPONSE_TYPE,
        grant_type=OAUTH_GRANT_TYPE,
        state=OAUTH_STATE,
    )
    session.set_token(auth_code.strip())
    logger.info("Exchanging authorization code for access token.")
    response = session.generate_token()

    if not isinstance(response, dict):
        raise FyersAuthError("Unexpected token response type from FYERS.")

    if response.get("s") not in (None, "ok") and "access_token" not in response:
        message = str(response.get("message", "Token exchange failed."))
        raise FyersAuthError(message)

    if "access_token" not in response or not response["access_token"]:
        raise FyersAuthError("Token response missing access_token.")

    return response


def save_access_token(
    token_data: dict[str, Any],
    token_path: Path | None = None,
) -> Path:
    """Persist a FYERS token response to disk."""
    destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(token_data, handle, indent=2)
        handle.write("\n")
    logger.info("FYERS token saved to %s", destination)
    return destination


def _validate_api_response(response: Any, action: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise FyersAPIError(f"{action} returned an unexpected response type.", None)
    if response.get("s") == "ok":
        return response
    message = str(response.get("message", f"{action} failed."))
    raise FyersAPIError(message, response)


class FyersClient:
    """
    Thin FYERS v3 REST connectivity wrapper (profile / quotes only).

    Parameters
    ----------
    app_id : str
        FYERS application id (``FYERS_APP_ID``).
    access_token : str
        Valid access token (raw token, not ``app_id:token``).
    log_level : str
        SDK internal log level.
    """

    def __init__(
        self,
        app_id: str,
        access_token: str,
        log_level: str = "ERROR",
    ) -> None:
        if not app_id or not app_id.strip():
            raise FyersCredentialError("FYERS app_id is required.")
        if not access_token or not access_token.strip():
            raise FyersTokenError("FYERS access_token is required.")

        # Strip app_id prefix if the caller passed a WS-formatted token.
        token = access_token.strip()
        if ":" in token:
            _, _, token = token.partition(":")
            token = token.strip()

        self.app_id = app_id.strip()
        self.access_token = token

        from fyers_apiv3 import fyersModel

        logger.info("Initializing FYERS FyersModel (connectivity client).")
        self._model = fyersModel.FyersModel(
            client_id=self.app_id,
            token=self.access_token,
            is_async=False,
            log_level=log_level,
        )
        logger.info("FYERS FyersModel initialized.")

    @classmethod
    def from_env(
        cls,
        env_path: Path | None = None,
        token_path: Path | None = None,
        **kwargs: Any,
    ) -> FyersClient:
        """Create a client from ``.env`` credentials and access token."""
        credentials = load_credentials(env_path)
        access_token = load_access_token(credentials, token_path=token_path)
        return cls(app_id=credentials.app_id, access_token=access_token, **kwargs)

    def get_profile(self) -> dict[str, Any]:
        """Fetch the authenticated user profile (token validation)."""
        logger.info("Calling FYERS get_profile.")
        response = self._model.get_profile()
        return _validate_api_response(response, "get_profile")

    def get_quotes(self, symbol: str = DEFAULT_QUOTE_SYMBOL) -> dict[str, Any]:
        """Fetch quotes for a symbol (default ``NSE:NIFTY50-INDEX``)."""
        if not symbol or not symbol.strip():
            raise FyersConnectivityError("Symbol is required for get_quotes.")
        symbol = symbol.strip()
        logger.info("Calling FYERS quotes for %s", symbol)
        response = self._model.quotes({"symbols": symbol})
        return _validate_api_response(response, f"get_quotes({symbol})")

    def validate_token(self) -> dict[str, Any]:
        """Validate the access token by calling ``get_profile``."""
        profile = self.get_profile()
        logger.info("FYERS access token validated successfully.")
        return profile


def main() -> int:
    """CLI: load credentials, validate token, print profile + NIFTY50 quote."""
    try:
        logger.info("Starting FYERS connectivity check.")
        credentials = load_credentials()
        client = FyersClient.from_env()
        profile = client.validate_token()
        profile_data = profile.get("data", {}) if isinstance(profile.get("data"), dict) else {}
        name = (
            profile_data.get("name")
            or profile_data.get("display_name")
            or "Unknown"
        )
        fy_id = profile_data.get("fy_id") or profile_data.get("client_id") or "Unknown"

        print("FYERS connected successfully")
        print(f"App ID: {credentials.app_id}")
        print(f"Client Name: {name}")
        print(f"Client ID: {fy_id}")

        try:
            quotes = client.get_quotes(DEFAULT_QUOTE_SYMBOL)
            print(f"Quotes ({DEFAULT_QUOTE_SYMBOL}): {quotes}")
        except FyersAPIError as exc:
            logger.warning("Quotes check failed (market may be closed): %s", exc)
            print(f"Quotes check skipped/failed: {exc}")

        logger.info("FYERS connectivity check completed.")
        return 0
    except FileNotFoundError as exc:
        logger.error("Configuration file error: %s", exc)
        print(f"Configuration file error: {exc}", file=sys.stderr)
        return 1
    except FyersConnectivityError as exc:
        logger.error("FYERS connectivity error: %s", exc)
        print(f"FYERS connectivity error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected FYERS connectivity failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
