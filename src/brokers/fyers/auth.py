"""
Fyers OAuth2 authentication for SmartMoneyEngine.

Loads broker configuration, completes the browser-based login flow,
exchanges the authorization code for an access token, refreshes expired
tokens automatically, and persists tokens to ``data/tokens/fyers_token.json``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from fyers_apiv3 import fyersModel

from src.brokers.fyers.config import Config, ConfigurationError, load_config
from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "data" / "tokens" / "fyers_token.json"
DEFAULT_CALLBACK_TIMEOUT_SECONDS = 300
REFRESH_TOKEN_URL = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
TOKEN_EXPIRY_BUFFER_SECONDS = 60
OAUTH_STATE = "smartmoneyengine"
OAUTH_RESPONSE_TYPE = "code"
OAUTH_GRANT_TYPE = "authorization_code"
REFRESH_GRANT_TYPE = "refresh_token"

_REFRESH_LOCK = threading.Lock()

SUCCESS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>SmartMoneyEngine</title></head>
<body>
  <h2>Authentication successful</h2>
  <p>You can close this window and return to the terminal.</p>
</body>
</html>
"""

ERROR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>SmartMoneyEngine</title></head>
<body>
  <h2>Authentication failed</h2>
  <p>Return to the terminal for details.</p>
</body>
</html>
"""


class AuthenticationError(Exception):
    """Base exception for Fyers authentication failures."""


class BrowserOpenError(AuthenticationError):
    """Raised when the login URL cannot be opened in a browser."""


class CallbackTimeoutError(AuthenticationError):
    """Raised when the OAuth callback is not received before timeout."""


class CallbackNotReceivedError(AuthenticationError):
    """Raised when the callback request does not contain an authorization code."""


class InvalidAuthCodeError(AuthenticationError):
    """Raised when the authorization code is missing or rejected."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when Fyers rejects the application credentials."""


class NetworkError(AuthenticationError):
    """Raised when token exchange fails due to network issues."""


class FyersAPIError(AuthenticationError):
    """Raised when the Fyers API returns an error response."""

    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response


@dataclass
class CallbackResult:
    """OAuth callback payload captured by the local HTTP server."""

    auth_code: str | None = None
    error_message: str | None = None


def normalize_raw_access_token(access_token: str) -> str:
    """Strip optional ``app_id:`` prefix; return raw JWT / token body."""
    token = access_token.strip()
    if ":" in token:
        _, _, token = token.partition(":")
        token = token.strip()
    if not token:
        raise AuthenticationError("Access token is empty after normalization.")
    return token


def format_ws_access_token(app_id: str, access_token: str) -> str:
    """Build FYERS websocket token ``{app_id}:{raw_jwt}``."""
    if not app_id or not app_id.strip():
        raise AuthenticationError("FYERS app_id is required for websocket token formatting.")
    raw = normalize_raw_access_token(access_token)
    return f"{app_id.strip()}:{raw}"


def load_token_file(token_path: Path | None = None) -> dict[str, Any]:
    """Load the persisted FYERS token JSON bundle."""
    destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH
    if not destination.exists():
        raise AuthenticationError(f"Fyers token file not found: {destination}")

    try:
        with destination.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AuthenticationError(f"Invalid JSON in token file: {destination}") from exc

    if not isinstance(payload, dict):
        raise AuthenticationError("Token file must contain a JSON object.")

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthenticationError(f"'access_token' missing or empty in {destination}")

    if payload.get("s") not in (None, "ok"):
        raise AuthenticationError(
            f"Token file indicates unsuccessful authentication: {destination}"
        )

    return payload


def access_token_expiry(access_token: str) -> int | None:
    """Return JWT ``exp`` epoch seconds for a raw or prefixed access token."""
    raw = normalize_raw_access_token(access_token)
    parts = raw.split(".")
    if len(parts) < 2:
        return None
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    exp = payload.get("exp")
    return int(exp) if isinstance(exp, (int, float)) else None


def is_access_token_valid(
    access_token: str,
    *,
    buffer_seconds: int = TOKEN_EXPIRY_BUFFER_SECONDS,
) -> bool:
    """Return True when JWT ``exp`` is still in the future (with buffer)."""
    exp = access_token_expiry(access_token)
    if exp is None:
        return False
    return exp - int(time.time()) > buffer_seconds


def _resolve_pin(pin: str | None = None) -> str | None:
    if pin is not None and str(pin).strip():
        return str(pin).strip()
    env_pin = os.getenv("FYERS_PIN")
    if env_pin and env_pin.strip():
        return env_pin.strip()
    return None


def _app_id_hash(config: Config) -> str:
    return hashlib.sha256(f"{config.app_id}:{config.secret_key}".encode()).hexdigest()


def _create_session(config: Config) -> fyersModel.SessionModel:
    """
    Build a Fyers ``SessionModel`` from validated configuration.

    Parameters
    ----------
    config : Config
        Loaded Fyers configuration.

    Returns
    -------
    fyersModel.SessionModel
        Session object for OAuth operations.
    """
    return fyersModel.SessionModel(
        client_id=config.app_id,
        secret_key=config.secret_key,
        redirect_uri=config.redirect_uri,
        response_type=OAUTH_RESPONSE_TYPE,
        grant_type=OAUTH_GRANT_TYPE,
        state=OAUTH_STATE,
    )


def generate_login_url(config: Config) -> str:
    """
    Generate the Fyers OAuth login URL.

    Parameters
    ----------
    config : Config
        Loaded Fyers configuration.

    Returns
    -------
    str
        Browser login URL.
    """
    session = _create_session(config)
    return session.generate_authcode()


def _parse_redirect_target(redirect_uri: str) -> tuple[str, int]:
    """
    Extract host and port from the configured redirect URI.

    Parameters
    ----------
    redirect_uri : str
        Registered OAuth redirect URI.

    Returns
    -------
    tuple[str, int]
        Hostname and port for the callback server.

    Raises
    ------
    AuthenticationError
        If the redirect URI is invalid.
    """
    parsed = urlparse(redirect_uri)

    if parsed.scheme not in {"http", "https"}:
        raise AuthenticationError(
            f"Unsupported redirect URI scheme: {parsed.scheme!r}. "
            "Use http or https for local OAuth callbacks."
        )

    host = parsed.hostname
    if not host:
        raise AuthenticationError(f"Invalid redirect URI: {redirect_uri}")

    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80

    return host, port


def start_callback_server(
    redirect_uri: str,
    timeout_seconds: int = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
) -> str:
    """
    Start a temporary HTTP server and wait for the OAuth callback.

    Parameters
    ----------
    redirect_uri : str
        Registered redirect URI used to derive host and port.
    timeout_seconds : int, optional
        Maximum seconds to wait for callback, by default 300.

    Returns
    -------
    str
        Authorization code from the callback query string.

    Raises
    ------
    CallbackTimeoutError
        If no callback arrives before timeout.
    CallbackNotReceivedError
        If callback arrives without an authorization code.
    AuthenticationError
        If the callback server cannot start.
    FyersAPIError
        If Fyers returns an error in the callback query string.
    """
    host, port = _parse_redirect_target(redirect_uri)
    result = CallbackResult()
    completion_event = threading.Event()

    logger.info("Starting OAuth callback server on %s:%s", host, port)

    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        """Handle a single Fyers OAuth redirect."""

        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)

            if "auth_code" in query and query["auth_code"][0].strip():
                result.auth_code = query["auth_code"][0].strip()
            elif "authorization_code" in query and query["authorization_code"][0].strip():
                result.auth_code = query["authorization_code"][0].strip()
            elif query.get("s", [""])[0] == "error":
                message = query.get("message", query.get("msg", ["Unknown OAuth error"]))[0]
                result.error_message = message

            if result.auth_code or result.error_message:
                completion_event.set()

            status = 200 if result.auth_code else 400
            body = SUCCESS_HTML if result.auth_code else ERROR_HTML
            encoded = body.encode("utf-8")

            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("OAuth callback server: " + format, *args)

    try:
        server = HTTPServer((host, port), OAuthCallbackHandler)
    except OSError as exc:
        raise AuthenticationError(
            f"Unable to start callback server on {host}:{port}: {exc}"
        ) from exc

    server.timeout = 1.0

    def serve_until_complete() -> None:
        while not completion_event.is_set():
            server.handle_request()

    worker = threading.Thread(target=serve_until_complete, daemon=True)
    worker.start()

    if not completion_event.wait(timeout=timeout_seconds):
        server.server_close()
        raise CallbackTimeoutError(
            f"OAuth callback not received within {timeout_seconds} seconds."
        )

    server.server_close()
    worker.join(timeout=2)

    if result.error_message:
        raise FyersAPIError(
            f"Fyers OAuth callback returned an error: {result.error_message}"
        )

    if not result.auth_code:
        raise CallbackNotReceivedError(
            "OAuth callback received without an authorization code."
        )

    logger.info("OAuth authorization code received.")
    return result.auth_code


def _validate_token_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the token exchange response from Fyers.

    Parameters
    ----------
    response : dict[str, Any]
        Raw API response.

    Returns
    -------
    dict[str, Any]
        Validated token response.

    Raises
    ------
    InvalidAuthCodeError
        If the authorization code is invalid or expired.
    InvalidCredentialsError
        If application credentials are rejected.
    FyersAPIError
        For other API-level failures.
    """
    if not isinstance(response, dict):
        raise FyersAPIError("Unexpected token response format.", None)

    status = response.get("s")
    message = str(response.get("message", "Unknown Fyers API error"))
    code = response.get("code")

    if status == "ok" and response.get("access_token"):
        return response

    lowered = message.lower()

    if "auth" in lowered and "code" in lowered:
        raise InvalidAuthCodeError(message)

    if any(
        keyword in lowered
        for keyword in ("secret", "credential", "appid", "app id", "invalid client")
    ):
        raise InvalidCredentialsError(message)

    if code in {-108, -102, -99}:
        raise FyersAPIError(message, response)

    raise FyersAPIError(message, response)


def exchange_token(config: Config, auth_code: str) -> dict[str, Any]:
    """
    Exchange an authorization code for a Fyers access token.

    Parameters
    ----------
    config : Config
        Loaded Fyers configuration.
    auth_code : str
        Authorization code from the OAuth callback.

    Returns
    -------
    dict[str, Any]
        Validated token response from Fyers.

    Raises
    ------
    InvalidAuthCodeError
        If the authorization code is blank or rejected.
    InvalidCredentialsError
        If credentials are invalid.
    NetworkError
        If the HTTP request fails.
    FyersAPIError
        For other API errors.
    """
    if not auth_code or not auth_code.strip():
        raise InvalidAuthCodeError("Authorization code is empty.")

    session = _create_session(config)
    session.set_token(auth_code.strip())

    logger.info("Exchanging authorization code for access token.")
    try:
        response = session.generate_token()
    except requests.RequestException as exc:
        raise NetworkError(
            f"Network error while exchanging authorization code: {exc}"
        ) from exc

    return _validate_token_response(response)


def refresh_access_token(
    config: Config,
    refresh_token: str,
    pin: str,
) -> dict[str, Any]:
    """
    Refresh an expired access token using FYERS ``validate-refresh-token``.

    Parameters
    ----------
    config : Config
        Loaded Fyers configuration.
    refresh_token : str
        Persisted refresh token.
    pin : str
        FYERS account PIN.

    Returns
    -------
    dict[str, Any]
        Validated token response from Fyers.
    """
    if not refresh_token or not refresh_token.strip():
        raise AuthenticationError("Refresh token is empty.")
    if not pin or not pin.strip():
        raise AuthenticationError(
            "FYERS_PIN is required for refresh-token authentication."
        )

    payload = {
        "grant_type": REFRESH_GRANT_TYPE,
        "appIdHash": _app_id_hash(config),
        "refresh_token": refresh_token.strip(),
        "pin": pin.strip(),
    }
    logger.info("Refreshing FYERS access token via validate-refresh-token.")
    try:
        response = requests.post(
            REFRESH_TOKEN_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        token_data = response.json()
    except requests.RequestException as exc:
        raise NetworkError(f"Network error while refreshing access token: {exc}") from exc

    return _validate_token_response(token_data)


def save_token(
    token_data: dict[str, Any],
    token_path: Path | None = None,
) -> Path:
    """
    Persist token response JSON to disk.

    Parameters
    ----------
    token_data : dict[str, Any]
        Token payload returned by Fyers.
    token_path : Path | None, optional
        Output file path. Defaults to ``data/tokens/fyers_token.json``.

    Returns
    -------
    Path
        Path where the token file was written.
    """
    destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(token_data, handle, indent=2)
        handle.write("\n")

    logger.info("Fyers token saved to %s", destination)
    return destination


def _merge_refreshed_token(
    refreshed: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(previous)
    merged.update(refreshed)
    if not merged.get("refresh_token"):
        merged["refresh_token"] = previous.get("refresh_token")
    return merged


def ensure_valid_access_token(
    token_path: Path | None = None,
    *,
    pin: str | None = None,
    allow_interactive_oauth: bool = True,
    force_refresh: bool = False,
) -> str:
    """
    Return a valid access token, refreshing or re-authenticating when needed.

    Recovery order:
    1. Use current access token if JWT ``exp`` is valid.
    2. Refresh with ``refresh_token`` + ``FYERS_PIN``.
    3. Run interactive OAuth via ``authenticate()`` when allowed.
    """
    destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH

    with _REFRESH_LOCK:
        config = load_config()

        if not destination.exists():
            logger.warning("Token file missing at %s", destination)
            if not allow_interactive_oauth:
                raise AuthenticationError(
                    f"Token file not found: {destination}. Run OAuth login first."
                )
            token_data = authenticate(token_path=destination)
            return normalize_raw_access_token(str(token_data["access_token"]))

        bundle = load_token_file(destination)
        access_token = normalize_raw_access_token(str(bundle["access_token"]))

        if not force_refresh and is_access_token_valid(access_token):
            logger.info("FYERS access token is valid.")
            return access_token

        logger.warning("FYERS access token expired or near expiry.")
        refresh_value = bundle.get("refresh_token")
        pin_value = _resolve_pin(pin)

        if isinstance(refresh_value, str) and refresh_value.strip() and pin_value:
            try:
                refreshed = refresh_access_token(config, refresh_value, pin_value)
                merged = _merge_refreshed_token(refreshed, bundle)
                save_token(merged, token_path=destination)
                new_access = normalize_raw_access_token(str(merged["access_token"]))
                logger.info("FYERS access token refreshed successfully.")
                return new_access
            except AuthenticationError as exc:
                logger.warning("FYERS refresh-token flow failed: %s", exc)
        else:
            logger.warning(
                "Refresh unavailable (refresh_token=%s, FYERS_PIN=%s).",
                "set" if refresh_value else "missing",
                "set" if pin_value else "missing",
            )

        if not allow_interactive_oauth:
            raise AuthenticationError(
                "Access token expired and automatic recovery failed. "
                "Set FYERS_PIN and ensure refresh_token exists, or run OAuth login."
            )

        logger.info("Launching interactive FYERS OAuth login.")
        token_data = authenticate(token_path=destination)
        return normalize_raw_access_token(str(token_data["access_token"]))


def _open_browser(login_url: str) -> None:
    """
    Open the Fyers login URL in the default browser.

    Parameters
    ----------
    login_url : str
        OAuth login URL.

    Raises
    ------
    BrowserOpenError
        If the browser could not be opened automatically.
    """
    opened = webbrowser.open(login_url, new=1, autoraise=True)
    if not opened:
        raise BrowserOpenError(
            "Unable to open the default browser automatically. "
            "Open the login URL manually."
        )


def authenticate(
    token_path: Path | None = None,
    timeout_seconds: int = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Run the complete Fyers OAuth authentication flow.

    Parameters
    ----------
    token_path : Path | None, optional
        Destination for the saved token JSON.
    timeout_seconds : int, optional
        Callback wait timeout in seconds.

    Returns
    -------
    dict[str, Any]
        Token response returned by Fyers.

    Raises
    ------
    ConfigurationError
        If required configuration is missing.
    AuthenticationError
        If any authentication step fails.
    """
    print("Loading config...")
    logger.info("Loading Fyers configuration.")
    config = load_config()

    print("Generating login URL...")
    logger.info("Generating Fyers OAuth login URL.")
    login_url = generate_login_url(config)
    print(login_url)

    print("Opening browser...")
    logger.info("Opening browser for Fyers login.")
    try:
        _open_browser(login_url)
    except BrowserOpenError as exc:
        logger.warning(str(exc))
        print(str(exc))

    print("Waiting for login...")
    logger.info("Waiting for OAuth callback.")
    auth_code = start_callback_server(
        redirect_uri=config.redirect_uri,
        timeout_seconds=timeout_seconds,
    )
    print("Authorization code received.")
    logger.info("Authorization code received.")

    print("Generating access token...")
    token_data = exchange_token(config, auth_code)

    destination = save_token(token_data, token_path=token_path)
    print("Token saved successfully.")
    logger.info("Fyers authentication completed. Token stored at %s", destination)

    return token_data


def main() -> int:
    """
    CLI entry point for Fyers OAuth authentication.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="FYERS authentication utilities")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh-token flow (non-interactive; requires FYERS_PIN)",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Ensure token is valid (refresh or OAuth as needed)",
    )
    args = parser.parse_args()

    try:
        if args.refresh:
            token = ensure_valid_access_token(
                allow_interactive_oauth=False,
                force_refresh=True,
            )
            print(f"Access token refreshed ({token[:16]}...)")
            return 0
        if args.ensure:
            token = ensure_valid_access_token(allow_interactive_oauth=True)
            print(f"Access token ready ({token[:16]}...)")
            return 0
        authenticate()
        return 0
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except AuthenticationError as exc:
        logger.error("Authentication failed: %s", exc)
        print(f"Authentication failed: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        logger.error("Configuration file error: %s", exc)
        print(f"Configuration file error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected authentication failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
