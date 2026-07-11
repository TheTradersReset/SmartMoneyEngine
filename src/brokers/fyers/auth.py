"""
Fyers OAuth2 authentication for SmartMoneyEngine.

Loads broker configuration, completes the browser-based login flow,
exchanges the authorization code for an access token, and persists the
token response to ``data/tokens/fyers_token.json``.
"""

from __future__ import annotations

import json
import sys
import threading
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
OAUTH_STATE = "smartmoneyengine"
OAUTH_RESPONSE_TYPE = "code"
OAUTH_GRANT_TYPE = "authorization_code"

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
    try:
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
