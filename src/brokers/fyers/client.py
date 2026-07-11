"""
Fyers API client for SmartMoneyEngine.

Loads a persisted access token, initializes ``fyers-apiv3`` ``FyersModel``,
and exposes typed broker API helpers with retry and timeout handling.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests
from fyers_apiv3 import fyersModel

from src.brokers.fyers.config import ConfigurationError, load_config
from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "data" / "tokens" / "fyers_token.json"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0

T = TypeVar("T")


class FyersClientError(Exception):
    """Base exception for Fyers client failures."""


class TokenNotFoundError(FyersClientError):
    """Raised when the token file is missing."""


class TokenInvalidError(FyersClientError):
    """Raised when the token file is invalid or incomplete."""


class FyersNetworkError(FyersClientError):
    """Raised when a network error occurs while calling Fyers."""


class FyersTimeoutError(FyersClientError):
    """Raised when a Fyers API call exceeds the configured timeout."""


class FyersAPIError(FyersClientError):
    """Raised when the Fyers API returns an error response."""

    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response


class FyersClient:
    """
    Production-ready wrapper around ``fyers-apiv3`` ``FyersModel``.

    Parameters
    ----------
    client_id : str
        Fyers application identifier (APP ID).
    access_token : str
        Valid Fyers access token.
    max_retries : int, optional
        Maximum retry attempts per API call.
    timeout_seconds : int, optional
        Per-call timeout in seconds.
    retry_backoff_seconds : float, optional
        Initial backoff between retries; doubled after each failure.
    log_level : str, optional
        Fyers SDK internal log level.
    """

    def __init__(
        self,
        client_id: str,
        access_token: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        log_level: str = "ERROR",
    ) -> None:
        if not client_id or not client_id.strip():
            raise TokenInvalidError("Fyers client_id is required.")
        if not access_token or not access_token.strip():
            raise TokenInvalidError("Fyers access_token is required.")

        self.client_id = client_id.strip()
        self.access_token = access_token.strip()
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.retry_backoff_seconds = retry_backoff_seconds

        logger.info("Initializing FyersModel client.")
        self._model = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            is_async=False,
            log_level=log_level,
        )
        self._configure_request_timeout()
        logger.info("FyersModel client initialized.")

    @classmethod
    def from_token_file(
        cls,
        token_path: Path | None = None,
        **kwargs: Any,
    ) -> FyersClient:
        """
        Create a client using an access token file and application configuration.

        The access token is read exclusively from ``fyers_token.json``.
        The application ID is loaded through ``load_config()``.

        Parameters
        ----------
        token_path : Path | None, optional
            Token JSON path. Defaults to ``data/tokens/fyers_token.json``.
        **kwargs
            Optional overrides forwarded to ``FyersClient.__init__``.

        Returns
        -------
        FyersClient
            Connected client instance.
        """
        access_token = load_access_token(token_path)
        config = load_config()
        return cls(
            client_id=config.app_id,
            access_token=access_token,
            **kwargs,
        )

    def _configure_request_timeout(self) -> None:
        """Apply request timeout defaults to the underlying SDK session."""
        session = getattr(self._model.service, "session", None)
        if session is None:
            return

        original_request = session.request

        def request_with_timeout(method: str, url: str, **request_kwargs: Any) -> Any:
            request_kwargs.setdefault("timeout", self.timeout_seconds)
            return original_request(method, url, **request_kwargs)

        session.request = request_with_timeout  # type: ignore[method-assign]

    def _call_with_retry(self, operation: Callable[[], T], action: str) -> T:
        """
        Execute an SDK operation with retry, timeout, and error translation.

        Parameters
        ----------
        operation : Callable[[], T]
            Callable that performs the SDK request.
        action : str
            Human-readable action name for logging and errors.

        Returns
        -------
        T
            Operation result.

        Raises
        ------
        FyersTimeoutError
            If the operation times out on every attempt.
        FyersNetworkError
            If a network error persists after retries.
        FyersAPIError
            If Fyers returns an error response.
        """
        backoff = self.retry_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            logger.info("Fyers API call: %s (attempt %s/%s)", action, attempt, self.max_retries)
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(operation)
                    result = future.result(timeout=self.timeout_seconds)
                return self._validate_api_response(result, action)
            except FuturesTimeoutError as exc:
                last_error = FyersTimeoutError(
                    f"{action} timed out after {self.timeout_seconds} seconds."
                )
                logger.warning("%s", last_error)
            except requests.RequestException as exc:
                last_error = FyersNetworkError(
                    f"Network error during {action}: {exc}"
                )
                logger.warning("%s", last_error)
            except FyersAPIError:
                raise
            except Exception as exc:
                last_error = FyersClientError(f"Unexpected error during {action}: {exc}")
                logger.warning("%s", last_error)

            if attempt < self.max_retries:
                logger.info("Retrying %s in %.1f seconds.", action, backoff)
                time.sleep(backoff)
                backoff *= 2

        assert last_error is not None
        raise last_error

    @staticmethod
    def _validate_api_response(response: Any, action: str) -> dict[str, Any]:
        """
        Validate a Fyers SDK response dictionary.

        Parameters
        ----------
        response : Any
            Raw SDK response.
        action : str
            Human-readable action name.

        Returns
        -------
        dict[str, Any]
            Validated response dictionary.

        Raises
        ------
        FyersAPIError
            If the response indicates failure.
        """
        if not isinstance(response, dict):
            raise FyersAPIError(
                f"{action} returned an unexpected response type.",
                None,
            )

        if response.get("s") == "ok":
            return response

        message = str(response.get("message", f"{action} failed."))
        raise FyersAPIError(message, response)

    def get_profile(self) -> dict[str, Any]:
        """
        Fetch the authenticated Fyers user profile.

        Returns
        -------
        dict[str, Any]
            Fyers profile response.
        """
        return self._call_with_retry(self._model.get_profile, "get_profile")

    def get_market_status(self) -> dict[str, Any]:
        """
        Fetch current market status from Fyers.

        Returns
        -------
        dict[str, Any]
            Fyers market status response.
        """
        return self._call_with_retry(self._model.market_status, "get_market_status")

    def get_quotes(self, symbol: str) -> dict[str, Any]:
        """
        Fetch quote data for a symbol.

        Parameters
        ----------
        symbol : str
            Fyers symbol, for example ``NSE:NIFTY50-INDEX``.

        Returns
        -------
        dict[str, Any]
            Fyers quotes response.
        """
        if not symbol or not symbol.strip():
            raise FyersClientError("Symbol is required for get_quotes.")

        payload = {"symbols": symbol.strip()}
        return self._call_with_retry(
            lambda: self._model.quotes(payload),
            f"get_quotes({symbol.strip()})",
        )

    def get_history(
        self,
        symbol: str,
        resolution: str,
        date_from: str,
        date_to: str,
        date_format: int = 1,
    ) -> dict[str, Any]:
        """
        Fetch historical candle data.

        Parameters
        ----------
        symbol : str
            Fyers symbol, for example ``NSE:NIFTY50-INDEX``.
        resolution : str
            Candle resolution such as ``1``, ``5``, ``15``, ``60``, or ``1D``.
        date_from : str
            Start date in ``yyyy-mm-dd`` format when ``date_format=1``.
        date_to : str
            End date in ``yyyy-mm-dd`` format when ``date_format=1``.
        date_format : int, optional
            ``1`` for ``yyyy-mm-dd`` dates, ``0`` for epoch values.

        Returns
        -------
        dict[str, Any]
            Fyers history response.
        """
        if not symbol or not symbol.strip():
            raise FyersClientError("Symbol is required for get_history.")
        if not resolution or not str(resolution).strip():
            raise FyersClientError("Resolution is required for get_history.")
        if not date_from or not date_from.strip():
            raise FyersClientError("date_from is required for get_history.")
        if not date_to or not date_to.strip():
            raise FyersClientError("date_to is required for get_history.")

        payload = {
            "symbol": symbol.strip(),
            "resolution": str(resolution).strip(),
            "date_format": date_format,
            "range_from": date_from.strip(),
            "range_to": date_to.strip(),
            "cont_flag": "1",
        }
        return self._call_with_retry(
            lambda: self._model.history(payload),
            f"get_history({symbol.strip()})",
        )


def load_access_token(token_path: Path | None = None) -> str:
    """
    Load and validate a Fyers access token from disk.

    Parameters
    ----------
    token_path : Path | None, optional
        Token JSON path. Defaults to ``data/tokens/fyers_token.json``.

    Returns
    -------
    str
        Valid access token string.

    Raises
    ------
    TokenNotFoundError
        If the token file does not exist.
    TokenInvalidError
        If the token file is invalid or missing ``access_token``.
    """
    destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH

    if not destination.exists():
        raise TokenNotFoundError(f"Fyers token file not found: {destination}")

    logger.info("Loading Fyers access token from %s", destination)

    try:
        with destination.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise TokenInvalidError(
            f"Invalid JSON in Fyers token file: {destination}"
        ) from exc

    if not isinstance(payload, dict):
        raise TokenInvalidError("Fyers token file must contain a JSON object.")

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise TokenInvalidError(
            f"'access_token' is missing or empty in {destination}"
        )

    if payload.get("s") not in (None, "ok"):
        raise TokenInvalidError(
            f"Token file indicates unsuccessful authentication: {destination}"
        )

    logger.info("Fyers access token loaded successfully.")
    return access_token.strip()


def main() -> int:
    """
    CLI entry point.

    Loads the persisted token, connects to Fyers, and prints profile details.

    Returns
    -------
    int
        Process exit code.
    """
    try:
        logger.info("Starting Fyers client connectivity check.")
        client = FyersClient.from_token_file()
        profile = client.get_profile()
        profile_data = profile.get("data", {})

        client_name = profile_data.get("name") or profile_data.get("display_name") or "Unknown"
        client_id = profile_data.get("fy_id") or profile_data.get("client_id") or "Unknown"

        print("Connected successfully")
        print(f"Client Name: {client_name}")
        print(f"Client ID: {client_id}")
        logger.info("Fyers connectivity check completed successfully.")
        return 0
    except (TokenNotFoundError, TokenInvalidError) as exc:
        logger.error("Token error: %s", exc)
        print(f"Token error: {exc}", file=sys.stderr)
        return 1
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        logger.error("Configuration file error: %s", exc)
        print(f"Configuration file error: {exc}", file=sys.stderr)
        return 1
    except FyersClientError as exc:
        logger.error("Fyers client error: %s", exc)
        print(f"Fyers client error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Fyers client failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
