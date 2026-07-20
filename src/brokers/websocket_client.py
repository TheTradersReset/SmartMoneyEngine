"""
FYERS v3 websocket connectivity client for SmartMoneyEngine.

Connects to the FYERS data socket, subscribes to ``NSE:NIFTY50-INDEX``,
prints live ticks to the console, and reconnects with exponential backoff
on disconnect. Automatically refreshes expired access tokens before
connect and on websocket auth failures. Connectivity only — no order channel.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from src.brokers.fyers.auth import (
    DEFAULT_TOKEN_PATH,
    AuthenticationError,
    ensure_valid_access_token,
    format_ws_access_token,
)
from src.brokers.fyers.config import ConfigurationError, load_config
from src.brokers.fyers_client import (
    FyersConnectivityError,
    FyersCredentialError,
    FyersTokenError,
)
from src.core.logger import logger

NIFTY50_SYMBOL = "NSE:NIFTY50-INDEX"
DEFAULT_DATA_TYPE = "SymbolUpdate"
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


class WebsocketConnectivityError(FyersConnectivityError):
    """Raised when websocket connectivity fails."""


def validate_subscribe_symbol(symbol: str) -> str:
    """
    Validate and normalize a FYERS subscribe symbol.

    Expected format: ``EXCHANGE:SYMBOL`` (e.g. ``NSE:NIFTY50-INDEX``).
    """
    if not symbol or not str(symbol).strip():
        raise WebsocketConnectivityError("Subscribe symbol must be a non-empty string.")
    normalized = str(symbol).strip().upper()
    if ":" not in normalized:
        raise WebsocketConnectivityError(
            f"Invalid FYERS symbol '{symbol}'. Expected format EXCHANGE:SYMBOL "
            f"(e.g. {NIFTY50_SYMBOL})."
        )
    exchange, _, name = normalized.partition(":")
    if not exchange or not name:
        raise WebsocketConnectivityError(
            f"Invalid FYERS symbol '{symbol}'. Expected format EXCHANGE:SYMBOL "
            f"(e.g. {NIFTY50_SYMBOL})."
        )
    return f"{exchange}:{name}"


def compute_reconnect_backoff(
    attempt: int,
    initial_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
) -> float:
    """
    Compute exponential backoff delay for reconnect attempt ``attempt`` (1-based).
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    delay = initial_seconds * (multiplier ** (attempt - 1))
    return min(delay, max_seconds)


class FyersWebsocketClient:
    """
    FYERS data-websocket connectivity wrapper with reconnect + clean shutdown.

    Parameters
    ----------
    app_id : str
        FYERS application id.
    access_token : str
        Raw access token or ``app_id:token`` formatted string.
    symbols : list[str]
        Symbols to subscribe (default NIFTY50 index).
    data_type : str
        FYERS data type (default ``SymbolUpdate``).
    use_sdk_reconnect : bool
        Pass ``reconnect=True`` to the FYERS SDK socket.
    initial_backoff_seconds : float
        First outer-loop reconnect delay.
    max_backoff_seconds : float
        Cap for outer-loop reconnect delay.
    token_path : Path | None
        Persisted token JSON path for automatic refresh.
    sleep_fn : Callable
        Injectable sleep (for tests).
    socket_factory : Callable | None
        Injectable ``FyersDataSocket`` factory (for tests).
    """

    def __init__(
        self,
        app_id: str,
        access_token: str,
        symbols: list[str] | None = None,
        data_type: str = DEFAULT_DATA_TYPE,
        use_sdk_reconnect: bool = True,
        initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        token_path: Path | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        socket_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not app_id or not app_id.strip():
            raise FyersCredentialError("FYERS app_id is required.")
        if not access_token or not access_token.strip():
            raise FyersTokenError("FYERS access_token is required.")

        raw_symbols = symbols if symbols is not None else [NIFTY50_SYMBOL]
        self.symbols = [validate_subscribe_symbol(s) for s in raw_symbols]
        self.data_type = data_type
        self.app_id = app_id.strip()
        self.ws_access_token = format_ws_access_token(self.app_id, access_token.strip())
        self.use_sdk_reconnect = use_sdk_reconnect
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._token_path = token_path if token_path is not None else DEFAULT_TOKEN_PATH
        self._sleep = sleep_fn
        self._socket_factory = socket_factory

        self._socket: Any | None = None
        self._stop_event = threading.Event()
        self._tick_count = 0
        self.reconnect_attempts = 0
        self._needs_token_refresh = False

    @classmethod
    def from_env(
        cls,
        token_path: Path | None = None,
        **kwargs: Any,
    ) -> FyersWebsocketClient:
        """Build a websocket client from ``.env`` config + persisted token file."""
        destination = token_path if token_path is not None else DEFAULT_TOKEN_PATH
        try:
            config = load_config()
        except ConfigurationError as exc:
            raise FyersCredentialError(str(exc)) from exc

        try:
            access_token = ensure_valid_access_token(
                token_path=destination,
                allow_interactive_oauth=True,
            )
        except AuthenticationError as exc:
            raise FyersTokenError(str(exc)) from exc

        return cls(
            app_id=config.app_id,
            access_token=access_token,
            token_path=destination,
            **kwargs,
        )

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def update_access_token(self, access_token: str) -> None:
        """Apply a refreshed raw access token to this client."""
        self.ws_access_token = format_ws_access_token(self.app_id, access_token)

    def _ensure_fresh_token(self) -> None:
        """Refresh the access token when expired or after a websocket auth failure."""
        try:
            raw = ensure_valid_access_token(
                token_path=self._token_path,
                allow_interactive_oauth=True,
                force_refresh=self._needs_token_refresh,
            )
            self._needs_token_refresh = False
            self.update_access_token(raw)
        except AuthenticationError as exc:
            raise FyersTokenError(str(exc)) from exc

    def _reset_socket(self) -> None:
        """Close any existing socket and clear SDK singleton state for reconnect."""
        socket = self._socket
        if socket is not None:
            close = getattr(socket, "close_connection", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 — best-effort reset
                    logger.warning("Error while closing FYERS websocket: %s", exc)
        self._socket = None

        try:
            from fyers_apiv3.FyersWebsocket import data_ws

            data_ws.FyersDataSocket._instance = None  # type: ignore[attr-defined]
        except Exception:
            pass

    def request_stop(self) -> None:
        """Signal a clean shutdown (safe from signal handlers)."""
        logger.info("Shutdown requested for FYERS websocket client.")
        self._stop_event.set()
        self._reset_socket()

    def on_message(self, message: Any) -> None:
        """Handle an incoming tick / message and print it to the console."""
        self._tick_count += 1
        print(f"TICK #{self._tick_count}: {message}", flush=True)
        logger.info("FYERS tick #%s: %s", self._tick_count, message)

    def on_error(self, message: Any) -> None:
        logger.error("FYERS websocket error: %s", message)
        print(f"WS ERROR: {message}", file=sys.stderr, flush=True)
        text = str(message).lower()
        if "token is expired" in text or "token expired" in text:
            logger.warning("FYERS websocket token expired; scheduling token refresh.")
            self._needs_token_refresh = True
            self._reset_socket()

    def on_close(self, message: Any) -> None:
        logger.warning("FYERS websocket closed: %s", message)
        print(f"WS CLOSED: {message}", flush=True)

    def on_open(self) -> None:
        """Subscribe to configured symbols when the socket opens."""
        socket = self._socket
        if socket is None:
            raise WebsocketConnectivityError("Cannot subscribe: socket is not connected.")
        logger.info(
            "FYERS websocket open; subscribing symbols=%s data_type=%s",
            self.symbols,
            self.data_type,
        )
        socket.subscribe(symbols=self.symbols, data_type=self.data_type)
        print(f"Subscribed: {', '.join(self.symbols)} ({self.data_type})", flush=True)

    def _create_socket(self) -> Any:
        factory = self._socket_factory
        if factory is None:
            from fyers_apiv3.FyersWebsocket import data_ws

            factory = data_ws.FyersDataSocket

        socket = factory(
            access_token=self.ws_access_token,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=self.use_sdk_reconnect,
            on_connect=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        self._socket = socket
        return socket

    def connect_once(self) -> Any:
        """Create the socket and establish a single connection."""
        logger.info("Connecting FYERS data websocket.")
        socket = self._create_socket()
        socket.connect()
        logger.info("FYERS data websocket connect() returned.")
        return socket

    def run(self, max_reconnect_attempts: int | None = None) -> None:
        """
        Connect, keep receiving ticks, and reconnect with backoff on failure.

        Parameters
        ----------
        max_reconnect_attempts : int | None
            Cap on outer reconnect attempts. ``None`` means unlimited until stop.
        """
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            self.reconnect_attempts = attempt
            try:
                self._ensure_fresh_token()
                self._reset_socket()
                socket = self.connect_once()
                keep_running = getattr(socket, "keep_running", None)
                if callable(keep_running):
                    keep_running()
                else:
                    # Fallback idle loop when keep_running is unavailable (tests).
                    while not self._stop_event.is_set():
                        self._sleep(0.25)
                # Normal return from keep_running — reconnect unless stopping.
                if self._stop_event.is_set():
                    break
                raise WebsocketConnectivityError("WebSocket session ended unexpectedly.")
            except Exception as exc:
                if self._stop_event.is_set():
                    logger.info("Stopping after websocket error during shutdown: %s", exc)
                    break
                if max_reconnect_attempts is not None and attempt >= max_reconnect_attempts:
                    logger.error(
                        "FYERS websocket failed after %s attempt(s): %s",
                        attempt,
                        exc,
                    )
                    raise WebsocketConnectivityError(
                        f"WebSocket failed after {attempt} attempt(s): {exc}"
                    ) from exc

                delay = compute_reconnect_backoff(
                    attempt,
                    initial_seconds=self.initial_backoff_seconds,
                    max_seconds=self.max_backoff_seconds,
                )
                logger.warning(
                    "FYERS websocket disconnected (attempt %s): %s. Reconnecting in %.1fs.",
                    attempt,
                    exc,
                    delay,
                )
                print(
                    f"Disconnected (attempt {attempt}): {exc}. Reconnecting in {delay:.1f}s...",
                    flush=True,
                )
                self._sleep(delay)

        logger.info("FYERS websocket client stopped (ticks=%s).", self._tick_count)
        print(f"Websocket stopped. Total ticks: {self._tick_count}", flush=True)


def _install_signal_handlers(client: FyersWebsocketClient) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}; shutting down...", flush=True)
        client.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Signal handlers may be unavailable on some threads / platforms.
            pass


def main() -> int:
    """CLI entry: stream live NIFTY50 ticks until Ctrl+C."""
    client: FyersWebsocketClient | None = None
    try:
        logger.info("Starting FYERS websocket connectivity test.")
        client = FyersWebsocketClient.from_env()
        _install_signal_handlers(client)
        print(f"Connecting FYERS websocket for {', '.join(client.symbols)} ...", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        client.run()
        return 0
    except KeyboardInterrupt:
        if client is not None:
            client.request_stop()
        print("Interrupted.", flush=True)
        return 0
    except FileNotFoundError as exc:
        logger.error("Configuration file error: %s", exc)
        print(f"Configuration file error: {exc}", file=sys.stderr)
        return 1
    except FyersConnectivityError as exc:
        logger.error("FYERS websocket error: %s", exc)
        print(f"FYERS websocket error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected FYERS websocket failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
