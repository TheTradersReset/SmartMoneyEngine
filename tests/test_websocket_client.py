"""Unit tests for FYERS websocket connectivity client (mocked network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.brokers import websocket_client as ws
from src.brokers.fyers_client import FyersCredentialError, FyersTokenError


def test_validate_subscribe_symbol_ok() -> None:
    assert ws.validate_subscribe_symbol("nse:nifty50-index") == "NSE:NIFTY50-INDEX"
    assert ws.validate_subscribe_symbol("NSE:NIFTY50-INDEX") == "NSE:NIFTY50-INDEX"


def test_validate_subscribe_symbol_rejects_bad_format() -> None:
    with pytest.raises(ws.WebsocketConnectivityError, match="EXCHANGE:SYMBOL"):
        ws.validate_subscribe_symbol("NIFTY50")
    with pytest.raises(ws.WebsocketConnectivityError):
        ws.validate_subscribe_symbol("")
    with pytest.raises(ws.WebsocketConnectivityError):
        ws.validate_subscribe_symbol("NSE:")


def test_compute_reconnect_backoff() -> None:
    assert ws.compute_reconnect_backoff(1, initial_seconds=1.0, max_seconds=60.0) == 1.0
    assert ws.compute_reconnect_backoff(2, initial_seconds=1.0, max_seconds=60.0) == 2.0
    assert ws.compute_reconnect_backoff(3, initial_seconds=1.0, max_seconds=60.0) == 4.0
    assert ws.compute_reconnect_backoff(10, initial_seconds=1.0, max_seconds=30.0) == 30.0
    with pytest.raises(ValueError):
        ws.compute_reconnect_backoff(0)


def test_client_rejects_missing_credentials() -> None:
    with pytest.raises(FyersCredentialError):
        ws.FyersWebsocketClient(app_id="", access_token="tok")
    with pytest.raises(FyersTokenError):
        ws.FyersWebsocketClient(app_id="APP-100", access_token="")


def test_default_symbol_is_nifty50() -> None:
    client = ws.FyersWebsocketClient(app_id="APP-100", access_token="tok")
    assert client.symbols == ["NSE:NIFTY50-INDEX"]
    assert client.ws_access_token == "APP-100:tok"


def test_on_message_prints_ticks(capsys: pytest.CaptureFixture[str]) -> None:
    client = ws.FyersWebsocketClient(app_id="APP-100", access_token="tok")
    client.on_message({"symbol": "NSE:NIFTY50-INDEX", "ltp": 25001.5})
    captured = capsys.readouterr()
    assert "TICK #1" in captured.out
    assert "25001.5" in captured.out
    assert client.tick_count == 1


def test_on_open_subscribes() -> None:
    mock_socket = MagicMock()
    client = ws.FyersWebsocketClient(app_id="APP-100", access_token="tok")
    client._socket = mock_socket
    client.on_open()
    mock_socket.subscribe.assert_called_once_with(
        symbols=["NSE:NIFTY50-INDEX"],
        data_type="SymbolUpdate",
    )


def test_connect_once_uses_factory() -> None:
    created: dict[str, Any] = {}

    def factory(**kwargs: Any) -> MagicMock:
        created["kwargs"] = kwargs
        socket = MagicMock()
        created["socket"] = socket
        return socket

    client = ws.FyersWebsocketClient(
        app_id="APP-100",
        access_token="tok",
        socket_factory=factory,
    )
    socket = client.connect_once()
    assert socket is created["socket"]
    created["socket"].connect.assert_called_once()
    assert created["kwargs"]["access_token"] == "APP-100:tok"
    assert created["kwargs"]["reconnect"] is True


def test_run_reconnects_with_backoff() -> None:
    sleeps: list[float] = []
    connect_calls = {"n": 0}

    class FlakySocket:
        def connect(self) -> None:
            connect_calls["n"] += 1
            if connect_calls["n"] < 3:
                raise RuntimeError(f"boom-{connect_calls['n']}")

        def keep_running(self) -> None:
            # Succeed on third attempt, then stop the client.
            client.request_stop()

        def subscribe(self, symbols: list[str], data_type: str) -> None:
            return None

        def close_connection(self) -> None:
            return None

    def factory(**_kwargs: Any) -> FlakySocket:
        return FlakySocket()

    client = ws.FyersWebsocketClient(
        app_id="APP-100",
        access_token="tok",
        socket_factory=factory,
        sleep_fn=lambda s: sleeps.append(s),
        initial_backoff_seconds=1.0,
        max_backoff_seconds=10.0,
    )
    client.run(max_reconnect_attempts=5)

    assert connect_calls["n"] == 3
    assert sleeps == [1.0, 2.0]
    assert client.reconnect_attempts == 3


def test_run_raises_after_max_attempts() -> None:
    def factory(**_kwargs: Any) -> MagicMock:
        socket = MagicMock()
        socket.connect.side_effect = RuntimeError("always-fail")
        return socket

    client = ws.FyersWebsocketClient(
        app_id="APP-100",
        access_token="tok",
        socket_factory=factory,
        sleep_fn=lambda _s: None,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.01,
    )
    with pytest.raises(ws.WebsocketConnectivityError, match="after 2 attempt"):
        client.run(max_reconnect_attempts=2)


def test_request_stop_closes_socket() -> None:
    mock_socket = MagicMock()
    client = ws.FyersWebsocketClient(app_id="APP-100", access_token="tok")
    client._socket = mock_socket
    client.request_stop()
    mock_socket.close_connection.assert_called_once()
    assert client._stop_event.is_set()
