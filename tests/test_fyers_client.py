"""Unit tests for FYERS connectivity client (mocked network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.brokers import fyers_client as fc


def _write_env(path: Path, **values: str) -> Path:
    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_credentials_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        FYERS_APP_ID="APP-100",
        FYERS_SECRET_KEY="secret",
        FYERS_REDIRECT_URI="http://127.0.0.1:8000",
        FYERS_ACCESS_TOKEN="tok123",
    )
    for name in (
        "FYERS_APP_ID",
        "FYERS_SECRET_KEY",
        "FYERS_REDIRECT_URI",
        "FYERS_ACCESS_TOKEN",
        "FYERS_CLIENT_ID",
        "FYERS_PIN",
    ):
        monkeypatch.delenv(name, raising=False)

    creds = fc.load_credentials(env_path)
    assert creds.app_id == "APP-100"
    assert creds.secret_key == "secret"
    assert creds.redirect_uri == "http://127.0.0.1:8000"
    assert creds.access_token == "tok123"


def test_load_credentials_client_id_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        FYERS_CLIENT_ID="CLIENT-100",
        FYERS_SECRET_KEY="secret",
        FYERS_REDIRECT_URI="http://127.0.0.1:8000",
    )
    for name in (
        "FYERS_APP_ID",
        "FYERS_SECRET_KEY",
        "FYERS_REDIRECT_URI",
        "FYERS_ACCESS_TOKEN",
        "FYERS_CLIENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    creds = fc.load_credentials(env_path)
    assert creds.app_id == "CLIENT-100"


def test_load_credentials_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        FYERS_APP_ID="APP-100",
        FYERS_REDIRECT_URI="http://127.0.0.1:8000",
    )
    for name in (
        "FYERS_APP_ID",
        "FYERS_SECRET_KEY",
        "FYERS_REDIRECT_URI",
        "FYERS_ACCESS_TOKEN",
        "FYERS_CLIENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(fc.FyersCredentialError, match="FYERS_SECRET_KEY"):
        fc.load_credentials(env_path)


def test_load_dotenv_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"
    with pytest.raises(FileNotFoundError):
        fc.load_dotenv_file(missing)


def test_load_access_token_from_env() -> None:
    creds = fc.FyersCredentials(
        app_id="APP-100",
        secret_key="secret",
        redirect_uri="http://127.0.0.1:8000",
        access_token="env-token",
    )
    assert fc.load_access_token(creds) == "env-token"


def test_load_access_token_from_file(tmp_path: Path) -> None:
    token_path = tmp_path / "fyers_token.json"
    token_path.write_text(
        json.dumps({"s": "ok", "access_token": "file-token"}),
        encoding="utf-8",
    )
    creds = fc.FyersCredentials(
        app_id="APP-100",
        secret_key="secret",
        redirect_uri="http://127.0.0.1:8000",
        access_token=None,
    )
    assert fc.load_access_token(creds, token_path=token_path) == "file-token"


def test_load_access_token_missing_raises(tmp_path: Path) -> None:
    creds = fc.FyersCredentials(
        app_id="APP-100",
        secret_key="secret",
        redirect_uri="http://127.0.0.1:8000",
        access_token=None,
    )
    with pytest.raises(fc.FyersTokenError):
        fc.load_access_token(creds, token_path=tmp_path / "nope.json")


def test_format_ws_access_token() -> None:
    assert fc.format_ws_access_token("APP-100", "tok") == "APP-100:tok"
    assert fc.format_ws_access_token("APP-100", "APP-100:tok") == "APP-100:tok"


def test_fyers_client_validate_and_quotes() -> None:
    mock_model = MagicMock()
    mock_model.get_profile.return_value = {
        "s": "ok",
        "data": {"name": "Tester", "fy_id": "XY123"},
    }
    mock_model.quotes.return_value = {
        "s": "ok",
        "d": [{"n": "NSE:NIFTY50-INDEX", "v": {"lp": 25000}}],
    }

    with patch("fyers_apiv3.fyersModel.FyersModel", return_value=mock_model):
        client = fc.FyersClient(app_id="APP-100", access_token="tok")
        profile = client.validate_token()
        quotes = client.get_quotes("NSE:NIFTY50-INDEX")

    assert profile["data"]["name"] == "Tester"
    assert quotes["s"] == "ok"
    mock_model.quotes.assert_called_once_with({"symbols": "NSE:NIFTY50-INDEX"})


def test_fyers_client_api_error() -> None:
    mock_model = MagicMock()
    mock_model.get_profile.return_value = {"s": "error", "message": "invalid token"}

    with patch("fyers_apiv3.fyersModel.FyersModel", return_value=mock_model):
        client = fc.FyersClient(app_id="APP-100", access_token="bad")
        with pytest.raises(fc.FyersAPIError, match="invalid token"):
            client.get_profile()


def test_exchange_auth_code_success() -> None:
    creds = fc.FyersCredentials(
        app_id="APP-100",
        secret_key="secret",
        redirect_uri="http://127.0.0.1:8000",
    )
    mock_session = MagicMock()
    mock_session.generate_token.return_value = {
        "s": "ok",
        "access_token": "new-token",
    }

    with patch("fyers_apiv3.fyersModel.SessionModel", return_value=mock_session):
        result = fc.exchange_auth_code(creds, "auth-code-xyz")

    assert result["access_token"] == "new-token"
    mock_session.set_token.assert_called_once_with("auth-code-xyz")


def test_exchange_auth_code_empty_raises() -> None:
    creds = fc.FyersCredentials(
        app_id="APP-100",
        secret_key="secret",
        redirect_uri="http://127.0.0.1:8000",
    )
    with pytest.raises(fc.FyersAuthError, match="empty"):
        fc.exchange_auth_code(creds, "  ")


def test_save_access_token(tmp_path: Path) -> None:
    dest = tmp_path / "tokens" / "fyers_token.json"
    path = fc.save_access_token({"s": "ok", "access_token": "abc"}, token_path=dest)
    assert path == dest
    payload: dict[str, Any] = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["access_token"] == "abc"


def test_from_env_uses_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        FYERS_APP_ID="APP-100",
        FYERS_SECRET_KEY="secret",
        FYERS_REDIRECT_URI="http://127.0.0.1:8000",
        FYERS_ACCESS_TOKEN="tok-from-env",
    )
    for name in (
        "FYERS_APP_ID",
        "FYERS_SECRET_KEY",
        "FYERS_REDIRECT_URI",
        "FYERS_ACCESS_TOKEN",
        "FYERS_CLIENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    mock_model = MagicMock()
    mock_model.get_profile.return_value = {"s": "ok", "data": {}}

    with patch("fyers_apiv3.fyersModel.FyersModel", return_value=mock_model) as ctor:
        client = fc.FyersClient.from_env(env_path=env_path)
        client.get_profile()

    ctor.assert_called_once()
    assert ctor.call_args.kwargs["client_id"] == "APP-100"
    assert ctor.call_args.kwargs["token"] == "tok-from-env"
