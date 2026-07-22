"""Unit tests for Phase-1 pipeline_v2 configuration flags."""

from __future__ import annotations

from pathlib import Path

from src.live_paper.config import LivePaperConfig, load_config


def test_defaults_keep_pipeline_v2_disabled() -> None:
    cfg = LivePaperConfig()
    assert cfg.enable_pipeline_v2 is False
    assert cfg.live_close_queue_max == 32
    assert cfg.shutdown_drain_sec == 5.0


def test_yaml_loads_pipeline_v2_keys(tmp_path: Path, monkeypatch) -> None:
    yaml_path = tmp_path / "live_paper.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "capital_mode: paper",
                "enable_pipeline_v2: false",
                "live_close_queue_max: 16",
                "shutdown_drain_sec: 2.5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LIVE_PAPER_ENABLE_PIPELINE_V2", raising=False)
    monkeypatch.delenv("LIVE_PAPER_LIVE_CLOSE_QUEUE_MAX", raising=False)
    monkeypatch.delenv("LIVE_PAPER_SHUTDOWN_DRAIN_SEC", raising=False)
    monkeypatch.setenv("LIVE_PAPER_CAPITAL_MODE", "paper")
    cfg = load_config(yaml_path=yaml_path, env_path=tmp_path / "missing.env")
    assert cfg.enable_pipeline_v2 is False
    assert cfg.live_close_queue_max == 16
    assert cfg.shutdown_drain_sec == 2.5


def test_env_overrides_pipeline_v2_keys(tmp_path: Path, monkeypatch) -> None:
    yaml_path = tmp_path / "live_paper.yaml"
    yaml_path.write_text("capital_mode: paper\nenable_pipeline_v2: false\n", encoding="utf-8")
    monkeypatch.setenv("LIVE_PAPER_CAPITAL_MODE", "paper")
    monkeypatch.setenv("LIVE_PAPER_ENABLE_PIPELINE_V2", "true")
    monkeypatch.setenv("LIVE_PAPER_LIVE_CLOSE_QUEUE_MAX", "8")
    monkeypatch.setenv("LIVE_PAPER_SHUTDOWN_DRAIN_SEC", "1.0")
    cfg = load_config(yaml_path=yaml_path, env_path=tmp_path / "missing.env")
    assert cfg.enable_pipeline_v2 is True
    assert cfg.live_close_queue_max == 8
    assert cfg.shutdown_drain_sec == 1.0