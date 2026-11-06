"""Tests for settings parsing edge cases."""

from __future__ import annotations

import pytest

from owui_minimax_image_proxy.config import Settings


def test_k8s_injected_port_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kubernetes injects PORT=tcp://<svc-ip>:<port> into every pod.

    That value is not a listen port (the Dockerfile CMD pins 8000), so
    it must not crash validation — the default 8000 wins.
    """
    monkeypatch.setenv("PORT", "tcp://10.105.244.2:80")
    cfg = Settings(minimax_api_key="test-key")
    assert cfg.port == 8000


def test_image_proxy_port_alias_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_PROXY_PORT", "9000")
    cfg = Settings(minimax_api_key="test-key")
    assert cfg.port == 9000


def test_explicit_port_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "tcp://10.0.0.1:80")
    cfg = Settings(minimax_api_key="test-key", port=8000)
    assert cfg.port == 8000


def test_missing_api_key_fails_fast() -> None:
    """MINIMAX_API_KEY is required — empty/whitespace must raise at startup."""
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        Settings(minimax_api_key="")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        Settings(minimax_api_key="   ")


def test_base_url_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_BASE_URL", "https://api.minimax.io/v1/")
    cfg = Settings(minimax_api_key="test-key")
    assert cfg.minimax_api_base_url == "https://api.minimax.io/v1"


def test_allowed_models_parsed_from_comma_list() -> None:
    cfg = Settings(minimax_api_key="test-key")
    assert cfg.allowed_models == frozenset({"image-01", "image-01-live"})

    cfg2 = Settings(minimax_api_key="test-key", minimax_allowed_models=" image-01 , custom-model ")
    assert cfg2.allowed_models == frozenset({"image-01", "custom-model"})


def test_defaults() -> None:
    cfg = Settings(minimax_api_key="test-key")
    assert cfg.minimax_api_base_url == "https://api.minimax.io/v1"
    assert cfg.image_gen_timeout_secs == 120.0
    assert cfg.max_prompt_chars == 1500
    assert cfg.port == 8000
