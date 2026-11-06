"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from owui_minimax_image_proxy.config import Settings
from owui_minimax_image_proxy.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Settings with a fixed test upstream so the env is irrelevant."""
    return Settings(
        minimax_api_key="test-key",
        minimax_api_base_url="http://upstream.test/v1",
        image_gen_timeout_secs=5.0,
    )


@pytest.fixture
def make_client():  # type: ignore[no-untyped-def]
    """Build a TestClient wired to a `MockTransport`-backed httpx client.

    Usage::

        def test_x(make_client):
            def handler(request):
                return httpx.Response(200, json={...})
            with make_client(handler) as client:
                r = client.post("/images/generations", json={...})
                assert r.status_code == 200

    Optional kwargs are forwarded to `Settings`.
    """
    upstreams: list[httpx.AsyncClient] = []

    def _factory(
        handler: Callable[[httpx.Request], httpx.Response],
        **settings_overrides: Any,
    ) -> TestClient:
        transport = httpx.MockTransport(handler)
        defaults: dict[str, Any] = {
            "minimax_api_key": "test-key",
            "minimax_api_base_url": "http://upstream.test/v1",
            "image_gen_timeout_secs": 5.0,
        }
        cfg = Settings(**(defaults | settings_overrides))
        upstream = httpx.AsyncClient(
            transport=transport,
            base_url=cfg.minimax_api_base_url,
            headers={
                "Authorization": f"Bearer {cfg.minimax_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        upstreams.append(upstream)
        app = create_app(settings=cfg, http_client=upstream)
        return TestClient(app)

    yield _factory

    # Cleanup: close any upstream clients we created. TestClient closes
    # the FastAPI lifespan, which leaves the client alone (owns_client=False).
    # httpx.AsyncClient only exposes async `aclose()`, so we drive it
    # via a fresh event loop in the fixture teardown. MockTransport
    # doesn't hold sockets, but we close anyway to silence resource
    # warnings.
    for u in upstreams:
        if not u.is_closed:
            asyncio.run(u.aclose())


def make_minimax_response(
    *,
    images: list[str] | None = None,
    status_code: int = 0,
    status_msg: str = "success",
) -> dict[str, Any]:
    """Build a MiniMax native `/image_generation` response body.

    `images` defaults to one placeholder base64 string per requested image.
    """
    if images is None:
        images = ["aGVsbG8="]  # "hello"
    return {
        "id": "test-id",
        "data": {"image_base64": images},
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }
