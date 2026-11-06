"""Integration tests against the live MiniMax API.

These tests are NOT run by `poe test` (the default test task
excludes the `integration` marker). They're invoked by
`poe integration-test`, which:

1. Checks `MINIMAX_API_KEY` is set (skips cleanly with a clear message
   if not — the pytest suite itself also skips via `pytest.skip`).
2. `docker compose up -d --build` — starts this proxy only (upstream
   MiniMax is remote; the key comes from the env).
3. Waits for `GET $PROXY_HEALTH_URL` to return 2xx.
4. Runs `pytest -m integration -v --tb=short`.
5. `docker compose down` — tears the stack down (via a shell `trap`).

The proxy is reachable at http://localhost:8000 (mapped from the
compose port forward). The `PROXY_URL` env var exists so CI /
dockerized runners can override the target.

The key is never committed — pass it via the environment:

    MINIMAX_API_KEY=... poe integration-test

Why a separate test file
------------------------
The unit tests in `test_images_generations.py` / `test_health.py` use
`httpx.MockTransport` to inject canned responses. That's fast and
deterministic, but it doesn't catch:

- A shape drift between what MiniMax returns and what the proxy expects.
- A real auth failure (401) or network error path.
- That `image-01` actually produces a decodable image.

Marking convention
------------------
Every test in this file is decorated with
`@pytest.mark.integration` so the default `poe test` run skips it.
The matching `[tool.pytest.ini_options].markers` entry in
`pyproject.toml` registers the marker so `--strict-markers` is
satisfied.
"""

from __future__ import annotations

import base64
import os

import httpx
import pytest

# Reuse the proxy's expected port forward (set by docker compose).
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8000")

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

pytestmark = pytest.mark.integration

# PNG and JPEG magic headers (JPEG starts with FF D8 FF).
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")


def _is_image(raw: bytes) -> bool:
    return any(raw.startswith(m) for m in IMAGE_MAGIC)


def _client() -> httpx.Client:
    """Plain sync httpx client — no need for async, the proxy is HTTP/1.1."""
    # One 1:1 image takes ~24s; allow generous headroom.
    return httpx.Client(base_url=PROXY_URL, timeout=httpx.Timeout(180.0))


@pytest.fixture(autouse=True)
def _require_key() -> None:
    if not MINIMAX_API_KEY:
        pytest.skip("MINIMAX_API_KEY not set — skipping live MiniMax integration tests")


def test_health_endpoint_reports_ok_against_live_minimax() -> None:
    """`/health` returns 200 + `{"status": "ok"}` against the live API."""
    with _client() as c:
        r = c.get("/health")

    assert r.status_code == 200, f"unexpected status: {r.status_code} body={r.text!r}"
    assert r.json() == {"status": "ok"}


def test_generate_one_1x1_image_end_to_end() -> None:
    """Generate one 1:1 image via the shim; assert 200, one entry, image magic bytes.

    Exercises the full path: POST /images/generations (OWUI dialect) →
    proxy → MiniMax native /image_generation → base64 mapping →
    OWUI-shaped response.
    """
    with _client() as c:
        r = c.post(
            "/images/generations",
            json={
                "model": "image-01",
                "prompt": "a minimal red square on a white background, flat vector style",
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json",
            },
        )

    assert r.status_code == 200, f"unexpected status: {r.status_code} body={r.text!r}"
    body = r.json()
    assert isinstance(body["created"], int) and body["created"] > 0
    assert len(body["data"]) == 1
    b64 = body["data"][0]["b64_json"]
    raw = base64.b64decode(b64, validate=True)
    assert _is_image(raw), f"decoded bytes are not a PNG/JPEG: {raw[:16]!r}"


def test_generate_rejects_disallowed_model_against_live_proxy() -> None:
    """Smoke test that the running proxy enforces the model allowlist (400)."""
    with _client() as c:
        r = c.post(
            "/images/generations",
            json={"model": "dall-e-3", "prompt": "a cat"},
        )

    assert r.status_code == 400
