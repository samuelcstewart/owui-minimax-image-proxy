"""Integration tests: sanity-check the RUNNING IMAGE against a stub upstream.

These tests are NOT run by `poe test` (the default test task excludes
the `integration` marker). They're invoked by `poe integration-test`,
which:

1. `docker compose up -d --build` — starts this proxy image plus a
   repo-owned stub (`tests/stub_upstream.py`, served by
   `python:3.14-slim` on :9000) that mimics the MiniMax API. No real
   MiniMax account, API key, or external network is involved.
2. Waits for `GET $PROXY_HEALTH_URL` to return 2xx (health proxies the
   stub's `GET /models`).
3. Runs `pytest -m integration -v --tb=short`.
4. `docker compose down` — tears the stack down (via a shell `trap`).

What is asserted (image sanity, not MiniMax behaviour):
- the built container boots and serves HTTP on :8000 (no crash loop)
- `GET /health` is 200 `{"status": "ok"}` when the upstream answers
- `POST /images/generations` round-trips through the running image:
  OWUI-dialect request → proxy → stub `/image_generation` → OWUI-shaped
  `{"created", "data": [{"b64_json"}]}` response with exactly `n`
  entries, each decodable to PNG/JPEG magic bytes
- validation enforced by the running image: disallowed model → 400,
  empty prompt → 400, `n` out of range → 400

The proxy is reachable at http://localhost:8000 (compose port
forward). `PROXY_URL` can override the target for dockerized runners.

Marking convention
------------------
Every test in this file is decorated with `@pytest.mark.integration`
so the default `poe test` run skips it. The matching
`[tool.pytest.ini_options].markers` entry in `pyproject.toml` registers
the marker so `--strict-markers` is satisfied.
"""

from __future__ import annotations

import base64
import os

import httpx
import pytest

# Reuse the proxy's expected port forward (set by docker compose).
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8000")

pytestmark = pytest.mark.integration

# PNG and JPEG magic headers (JPEG starts with FF D8 FF).
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")


def _is_image(raw: bytes) -> bool:
    return any(raw.startswith(m) for m in IMAGE_MAGIC)


def _client() -> httpx.Client:
    """Plain sync httpx client — the proxy is HTTP/1.1."""
    return httpx.Client(base_url=PROXY_URL, timeout=httpx.Timeout(30.0))


def test_health_endpoint_reports_ok_against_stub_upstream() -> None:
    """`/health` returns 200 + `{"status": "ok"}` when the upstream answers."""
    with _client() as c:
        r = c.get("/health")

    assert r.status_code == 200, f"unexpected status: {r.status_code} body={r.text!r}"
    assert r.json() == {"status": "ok"}


def test_generate_image_round_trip_through_running_image() -> None:
    """POST /images/generations → stub → OWUI-shaped response, n entries, image magic.

    Exercises the full path inside the built image: request validation →
    size→aspect_ratio mapping → upstream call → base64 mapping →
    response shape. The stub returns a real (tiny) PNG per image.
    """
    with _client() as c:
        r = c.post(
            "/images/generations",
            json={
                "model": "image-01",
                "prompt": "a minimal red square on a white background, flat vector style",
                "n": 2,
                "size": "1024x1024",
                "response_format": "b64_json",
            },
        )

    assert r.status_code == 200, f"unexpected status: {r.status_code} body={r.text!r}"
    body = r.json()
    assert isinstance(body["created"], int) and body["created"] > 0
    assert len(body["data"]) == 2
    for entry in body["data"]:
        raw = base64.b64decode(entry["b64_json"], validate=True)
        assert _is_image(raw), f"decoded bytes are not a PNG/JPEG: {raw[:16]!r}"


def test_generate_rejects_disallowed_model() -> None:
    """The running image enforces the model allowlist (400)."""
    with _client() as c:
        r = c.post(
            "/images/generations",
            json={"model": "dall-e-3", "prompt": "a cat"},
        )

    assert r.status_code == 400


def test_generate_rejects_empty_prompt() -> None:
    """The running image rejects an empty prompt (400)."""
    with _client() as c:
        r = c.post(
            "/images/generations",
            json={"model": "image-01", "prompt": ""},
        )

    assert r.status_code == 400


def test_generate_rejects_out_of_range_n() -> None:
    """The running image rejects n outside 1-9 (400)."""
    with _client() as c:
        r = c.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "a cat", "n": 99},
        )

    assert r.status_code == 400
