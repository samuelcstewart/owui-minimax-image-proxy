"""Tests for `POST /images/generations`."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from tests.conftest import make_minimax_response


def test_generations_happy_path_maps_base64_and_preserves_order(make_client):  # type: ignore[no-untyped-def]
    """MiniMax `data.image_base64[]` → `data[].b64_json`, same order and count."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/image_generation"
        assert request.headers["Authorization"] == "Bearer test-key"
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=make_minimax_response(images=["QUJD", "REVG", "R0hJ"]),
        )

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={
                "model": "image-01",
                "prompt": "a cat on the moon",
                "n": 3,
                "size": "1024x1024",
                "response_format": "b64_json",
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["created"], int) and body["created"] > 0
    assert body["data"] == [
        {"b64_json": "QUJD", "url": None},
        {"b64_json": "REVG", "url": None},
        {"b64_json": "R0hJ", "url": None},
    ]
    # Upstream body must be MiniMax's native shape — aspect_ratio mapped
    # from size, base64 always requested, no size/response_format passthrough.
    assert captured["body"] == {
        "model": "image-01",
        "prompt": "a cat on the moon",
        "aspect_ratio": "1:1",
        "n": 3,
        "response_format": "base64",
    }


@pytest.mark.parametrize(
    ("size", "expected_aspect"),
    [
        ("1024x1024", "1:1"),
        ("1792x1024", "16:9"),
        ("1024x1792", "9:16"),
        ("1234x777", "3:2"),
    ],
)
def test_generations_maps_size_to_aspect_ratio(make_client, size, expected_aspect):  # type: ignore[no-untyped-def]
    """`size` "WxH" is mapped to the nearest aspect_ratio upstream."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p", "size": size},
        )

    assert r.status_code == 200
    assert captured["body"]["aspect_ratio"] == expected_aspect
    # Never send width/height upstream.
    assert "width" not in captured["body"]
    assert "height" not in captured["body"]
    assert "size" not in captured["body"]


def test_generations_defaults_to_1x1_when_no_size(make_client):  # type: ignore[no-untyped-def]
    """No `size` in the request → default aspect_ratio 1:1."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 200
    assert captured["body"]["aspect_ratio"] == "1:1"
    assert captured["body"]["n"] == 1


def test_generations_accepts_image_01_live(make_client):  # type: ignore[no-untyped-def]
    """`image-01-live` is in the default allowlist."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "image-01-live"
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01-live", "prompt": "p"},
        )

    assert r.status_code == 200


def test_generations_passthrough_n_and_asserts_count(make_client):  # type: ignore[no-untyped-def]
    """`n` is forwarded; a mismatched upstream image count is a hard 502."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["n"] == 4
        # Adversarial upstream: returns only 2 images for n=4.
        return httpx.Response(200, json=make_minimax_response(images=["QQ==", "Qg=="]))

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p", "n": 4},
        )

    assert r.status_code == 502
    assert "expected 4" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "p"},  # missing model
        {"model": "dall-e-3", "prompt": "p"},  # model not in allowlist
        {"model": "image-01"},  # missing prompt
        {"model": "image-01", "prompt": ""},  # empty prompt
        {"model": "image-01", "prompt": "   "},  # whitespace-only prompt
        {"model": "image-01", "prompt": "p", "n": 0},  # n below 1
        {"model": "image-01", "prompt": "p", "n": 10},  # n above 9
        {"model": "image-01", "prompt": "p", "n": "three"},  # non-int n
        {"model": "image-01", "prompt": "p", "size": "1024"},  # malformed size
        {"model": "image-01", "prompt": "x" * 1501},  # over MAX_PROMPT_CHARS
    ],
)
def test_generations_rejects_invalid_requests_with_400(make_client, payload):  # type: ignore[no-untyped-def]
    """Invalid model / prompt / n / size → 400 (not 422), upstream never called."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should not be called")

    with make_client(handler) as client:
        r = client.post("/images/generations", json=payload)

    assert r.status_code == 400
    assert "detail" in r.json()


def test_generations_accepts_prompt_at_exactly_max(make_client):  # type: ignore[no-untyped-def]
    """A prompt of exactly MAX_PROMPT_CHARS is accepted (boundary)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "x" * 1500},
        )

    assert r.status_code == 200


def test_generations_custom_allowed_models(make_client):  # type: ignore[no-untyped-def]
    """MINIMAX_ALLOWED_MODELS narrows the inbound model allowlist."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should not be called")

    with make_client(handler, minimax_allowed_models="custom-model") as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )
    assert r.status_code == 400

    def handler2(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler2, minimax_allowed_models="custom-model") as client:
        r = client.post(
            "/images/generations",
            json={"model": "custom-model", "prompt": "p"},
        )
    assert r.status_code == 200


def test_generations_logs_400_validation_errors(  # type: ignore[no-untyped-def]
    make_client, caplog: pytest.LogCaptureFixture
) -> None:
    """400s are logged so request-shape drift is visible in proxy logs."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should not be called")

    with (
        make_client(handler) as client,
        caplog.at_level(logging.WARNING, logger="owui_minimax_image_proxy"),
    ):
        # n=99 fails pydantic validation (le=9) → RequestValidationError →
        # the custom 400 handler logs it.
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p", "n": 99},
        )

    assert r.status_code == 400
    assert any(
        "rejected request to /images/generations" in rec.getMessage() for rec in caplog.records
    )


def test_generations_maps_base_resp_2013_to_400(make_client):  # type: ignore[no-untyped-def]
    """`base_resp.status_code == 2013` (invalid params) → 400 with status_msg."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=make_minimax_response(status_code=2013, status_msg="invalid parameters"),
        )

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 400
    assert r.json()["detail"] == "invalid parameters"


@pytest.mark.parametrize("base_status", [1002, 1004, 1039, 2038, 4000, 5000])
def test_generations_maps_other_base_resp_errors_to_502(make_client, base_status):  # type: ignore[no-untyped-def]
    """Any other non-zero `base_resp.status_code` → 502 with status_msg."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=make_minimax_response(status_code=base_status, status_msg=f"err {base_status}"),
        )

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502
    assert r.json()["detail"] == f"err {base_status}"


@pytest.mark.parametrize("upstream_status", [400, 401, 403, 404, 422, 500, 502, 503, 504])
def test_generations_maps_upstream_http_error_to_502(make_client, upstream_status):  # type: ignore[no-untyped-def]
    """Upstream HTTP-level errors → 502 (with status_msg when present)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            upstream_status,
            json={"base_resp": {"status_code": 1001, "status_msg": "upstream exploded"}},
        )

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502
    assert r.json()["detail"] == "upstream exploded"


def test_generations_maps_upstream_http_error_without_body_to_502(make_client):  # type: ignore[no-untyped-def]
    """Upstream HTTP error with a non-JSON body → 502 with the status in the message."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="<html>nginx error</html>")

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502
    assert "status 500" in r.json()["detail"]


def test_generations_maps_timeout_to_502(make_client):  # type: ignore[no-untyped-def]
    """httpx timeout → 502 `upstream unreachable`."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502
    assert r.json() == {"detail": "upstream unreachable"}


def test_generations_maps_connection_error_to_502(make_client):  # type: ignore[no-untyped-def]
    """httpx transport errors (connection refused, DNS) → 502."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502
    assert r.json() == {"detail": "upstream unreachable"}


def test_generations_handles_malformed_upstream_json(make_client):  # type: ignore[no-untyped-def]
    """Upstream 200 with a non-JSON body → 502."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>oops</html>")

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502


def test_generations_handles_missing_base_resp(make_client):  # type: ignore[no-untyped-def]
    """Upstream 200 without `base_resp` → 502 (malformed envelope)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "data": {"image_base64": ["QQ=="]}})

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 502
    assert "malformed" in r.json()["detail"]


def test_generations_ignores_inbound_response_format_and_extras(make_client):  # type: ignore[no-untyped-def]
    """`response_format: "url"` and extra OWUI params are accepted but not forwarded."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={
                "model": "image-01",
                "prompt": "p",
                "response_format": "url",
                "quality": "hd",
                "style": "vivid",
            },
        )

    assert r.status_code == 200
    assert captured["body"]["response_format"] == "base64"
    assert "quality" not in captured["body"]
    assert "style" not in captured["body"]


# ---------------------------------------------------------------------------
# Auth posture: no inbound auth, bearer key outbound
# ---------------------------------------------------------------------------


def test_generations_ignores_inbound_authorization_header(make_client):  # type: ignore[no-untyped-def]
    """Inbound `Authorization` is never validated; any value is accepted.

    OWUI sends `Bearer $IMAGES_OPENAI_API_KEY`; the proxy ignores it.
    Access control is enforced by the cluster's CiliumNetworkPolicy.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler) as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
            headers={"Authorization": "Bearer anything-goes"},
        )

    assert r.status_code == 200


def test_generations_sends_bearer_key_upstream(make_client):  # type: ignore[no-untyped-def]
    """The upstream MiniMax call always carries `Authorization: Bearer $MINIMAX_API_KEY`."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer secret-123"
        return httpx.Response(200, json=make_minimax_response())

    with make_client(handler, minimax_api_key="secret-123") as client:
        r = client.post(
            "/images/generations",
            json={"model": "image-01", "prompt": "p"},
        )

    assert r.status_code == 200
