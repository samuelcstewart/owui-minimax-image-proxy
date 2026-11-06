"""FastAPI app: bridges Open WebUI's OpenAI-dialect `/images/generations` to MiniMax.

Endpoints
---------
- `POST /images/generations` — Open WebUI image generation contract
  (`IMAGE_GENERATION_ENGINE=openai`). OWUI POSTs to
  `{IMAGES_OPENAI_API_BASE_URL}/images/generations` (base URL verbatim, no
  `/v1` appended) with `{"model", "prompt", "n", "size"?, "response_format":
  "b64_json", ...}` and expects `{"created", "data": [{"b64_json": ...}]}`
  with exactly `n` entries.
- `GET /health` — kubelet probe target. Probes upstream MiniMax
  `GET /models` with the bearer key and returns 200/503.

Auth posture
------------
This proxy is designed to run on a trusted cluster network alongside
Open WebUI. The inbound side is NOT authenticated: the proxy never reads
the inbound `Authorization` header (OWUI sends `Bearer
$IMAGES_OPENAI_API_KEY`; it is ignored). Access control is the
deployment's job (CiliumNetworkPolicy in the cluster), not the app's.
The outbound side always sends `Authorization: Bearer $MINIMAX_API_KEY`
to MiniMax.

Error mapping for `/images/generations`
---------------------------------------
- pydantic validation failure (bad model / empty or over-limit prompt /
  n out of 1-9 / malformed size)            → 400
- upstream HTTP error or `base_resp.status_code == 2013` (invalid params)
                                            → 400
- upstream any other `base_resp.status_code != 0`, HTTP 5xx, connection
  error, or timeout                          → 502
- upstream success but wrong image count    → 502
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from owui_minimax_image_proxy.config import Settings, load_settings
from owui_minimax_image_proxy.models import (
    DEFAULT_ASPECT_RATIO,
    MINIMAX_INVALID_PARAMS_STATUS,
    GenerationRequest,
    GenerationResponse,
    ImageData,
    MiniMaxImageGenerationRequest,
    MiniMaxImageGenerationResponse,
    size_to_aspect_ratio,
)

logger = logging.getLogger("owui_minimax_image_proxy")

# Endpoints we call on the upstream MiniMax API.
_IMAGE_GENERATION_PATH = "/image_generation"
_MODELS_PATH = "/models"

_router = APIRouter()


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    settings:
        Optional pre-built `Settings`. If `None`, `load_settings()` is
        called from inside the lifespan (which fails fast when
        `MINIMAX_API_KEY` is unset). Tests pass a `Settings`
        constructed with explicit overrides so the env is irrelevant.
    http_client:
        Optional pre-built `httpx.AsyncClient`. If `None`, one is created
        inside the lifespan. Tests pass a `MockTransport`-backed client
        to short-circuit upstream calls.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Manage the upstream httpx client lifecycle."""
        if http_client is not None:
            # Tests pass a client they own; we must not close it.
            app.state.http_client = http_client
            app.state.owns_client = False
            app.state.settings = settings or load_settings()
            try:
                yield
            finally:
                app.state.http_client = None
            return

        cfg = settings or load_settings()
        app.state.settings = cfg
        # Outbound auth is always the MiniMax API key.
        client = httpx.AsyncClient(
            base_url=cfg.minimax_api_base_url,
            timeout=httpx.Timeout(cfg.image_gen_timeout_secs),
            headers={
                "Authorization": f"Bearer {cfg.minimax_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        app.state.http_client = client
        app.state.owns_client = True
        try:
            yield
        finally:
            await client.aclose()
            app.state.http_client = None

    app = FastAPI(
        title="owui-minimax-image-proxy",
        version="0.1.0",
        lifespan=lifespan,
        # Hide OpenAPI by default — callers pass a pre-built Settings
        # in tests, which means docs are off; otherwise expose /docs for
        # local exploration.
        docs_url="/docs" if (settings is None) else None,
        redoc_url=None,
    )

    # Log 400s so a future request-shape drift (e.g. an Open WebUI
    # upgrade changing the images contract) is visible in proxy logs
    # instead of only in the caller's logs.
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # `exc.errors()` can carry non-JSON-serialisable objects (e.g.
        # the original `ValueError` instance under `ctx['error']`).
        # FastAPI's stock handler runs the same payload through
        # `jsonable_encoder`; mirror that so the response is
        # serialisable.
        logger.warning(
            "rejected request to %s: %s",
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    app.include_router(_router)
    return app


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_http_client(request: Request) -> httpx.AsyncClient:
    """FastAPI dependency that yields the lifespan-managed httpx client."""
    client = getattr(request.app.state, "http_client", None)
    if client is None:  # pragma: no cover — lifespan guarantees this
        raise RuntimeError("http_client not initialised; lifespan did not run")
    return client


def get_settings(request: Request) -> Settings:
    """FastAPI dependency that yields the lifespan-managed settings."""
    cfg = getattr(request.app.state, "settings", None)
    if cfg is None:  # pragma: no cover — lifespan guarantees this
        cfg = load_settings()
        request.app.state.settings = cfg
    return cfg


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@_router.post("/images/generations", response_model=GenerationResponse)
async def images_generations(
    body: GenerationRequest,
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> GenerationResponse:
    """Open WebUI OpenAI-dialect image generation, bridged to MiniMax `/image_generation`.

    Validates `model` against the allowlist, `prompt` against
    `MAX_PROMPT_CHARS`, and `n` against 1-9; maps `size` "WxH" to the
    nearest supported MiniMax `aspect_ratio`; then forwards
    `{model, prompt, aspect_ratio, n, response_format: "base64"}` to
    MiniMax and maps `data.image_base64[]` to `data[].b64_json`,
    preserving order and count.
    """
    if body.model not in settings.allowed_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"model {body.model!r} is not allowed; allowed models: "
                f"{sorted(settings.allowed_models)}"
            ),
        )
    if len(body.prompt) > settings.max_prompt_chars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"prompt is {len(body.prompt)} chars; maximum is "
                f"MAX_PROMPT_CHARS={settings.max_prompt_chars}"
            ),
        )

    try:
        aspect_ratio = (
            size_to_aspect_ratio(body.size) if body.size is not None else DEFAULT_ASPECT_RATIO
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    upstream_body = MiniMaxImageGenerationRequest(
        model=body.model,
        prompt=body.prompt,
        aspect_ratio=aspect_ratio,
        n=body.n,
        response_format="base64",
    ).model_dump(mode="json")
    try:
        resp = await client.post(_IMAGE_GENERATION_PATH, json=upstream_body)
    except httpx.HTTPError as exc:
        logger.warning("upstream connection error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="upstream unreachable",
        ) from exc

    payload = _safe_json(resp)
    if resp.status_code >= 400 or not isinstance(payload, dict):
        logger.warning("upstream /image_generation returned status %s", resp.status_code)
        message = (
            payload.get("base_resp", {}).get("status_msg") if isinstance(payload, dict) else None
        )
        if not message:
            message = f"upstream error (status {resp.status_code})"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=message,
        )

    try:
        upstream = MiniMaxImageGenerationResponse.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="upstream returned malformed image_generation response",
        ) from exc

    if upstream.base_resp.status_code != 0:
        # 2013 = invalid parameters → the caller can fix the request → 400.
        # Anything else is an upstream-side failure → 502.
        status_code = (
            status.HTTP_400_BAD_REQUEST
            if upstream.base_resp.status_code == MINIMAX_INVALID_PARAMS_STATUS
            else status.HTTP_502_BAD_GATEWAY
        )
        logger.warning(
            "upstream base_resp error: status_code=%s status_msg=%s",
            upstream.base_resp.status_code,
            upstream.base_resp.status_msg,
        )
        raise HTTPException(
            status_code=status_code,
            detail=upstream.base_resp.status_msg or f"upstream error (status {resp.status_code})",
        )

    images = upstream.data.image_base64
    if len(images) != body.n:
        logger.warning(
            "upstream returned %d images for n=%d",
            len(images),
            body.n,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"upstream returned {len(images)} images, expected {body.n}",
        )

    return GenerationResponse(
        created=int(time.time()),
        data=[ImageData(b64_json=s) for s in images],
    )


@_router.get("/health")
async def health(
    client: httpx.AsyncClient = Depends(get_http_client),
) -> JSONResponse:
    """kubelet probe target.

    Returns 200 if upstream MiniMax `GET /models` answers 2xx with the
    bearer key, 503 otherwise.
    """
    try:
        resp = await client.get(_MODELS_PATH)
    except httpx.HTTPError as exc:
        logger.warning("upstream health unreachable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "reason": "upstream unreachable"},
        )

    if 200 <= resp.status_code < 300:
        return JSONResponse(status_code=200, content={"status": "ok"})

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "degraded", "reason": f"upstream status {resp.status_code}"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(resp: httpx.Response) -> Any:
    """Return the parsed JSON body or `None` on parse failure."""
    try:
        return resp.json()
    except ValueError:
        logger.warning("upstream returned non-JSON body (status=%s)", resp.status_code)
        return None


# Module-level app for `uvicorn owui_minimax_image_proxy.main:app`. The
# lifespan will load settings from the environment on first request
# (and fail fast if MINIMAX_API_KEY is unset).
app = create_app()
