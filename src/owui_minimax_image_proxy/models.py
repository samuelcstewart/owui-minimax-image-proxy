"""Request/response models.

The public `/images/generations` contract matches Open WebUI's
OpenAI-dialect image generation client (`backend/open_webui/routers/images.py`
in the open-webui/open-webui repo, ~line 627, app 0.13.0 — the source of
truth):

- Request: `POST {IMAGES_OPENAI_API_BASE_URL}/images/generations` — the base
  URL is used verbatim with `/images/generations` appended (no `/v1`) — with
  `{"model": str, "prompt": str, "n": int, "size": "WxH" (optional),
  "response_format": "b64_json", ...IMAGES_OPENAI_API_PARAMS}`.
- Response: `{"created": int, "data": [{"url": str} | {"b64_json": str}, ...]}`.
  OWUI calls `raise_for_status()` then reads `res["data"]`, so a non-2xx
  must carry a JSON error body.

The internal upstream payload is MiniMax's native image generation shape:

- `POST {MINIMAX_API_BASE_URL}/image_generation` with `{"model", "prompt",
  "aspect_ratio", "n", "response_format": "base64"}` →
  `{"id": ..., "data": {"image_base64": [str, ...]}, "base_resp":
  {"status_code": int, "status_msg": str}}`.
- MiniMax has **no** `/v1/images/generations` (404) — that gap is the
  whole reason this shim exists.
- `aspect_ratio` takes priority over `width`/`height` upstream, so the
  proxy always sends `aspect_ratio` and never `width`/`height`.
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# MiniMax image-01 supported aspect ratios, as (label, width, height).
SUPPORTED_ASPECT_RATIOS: tuple[tuple[str, int, int], ...] = (
    ("1:1", 1, 1),
    ("16:9", 16, 9),
    ("4:3", 4, 3),
    ("3:2", 3, 2),
    ("2:3", 2, 3),
    ("3:4", 3, 4),
    ("9:16", 9, 16),
    ("21:9", 21, 9),
)

DEFAULT_ASPECT_RATIO = "1:1"

_SIZE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)

# MiniMax base_resp.status_code signalling invalid parameters.
MINIMAX_INVALID_PARAMS_STATUS = 2013


def size_to_aspect_ratio(size: str) -> str:
    """Map an OpenAI-dialect `size` string ("WxH") to the nearest supported
    MiniMax `aspect_ratio` label.

    The ratio comparison is done in log space so portrait vs landscape
    are symmetric: 1024x1792 (ratio 0.571) is as close to 9:16 (0.5625)
    as 1792x1024 (1.75) is to 16:9 (1.778).

    Raises `ValueError` on a malformed size string.
    """
    match = _SIZE_RE.match(size)
    if match is None:
        raise ValueError(f"size must be of the form 'WxH' (e.g. '1024x1024'), got {size!r}")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"size dimensions must be positive, got {size!r}")

    log_ratio = math.log(width / height)
    label, _, _ = min(
        SUPPORTED_ASPECT_RATIOS,
        key=lambda wh: abs(math.log(wh[1] / wh[2]) - log_ratio),
    )
    return label


class GenerationRequest(BaseModel):
    """Request body for `POST /images/generations` — the OpenAI dialect OWUI sends.

    `response_format` and any `IMAGES_OPENAI_API_PARAMS` extras are accepted
    and ignored: the proxy always asks MiniMax for base64 and maps the
    result to `b64_json` itself.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., min_length=1, description="MiniMax model id, e.g. 'image-01'.")
    prompt: str = Field(..., min_length=1, description="Text prompt (1..MAX_PROMPT_CHARS).")
    n: int = Field(default=1, ge=1, le=9, description="Number of images to generate (1-9).")
    size: str | None = Field(
        default=None,
        description="Optional 'WxH' size; mapped to the nearest MiniMax aspect_ratio.",
    )

    @field_validator("prompt")
    @classmethod
    def _prompt_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must be a non-empty, non-whitespace string")
        return v


class ImageData(BaseModel):
    """One generated image in the OpenAI response dialect."""

    b64_json: str = Field(..., description="Base64-encoded image bytes.")
    url: str | None = Field(default=None, description="Always None — the shim returns base64.")


class GenerationResponse(BaseModel):
    """Response body for `POST /images/generations` — OpenAI dialect."""

    created: int = Field(..., description="Unix timestamp of the response.")
    data: list[ImageData] = Field(..., description="Exactly `n` entries, order preserved.")


class MiniMaxImageGenerationRequest(BaseModel):
    """Internal request body for MiniMax's native `POST /image_generation`."""

    model: str
    prompt: str
    aspect_ratio: str
    n: int
    response_format: str = "base64"


class MiniMaxBaseResp(BaseModel):
    """MiniMax envelope status. Non-zero `status_code` is an error."""

    status_code: int
    status_msg: str = ""


class MiniMaxImageData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_base64: list[str] = Field(default_factory=list)


class MiniMaxImageGenerationResponse(BaseModel):
    """MiniMax native `/image_generation` response envelope."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    data: MiniMaxImageData = Field(default_factory=MiniMaxImageData)
    base_resp: MiniMaxBaseResp


class ErrorBody(BaseModel):
    """OpenAI-dialect error body: `{"error": {"message": ...}}`."""

    error: dict[str, Any]
