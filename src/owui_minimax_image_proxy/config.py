"""Settings for the proxy.

Env vars (no prefix, case-insensitive):

- `MINIMAX_API_KEY`        — **required**; MiniMax API key. Startup fails
  fast if empty. Sent as `Authorization: Bearer` on all upstream calls.
- `MINIMAX_API_BASE_URL`   — upstream base URL (default
  `https://api.minimax.io/v1`, trailing slash stripped). The image
  generation endpoint is `{MINIMAX_API_BASE_URL}/image_generation`.
- `MINIMAX_ALLOWED_MODELS` — comma-separated allowlist of inbound `model`
  values (default `image-01,image-01-live`). Anything else → 400.
- `IMAGE_GEN_TIMEOUT_SECS` — httpx timeout for upstream image generation
  calls (default 120; one 1:1 image takes ~24 s).
- `MAX_PROMPT_CHARS`       — inbound prompt length cap (default 1500,
  MiniMax's own limit).
- `IMAGE_PROXY_PORT` / `PORT` — listen port hint for local runs
  (default 8000). A non-numeric `PORT` (Kubernetes service-links inject
  `PORT=tcp://<cluster-ip>:<port>`) is ignored.

The proxy never validates any inbound `Authorization` header. It is
designed to run on a trusted cluster network next to Open WebUI; access
control is CiliumNetworkPolicy's job, not the app's. Outbound, the proxy
always attaches `Authorization: Bearer $MINIMAX_API_KEY` to MiniMax.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MINIMAX_API_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_ALLOWED_MODELS = "image-01,image-01-live"


class Settings(BaseSettings):
    """Runtime configuration loaded from env vars.

    The proxy does no inbound auth. If the deployment ever crosses a
    trust boundary, that's the deployment's job to enforce
    (NetworkPolicy, sidecar, ingress) — not the proxy's.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=None,  # never read from disk
        case_sensitive=False,
        extra="ignore",
    )

    minimax_api_key: str = Field(
        default="",
        description="MiniMax API key. Sent as `Authorization: Bearer` upstream.",
    )

    minimax_api_base_url: str = Field(
        default=DEFAULT_MINIMAX_API_BASE_URL,
        description="Upstream MiniMax base URL (trailing slash stripped).",
    )

    minimax_allowed_models: str = Field(
        default=DEFAULT_ALLOWED_MODELS,
        description="Comma-separated allowlist of inbound `model` values.",
    )

    image_gen_timeout_secs: float = Field(
        default=120.0,
        gt=0,
        description="httpx request timeout for upstream image generation calls (seconds).",
    )

    max_prompt_chars: int = Field(
        default=1500,
        ge=1,
        description="Inbound prompt length cap (MiniMax's own limit is 1500 chars).",
    )

    port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        validation_alias=AliasChoices("IMAGE_PROXY_PORT", "PORT"),
        description="Listen port hint for local (non-container) runs.",
    )

    @field_validator("port", mode="before")
    @classmethod
    def _ignore_non_integer_port(cls, v: Any) -> Any:
        # Kubernetes injects PORT=tcp://<svc-ip>:<port> into every pod
        # via service-links. It is not a listen port — the Dockerfile CMD
        # pins 8000 — so ignore non-integer values and fall back to the
        # default.
        if isinstance(v, str) and not v.isdigit():
            return 8000
        return v

    @field_validator("minimax_api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("minimax_api_key")
    @classmethod
    def _require_api_key(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "MINIMAX_API_KEY is required and must be non-empty — "
                "refusing to start without an upstream credential"
            )
        return v

    @property
    def allowed_models(self) -> frozenset[str]:
        """Parsed allowlist of inbound `model` values (whitespace-stripped)."""
        return frozenset(m.strip() for m in self.minimax_allowed_models.split(",") if m.strip())


def load_settings(**overrides: Any) -> Settings:
    """Construct a `Settings` instance.

    `overrides` are passed as keyword arguments and take precedence
    over env vars. Used by tests to inject config without touching
    the real environment.
    """
    return Settings(**overrides)
