# owui-minimax-image-proxy

A small HTTP service that bridges [Open WebUI](https://github.com/open-webui/open-webui)'s
OpenAI-dialect image generation endpoint (`POST /images/generations`, used when
`IMAGE_GENERATION_ENGINE=openai`) to [MiniMax](https://www.minimax.io)'s native
`POST /v1/image_generation` API (`image-01`). The proxy:

- Accepts `POST /images/generations` with the OpenAI-dialect body OWUI sends
  (`{"model", "prompt", "n", "size"?, "response_format": "b64_json", ...}`).
- Maps `size` "WxH" to the nearest supported MiniMax `aspect_ratio` (never sends
  `width`/`height` upstream).
- Forwards to MiniMax's native `POST {MINIMAX_API_BASE_URL}/image_generation`
  with `Authorization: Bearer $MINIMAX_API_KEY` — MiniMax has **no**
  `/v1/images/generations` (404), which is the gap this shim fills.
- Returns `{"created", "data": [{"b64_json": ...}, ...]}` with exactly `n`
  entries, order preserved — the shape OWUI reads after `raise_for_status()`.
- Exposes `GET /health` for liveness probes — probes upstream MiniMax
  `GET /models` with the bearer key and returns 200/503.

## Why this exists

Open WebUI 0.13.0 with `IMAGE_GENERATION_ENGINE=openai` POSTs to
`{IMAGES_OPENAI_API_BASE_URL}/images/generations` (base URL verbatim, **no `/v1`
appended**) — see `backend/open_webui/routers/images.py` (~line 627) in the
OWUI repo, the source of truth. MiniMax's image API lives at
`POST https://api.minimax.io/v1/image_generation` with a different body and
response envelope, and `/v1/images/generations` returns 404. This proxy is the
shim between the two dialects. Point OWUI at it with:

```
IMAGES_OPENAI_API_BASE_URL=http://owui-minimax-image-proxy.open-webui.svc
IMAGE_GENERATION_MODEL=image-01
```

## Stack

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for package management
- [ruff](https://docs.astral.sh/ruff/) for lint + format
- [ty](https://docs.astral.sh/ty/) for type checking
- FastAPI + uvicorn for the HTTP server
- httpx for upstream calls
- pydantic / pydantic-settings for config + request/response models

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MINIMAX_API_KEY` | **(required)** | MiniMax API key. Startup fails fast if empty. Sent as `Authorization: Bearer` on all upstream calls |
| `MINIMAX_API_BASE_URL` | `https://api.minimax.io/v1` | Upstream base URL (trailing slash stripped); image endpoint is `{base}/image_generation` |
| `MINIMAX_ALLOWED_MODELS` | `image-01,image-01-live` | Comma-separated allowlist of inbound `model` values; anything else → 400 |
| `IMAGE_GEN_TIMEOUT_SECS` | `120` | Upstream request timeout (seconds). One 1:1 image takes ~24 s |
| `MAX_PROMPT_CHARS` | `1500` | Inbound prompt length cap (MiniMax's own limit) |
| `IMAGE_PROXY_PORT` / `PORT` | `8000` | Listen port hint for local runs. A non-numeric `PORT` (Kubernetes service-links inject `PORT=tcp://<ip>:<port>`) is ignored |

The proxy never validates any inbound `Authorization` header on
`/images/generations`. OWUI sends one by default
(`Bearer $IMAGES_OPENAI_API_KEY`) and the proxy simply ignores it. This proxy is
designed to run on a trusted cluster network alongside Open WebUI — access
control is enforced by the cluster's Cilium NetworkPolicy, not by the app.
Outbound, the proxy always attaches `Authorization: Bearer $MINIMAX_API_KEY`.

## Local development

```bash
uv sync                              # install all deps
uv run pytest                        # run unit tests
uv run ruff check                    # lint
uv run ruff format                   # format
uv run ty check                      # type check
MINIMAX_API_KEY=... uv run uvicorn owui_minimax_image_proxy.main:app --reload --port 8000
```

## Docker

```bash
export MINIMAX_API_KEY=...           # real key, never committed
docker compose up --build            # brings up this proxy (upstream MiniMax is remote)
curl -X POST http://localhost:8000/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"image-01","prompt":"a red square on white","n":1,"size":"1024x1024","response_format":"b64_json"}' \
  | jq '.data[0].b64_json | length'
curl http://localhost:8000/health | jq
```

For a standalone build of the proxy image only:

```bash
poe build                            # multi-arch (amd64+arm64), --load (Docker 29+)
```

Integration tests against the live MiniMax API (one real image, ~24 s):

```bash
MINIMAX_API_KEY=... poe integration-test   # compose up → pytest -m integration → compose down
```

## API

### `POST /images/generations`

Open WebUI POSTs to `{IMAGES_OPENAI_API_BASE_URL}/images/generations` with
`{"model": "image-01", "prompt": "...", "n": 1, "size": "1024x1024" (optional),
"response_format": "b64_json", ...IMAGES_OPENAI_API_PARAMS}`. The proxy
validates (`model` in `MINIMAX_ALLOWED_MODELS`, prompt 1..`MAX_PROMPT_CHARS`,
`n` 1-9, `size` "WxH" if present), maps `size` to the nearest MiniMax
`aspect_ratio`, and returns:

```json
// Request (OWUI / OpenAI dialect)
{
  "model": "image-01",
  "prompt": "a red square on a white background",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json"
}
// Response
{
  "created": 1762400000,
  "data": [
    {"b64_json": "iVBORw0KGgo..."}
  ]
}
```

Internally the proxy calls MiniMax as:

```json
// POST {MINIMAX_API_BASE_URL}/image_generation
{
  "model": "image-01",
  "prompt": "a red square on a white background",
  "aspect_ratio": "1:1",
  "n": 1,
  "response_format": "base64"
}
```

and maps MiniMax's `{"data": {"image_base64": [...]}, "base_resp":
{"status_code": 0, "status_msg": "success"}}` to the OpenAI-dialect response.
`size` examples: `1024x1024`→`1:1`, `1792x1024`→`16:9`, `1024x1792`→`9:16`.

Errors (OpenAI-dialect `{"error": {"message": ...}}` semantics; OWUI surfaces
the body on non-2xx): invalid input or upstream `base_resp.status_code == 2013`
→ 400; any other upstream failure (HTTP error, other non-zero `base_resp`,
timeout, malformed response, wrong image count) → 502.

### `GET /health`

```json
{"status": "ok"}        // 200 if upstream MiniMax GET /v1/models answers 2xx with the bearer key
{"status": "degraded"}  // 503 otherwise (unreachable, 401, 5xx, ...)
```

## Publishing

A GitHub Actions workflow builds and pushes a multi-arch image to
GHCR. The same workflow also runs on every PR (build + tests, no
push) as a Dockerfile regression gate:

```
ghcr.io/samuelcstewart/owui-minimax-image-proxy
```

Tag scheme:

- main pushes → `:main`, `:<short-sha>`, `:latest`
- `v0.1.0` tag → `:v0.1.0`, `:v0.1`, `:v0`, `:<short-sha>`

PR builds run the same `buildx` invocation with `push: false`, so PRs
get a build verification (and the test suite) but never produce a
throwaway GHCR image.

Pullable directly:

```bash
docker pull ghcr.io/samuelcstewart/owui-minimax-image-proxy:latest
```

## License

MIT.
