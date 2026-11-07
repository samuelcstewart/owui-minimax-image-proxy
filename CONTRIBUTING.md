# Contributing / agent guidance

Guidance for AI agents and contributors working in this repo.
(Shipped as `CONTRIBUTING.md` — same content the project template keeps
in `AGENTS.md`.)

## Scope

This repo is the **source of truth for the `owui-minimax-image-proxy` container
image**. The image is consumed by anything that wants to give Open WebUI an
OpenAI-dialect `/images/generations` endpoint backed by MiniMax image-01.
Changes here should be small, reviewable, and focused on the proxy's HTTP
contract with Open WebUI and MiniMax. Do not redesign the service or refactor
unrelated code.

## Conventions

- **Language**: Python 3.14+, type-checked with `ty`.
- **Linting**: `ruff check` + `ruff format`. No warnings allowed.
- **Tests**: `pytest`. New endpoints or error paths need tests.
- **Commits**: conventional commits (`feat:`, `fix:`, `build:`, etc.).
  52/72 rule. Rebase only — no merge commits.
- **Dependency updates**: edit `pyproject.toml`. The Dockerfile runs
  `uv sync --frozen` against the committed `uv.lock`, so adding a
  runtime dep requires `uv lock` to regenerate the lockfile.
- **Secrets**: `MINIMAX_API_KEY` is never committed — pass it via env
  (compose reads it from the shell environment).

## Open WebUI contract

The proxy implements Open WebUI's OpenAI-dialect image generation client.
That contract is set by OWUI's image generation router (see
`backend/open_webui/routers/images.py` ~line 627 in the OWUI repo,
app 0.13.0) — when in doubt, that source is the source of truth, not
this repo's README.

`POST /images/generations` (OWUI POSTs to `{IMAGES_OPENAI_API_BASE_URL}/images/generations`
— base URL verbatim, **no `/v1` appended**):

- Request: `{"model": str, "prompt": str, "n": int, "size": "WxH" (optional),
  "response_format": "b64_json", ...IMAGES_OPENAI_API_PARAMS}`.
- Response: `{"created": int, "data": [{"url": str} | {"b64_json": str}, ...]}`.
  OWUI calls `raise_for_status()` then reads `res["data"]`; each entry's
  `b64_json` is uploaded to OWUI storage. The proxy MUST return exactly `n`
  entries.
- The proxy does not validate any inbound `Authorization` header.
  OWUI sends one (`Bearer $IMAGES_OPENAI_API_KEY`); the proxy ignores it.

Upstream contract (MiniMax native API — there is no `/v1/images/generations`):

- `POST {MINIMAX_API_BASE_URL}/image_generation` with
  `Authorization: Bearer $MINIMAX_API_KEY` and
  `{"model", "prompt", "aspect_ratio", "n", "response_format": "base64"}` →
  `{"id", "data": {"image_base64": [str, ...]}, "base_resp":
  {"status_code": int, "status_msg": str}}`.
- The proxy always sends `aspect_ratio` (mapped from inbound `size`, default
  `1:1`) and never `width`/`height` — `aspect_ratio` takes priority upstream.
- Non-zero `base_resp.status_code` is an error; `2013` = invalid parameters.

## Auth posture

The proxy intentionally does no inbound auth:

- Inbound: no dependency reads `Authorization`. Any header (or no
  header) is accepted on `/images/generations` and `/health`.
- Outbound: the lifespan attaches `Authorization: Bearer $MINIMAX_API_KEY`
  to every upstream MiniMax call.

This is a deliberate trust posture for "next to OWUI on a local
network", not an oversight. In the homelab cluster, access to the
proxy is enforced by Cilium NetworkPolicy, not by the app. If the
deployment ever crosses a trust boundary, that's the deployment's job
to enforce (NetworkPolicy, sidecar, ingress) — not the proxy's.

Changes to this contract are breaking for the OWUI side. Bump the
minor version in `pyproject.toml` and call out the change in the PR
description.

## Build

The image is built with buildx. The bundled `docker-compose.yml` does
this for local dev (proxy + a repo-owned stub upstream that mimics
the MiniMax API from `tests/stub_upstream.py` — no key or external
network needed):

```
docker compose build
```

For a multi-arch (`linux/amd64` + `linux/arm64`) image loaded into
the local Docker daemon:

```
poe build
```

For a multi-arch (`linux/amd64` + `linux/arm64`) image loaded into
the local Docker daemon:

```
poe build
```

Requires Docker 29+ with the containerd image store (snapshotter)
enabled. That's what makes `buildx build --load` accept a
multi-platform build; on classic single-platform storage you'll get
an error.

## Publish

A GitHub Actions workflow at `.github/workflows/ci.yml` runs on every
push to `main`, every `v*.*.*` tag, and every pull request against
`main`, and on manual dispatch. Two jobs:

- `test` — `uv sync` + `uv run poe check` (ruff + format-check + ty +
  pytest). Runs first; the build job depends on it.
- `build` — multi-arch (linux/amd64 + linux/arm64) `buildx build`.
  On `push` events, pushes the result to GHCR. On `pull_request`
  events, builds for verification only (`push: false`).

The published image lives at:

```
ghcr.io/samuelcstewart/owui-minimax-image-proxy
```

Tag scheme on push events:

- main push → `:main`, `:<short-sha>`, `:latest`
- `v0.1.0` tag → `:v0.1.0`, `:v0.1`, `:v0`, `:<short-sha>`

Visibility inherits from the repo (public). The `GITHUB_TOKEN` in the
runner has `packages: write` via the workflow's `permissions` block.

## What this repo is NOT

- Not a deployment repo. Bring the image up however you like
  (`docker run`, compose, Kubernetes, etc.) and set `MINIMAX_API_KEY`.
- Not an Open WebUI config repo. Wiring `IMAGE_GENERATION_ENGINE=openai`,
  `IMAGES_OPENAI_API_BASE_URL`, etc. into the OWUI HelmRelease is
  downstream of this project.
- Not an image edit/inpaint proxy. MiniMax `subject_reference` exists, but
  OWUI 0.13.0's `ENABLE_IMAGE_EDIT` path is a separate engine — this repo
  deliberately has only the text-to-image generations route.
- Not a MiniMax account manager. Key rotation/quotas are MiniMax-side.
- Not a test target for Open WebUI itself. The proxy just satisfies
  Open WebUI's image generation contract.
