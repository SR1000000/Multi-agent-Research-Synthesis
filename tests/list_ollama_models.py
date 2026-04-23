"""
List all Ollama models available through LiteLLM.

Two sources are shown:
  1. LiteLLM static registry  – models LiteLLM knows about for the ``ollama`` provider.
  2. Live Ollama API           – models returned by the running Ollama instance at
                                 ``OLLAMA_BASE_URL`` (falls back to http://localhost:11434).
                                 Requires the Ollama server to be reachable; API key is not
                                 needed for the tags endpoint.

Run from the repo root:

  .venv\\Scripts\\python scratch\\list_ollama_models.py

Environment (from .env at repo root):
  - ``OLLAMA_BASE_URL``  – base URL of the Ollama server (default: http://localhost:11434).
  - ``OLLAMA_API_KEY``   – only needed if your Ollama instance requires auth.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import litellm
import httpx

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _model_info(model_id: str) -> dict:
    """Return capability metadata from the LiteLLM model cost map (best-effort)."""
    try:
        info = litellm.get_model_info(model_id)
    except Exception:
        info = {}
    return info or {}


def _fmt_num(value: int | float | None) -> str:
    if value is None:
        return "?"
    if isinstance(value, float) and value < 1:
        return f"{value:.4f}"
    return f"{int(value):,}"


def _fmt_size(size_bytes: int | None) -> str:
    """Human-readable size from bytes."""
    if size_bytes is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Source 1 – LiteLLM static registry
# ---------------------------------------------------------------------------

def list_litellm_registry() -> list[str]:
    _section("LiteLLM static registry – ollama/* models")

    raw: list[str] = litellm.models_by_provider.get("ollama", [])
    models = sorted(f"ollama/{m}" if not m.startswith("ollama/") else m for m in raw)

    if not models:
        print("  (no models found – check your litellm version)")
        return []

    print(f"  {len(models)} models registered\n")

    col_model = 55
    print(f"  {'Model':<{col_model}}  {'Input $/1M':>10}  {'Output $/1M':>11}  {'Context':>10}  Vision")
    print(f"  {'-'*col_model}  {'-'*10}  {'-'*11}  {'-'*10}  ------")

    for m in models:
        info   = _model_info(m)
        inp    = info.get("input_cost_per_token")
        out    = info.get("output_cost_per_token")
        ctx    = info.get("max_tokens") or info.get("max_input_tokens")
        vision = "yes" if info.get("supports_vision") else "no"

        inp_str = _fmt_num(inp * 1_000_000) if inp is not None else "?"
        out_str = _fmt_num(out * 1_000_000) if out is not None else "?"
        ctx_str = _fmt_num(ctx)

        print(f"  {m:<{col_model}}  {inp_str:>10}  {out_str:>11}  {ctx_str:>10}  {vision}")

    return models


# ---------------------------------------------------------------------------
# Source 2 – Live Ollama API  (/api/tags)
# ---------------------------------------------------------------------------

def list_live_api(base_url: str, api_key: str | None) -> None:
    _section(f"Live Ollama API – {base_url}/api/tags")

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = base_url.rstrip("/") + "/api/tags"

    try:
        resp = httpx.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  FAIL – {type(exc).__name__}: {exc}")
        return

    data = resp.json()
    models: list[dict] = data.get("models", [])

    if not models:
        print("  (no models returned – is the Ollama server running?)")
        return

    models.sort(key=lambda m: m.get("name", ""))

    print(f"  {len(models)} models available on server\n")

    col_name   = 45
    col_family = 18
    print(f"  {'Name':<{col_name}}  {'Family':<{col_family}}  {'Size':>9}  {'Params':>9}  Quant")
    print(f"  {'-'*col_name}  {'-'*col_family}  {'-'*9}  {'-'*9}  -----")

    for m in models:
        name    = m.get("name", "?")
        details = m.get("details", {})
        family  = (details.get("family") or "")[:col_family]
        size    = _fmt_size(m.get("size"))
        params  = details.get("parameter_size") or "?"
        quant   = details.get("quantization_level") or "?"

        litellm_name = f"ollama/{name}"
        print(f"  {litellm_name:<{col_name}}  {family:<{col_family}}  {size:>9}  {params:>9}  {quant}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

    print(f"Python : {sys.executable}")
    try:
        version = litellm.version
    except AttributeError:
        try:
            from importlib.metadata import version as _v
            version = _v("litellm")
        except Exception:
            version = "unknown"
    print(f"litellm: {version}  ({litellm.__file__})")

    list_litellm_registry()

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    api_key  = os.environ.get("OLLAMA_API_KEY")
    list_live_api(base_url, api_key)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
