"""
List all Gemini models available through LiteLLM.

Two sources are shown:
  1. LiteLLM static registry  – models LiteLLM knows about for the ``gemini`` provider.
  2. Live Gemini API           – models returned by ``generativelanguage.googleapis.com``
                                 (only when ``GEMINI_API_KEY`` is set in .env).

Run from the repo root:

  .venv\\Scripts\\python scratch\\list_gemini_models.py

Environment:
  - ``GEMINI_API_KEY`` for live API listing (optional but recommended).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import litellm
import httpx

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Suppress LiteLLM noise
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


# ---------------------------------------------------------------------------
# Source 1 – LiteLLM static registry
# ---------------------------------------------------------------------------

def list_litellm_registry() -> list[str]:
    _section("LiteLLM static registry – gemini/* models")

    raw: list[str] = litellm.models_by_provider.get("gemini", [])
    models = sorted(f"gemini/{m}" if not m.startswith("gemini/") else m for m in raw)

    if not models:
        print("  (no models found – check your litellm version)")
        return []

    print(f"  {len(models)} models registered\n")

    # Header
    col_model = 55
    print(f"  {'Model':<{col_model}}  {'Input $/1M':>10}  {'Output $/1M':>11}  {'Context':>10}  Vision  Audio")
    print(f"  {'-'*col_model}  {'-'*10}  {'-'*11}  {'-'*10}  ------  -----")

    for m in models:
        info = _model_info(m)
        inp  = info.get("input_cost_per_token")
        out  = info.get("output_cost_per_token")
        ctx  = info.get("max_tokens") or info.get("max_input_tokens")
        vision = "yes" if info.get("supports_vision") else "no"
        audio  = "yes" if info.get("supports_audio_input") else "no"

        inp_str = _fmt_num(inp * 1_000_000) if inp is not None else "?"
        out_str = _fmt_num(out * 1_000_000) if out is not None else "?"
        ctx_str = _fmt_num(ctx)

        print(f"  {m:<{col_model}}  {inp_str:>10}  {out_str:>11}  {ctx_str:>10}  {vision:<6}  {audio}")

    return models


# ---------------------------------------------------------------------------
# Source 2 – Live Gemini API (generativelanguage.googleapis.com)
# ---------------------------------------------------------------------------

def list_live_api(api_key: str) -> None:
    _section("Live Gemini API – models endpoint")

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key, "pageSize": 200}

    try:
        resp = httpx.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  FAIL – {type(exc).__name__}: {exc}")
        return

    data = resp.json()
    models: list[dict] = data.get("models", [])

    if not models:
        print("  (empty response)")
        return

    # Sort by name
    models.sort(key=lambda m: m.get("name", ""))

    print(f"  {len(models)} models returned by API\n")

    col_name = 55
    col_disp = 40
    print(f"  {'Name':<{col_name}}  {'Display name':<{col_disp}}  Input limit  Output limit")
    print(f"  {'-'*col_name}  {'-'*col_disp}  -----------  ------------")

    for m in models:
        name  = m.get("name", "?").replace("models/", "gemini/")
        disp  = m.get("displayName", "")[:col_disp]
        inp_l = _fmt_num(m.get("inputTokenLimit"))
        out_l = _fmt_num(m.get("outputTokenLimit"))
        print(f"  {name:<{col_name}}  {disp:<{col_disp}}  {inp_l:>11}  {out_l:>12}")

    # Next-page token warning
    if data.get("nextPageToken"):
        print("\n  NOTE: results were paginated – increase pageSize to see all models.")


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

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        list_live_api(api_key)
    else:
        _section("Live Gemini API – SKIPPED")
        print("  Set GEMINI_API_KEY in .env to query the live model list.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
