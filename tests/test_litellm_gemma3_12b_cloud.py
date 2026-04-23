"""
Quick smoke test for LiteLLM completion via Ollama Cloud: gemma3:12b-cloud.

Run from the repo root:
  .venv\\Scripts\\python scratch\\test_litellm_gemma3_12b_cloud.py

Requires in .env (or environment):
  OLLAMA_API_KEY   - your Ollama Cloud API key
  OLLAMA_BASE_URL  - Ollama Cloud base URL (e.g. https://api.ollama.com)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import litellm
from litellm import completion

_MODEL = "ollama/gemma3:12b-cloud"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

    print(f"Python:  {sys.executable}")
    print(f"litellm: {litellm.__file__}")
    print(f"Model:   {_MODEL}\n")

    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("ERROR: OLLAMA_API_KEY not set. Add it to .env or your environment.", file=sys.stderr)
        return 1

    base_url = os.environ.get("OLLAMA_BASE_URL")
    if base_url:
        print(f"OLLAMA_BASE_URL: {base_url}")
    else:
        print("OLLAMA_BASE_URL not set; using LiteLLM's default Ollama base URL.")

    kwargs: dict = {
        "model": _MODEL,
        "messages": [
            {"role": "user", "content": "Say exactly: 'gemma3:12b-cloud works via LiteLLM'"},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    if base_url:
        kwargs["api_base"] = base_url

    print("\n--- Sending completion request ---")
    try:
        resp = completion(**kwargs)
    except Exception as exc:
        print(f"\nFAIL - completion raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    content = (resp.choices[0].message.content or "").strip()
    usage = resp.usage

    print(f"Response: {content}")
    print(f"Usage:    prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
    print("\nOK - call succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
