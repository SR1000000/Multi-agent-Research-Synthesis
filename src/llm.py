from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, Any
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client
from google import genai
from pydantic import BaseModel
from google.genai import types

import re
import time
import random
from functools import wraps

T = TypeVar("T", bound=BaseModel)

DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
DEFAULT_OLLAMA_MODEL = "qwen3.5:397b-cloud"
DEFAULT_GEMINI_MODEL="gemini-3.1-flash-lite-preview"  # 3.1 has higher RPD than 2.5 on free tier for some reason

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))

def with_retry(max_retries=3, initial_wait=60, backoff_factor=1.5):
    """
    Decorator to retry LLM calls automatically upon hitting a 429
    (Resource Exhausted / Rate Limit) error. Uses exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait = initial_wait
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    # Check if it's a rate limit error (429)
                    if "429" in err_str or "resource_exhausted" in err_str:
                        if attempt == max_retries:
                            raise
                        # Add a small randomness to avoid thundering herd
                        sleep_time = wait + random.uniform(0, 2)
                        print(f"[llm] Rate limit hit (429). Retrying in {sleep_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                        wait *= backoff_factor
                    else:
                        raise
        return wrapper
    return decorator

def _strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3).
    Also strips any leading/trailing whitespace so the remainder is clean JSON.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fence(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from LLM output.
    Some models emit these even when constrained to JSON mode via format/schema params.
    Falls back to extracting the first complete JSON object/array if fences aren't present.
    """
    # Fast path: single code fence wrapping the entire response
    fenced = re.sub(r"^```(?:json)?\s*\n?(.*?)\n?```$", r"\1", text, flags=re.DOTALL).strip()
    if fenced != text:
        return fenced

    # Fallback: prose preamble before the JSON — find the outermost { } or [ ]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

    return text

class Provider(str, Enum):
    OPENROUTER        = "openrouter"
    OLLAMA            = "ollama"
    GOOGLE_AI_STUDIO  = "google_ai_studio"

# For now we will use provider, model, base url and api key fields. Other fields will be updated once a need is found
@dataclass
class LLMConfig:
    provider: Provider | str = Provider.OLLAMA
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    think: bool | None = True
    base_url: str | None = None
    api_key: str | None = None

    def resolved_api_key(self, env_var: str) -> str:
        if self.api_key:
            return self.api_key
        key = os.environ.get(env_var)
        if not key:
            raise EnvironmentError(f"Missing API key: set LLMConfig.api_key or {env_var}")
        return key



class OpenRouterLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.resolved_api_key("OPENROUTER_API_KEY"),
            base_url=config.base_url,
        )

    #@with_retry()
    def complete(self, messages: list[dict], schema: type[T] | None = None, **kwargs) -> str | T:
        req_params = {
            "model": self.config.model,
            "messages": messages,
        }
        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            req_params["temperature"] = temp
            
        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            req_params["max_tokens"] = mt

        if schema is not None:
            # Prepare the schema for OpenAI-style structured outputs
            json_schema = schema.model_json_schema()
            
            def _make_strict(d: Any) -> None:
                if isinstance(d, dict):
                    if d.get("type") == "object":
                        d["additionalProperties"] = False
                        if "properties" in d:
                            # In strict mode, all properties must be in 'required'
                            d["required"] = list(d["properties"].keys())
                    for v in d.values():
                        _make_strict(v)
                elif isinstance(d, list):
                    for item in d:
                        _make_strict(item)

            _make_strict(json_schema)

            req_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": json_schema
                }
            }

        try:
            resp = self._client.chat.completions.create(**req_params)
        except Exception as e:
            # Fallback for models that don't support structured outputs (json_schema)
            if schema is not None and any(x in str(e).lower() for x in ["structured_outputs", "json_schema", "response_format"]):
                # Try falling back to basic JSON object mode
                req_params["response_format"] = {"type": "json_object"}
                
                # Nudge the model by appending to the last user message
                # We copy to avoid mutating the original message list for potential retries
                msgs_copy = list(messages)
                if msgs_copy:
                    last_msg = msgs_copy[-1].copy()
                    last_msg["content"] = str(last_msg.get("content", "")) + "\n\nIMPORTANT: You must respond with a valid JSON object matching the requested schema."
                    msgs_copy[-1] = last_msg
                    req_params["messages"] = msgs_copy
                
                try:
                    resp = self._client.chat.completions.create(**req_params)
                except Exception as fallback_err:
                    raise NotImplementedError(
                        f"Structured output (json_schema) and JSON mode not supported for model {self.config.model}. "
                        f"Original error: {e}. Fallback error: {fallback_err}"
                    ) from fallback_err
            else:
                raise

        msg = resp.choices[0].message
        raw = msg.content
        
        # OpenRouter / OpenAI SDK might separate reasoning for models like DeepSeek-R1
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if not reasoning and hasattr(msg, "model_extra") and msg.model_extra:
             reasoning = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")
             
        if reasoning:
            raw = f"<think>\n{reasoning}\n</think>\n\n{raw or ''}"
            
        return raw

class OllamaLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.resolved_api_key("OLLAMA_API_KEY")
        self._client = Client(
            host=config.base_url,
            headers={'Authorization': f'Bearer {api_key}'}
        )

    #@with_retry()
    def complete(self, messages: list[dict], schema: type[T] | None = None, **kwargs) -> str | T:
        """Invoke Ollama chat. If schema is provided, constrain output to JSON schema."""
        options = {}
        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            options["temperature"] = temp

        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            options["num_predict"] = mt

        req_params = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        if options:
            req_params["options"] = options

        if schema is not None:
            req_params["format"] = schema.model_json_schema()

        think_opt = kwargs.get("think", getattr(self.config, "think", True))
        if think_opt is not None:
            req_params["think"] = think_opt

        resp = self._client.chat(**req_params)
        raw_content = resp.message.content
        
        thinking_text = getattr(resp.message, "thinking", None)
        if thinking_text:
            raw_content = f"<think>\n{thinking_text}\n</think>\n\n{raw_content}"
        return raw_content

class GeminiLLM:
    """Google AI Studio provider via the google-genai SDK."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = genai.Client(
            api_key=config.resolved_api_key("GOOGLE_AI_STUDIO_API_KEY"),
        )

    @with_retry()
    def complete(self, messages: list[dict], schema: type[T] | None = None, **kwargs) -> str | T:
        system_instructions = []
        user_contents = []

        for m in messages:
            role = m.get('role', '')
            content_text = str(m.get('content', ''))
            
            if role == 'system':
                system_instructions.append(content_text)
            else:
                # Gemini expects 'model' for assistant, 'user' for user
                gemini_role = 'model' if role == 'assistant' else 'user'
                user_contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=content_text)]
                    )
                )

        gen_config = types.GenerateContentConfig()
        
        if system_instructions:
            gen_config.system_instruction = "\n".join(system_instructions)

        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            gen_config.temperature = temp

        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            gen_config.max_output_tokens = mt

        if schema is not None:
            gen_config.response_mime_type = "application/json"
            gen_config.response_json_schema = schema.model_json_schema()

        think_opt = kwargs.get("think", getattr(self.config, "think", True))
        if think_opt:
            gen_config.thinking_config = types.ThinkingConfig(include_thoughts=True)

        response = self._client.models.generate_content(
            model=self.config.model,
            contents=user_contents,
            config=gen_config,
        )

        thinking_text = ""
        answer_text = ""
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thinking_text += part.text + "\n"
            else:
                answer_text += part.text + "\n"

        if thinking_text:
            raw_content = f"<think>\n{thinking_text.strip()}\n</think>\n\n{answer_text.strip()}"
        else:
            raw_content = answer_text.strip()
            
        return raw_content

_PROVIDERS: dict[Provider, dict] = {
    Provider.OPENROUTER: {
        "cls": OpenRouterLLM,
        "defaults": {
            "model": DEFAULT_OPENROUTER_MODEL,
            "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        }
    },
    Provider.OLLAMA: {
        "cls": OllamaLLM,
        "defaults": {
            "model": DEFAULT_OLLAMA_MODEL,
            "base_url": os.environ.get("OLLAMA_BASE_URL", "https://ollama.com"),
        }
    },
    Provider.GOOGLE_AI_STUDIO: {
        "cls": GeminiLLM,
        "defaults": {
            "model": DEFAULT_GEMINI_MODEL,
        }
    },
}

GLOBAL_CONFIG = LLMConfig()

def get_llm(config: LLMConfig | None = None) -> OpenRouterLLM | OllamaLLM | GeminiLLM:
    if config is None:
        config = GLOBAL_CONFIG

    provider = Provider(config.provider)

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Valid: {list(_PROVIDERS)}")

    for field_name, default_val in _PROVIDERS[provider]["defaults"].items():
        if getattr(config, field_name) is None:
            setattr(config, field_name, default_val)

    return _PROVIDERS[provider]["cls"](config)
