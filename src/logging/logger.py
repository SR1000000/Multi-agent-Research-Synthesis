import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

VALIDATION_ERRORS_DIR = Path(__file__).parent.parent.parent / "validation_errors"

class AgentLogger:
    """
    AgentLogger handles integration with Langfuse to provide observability 
    for the multi-agent graph and LLM responses natively.

    Implemented as a singleton so that repeated ``AgentLogger()`` calls across
    agents and modules share a single Langfuse client and Python logger instance.
    """
    _instance: "AgentLogger | None" = None

    def __new__(cls) -> "AgentLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        # Langfuse automatically grabs LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, 
        # and LANGFUSE_BASE_URL from environment variables.
        # Langfuse Python SDK v2 reads LANGFUSE_HOST for the API URL, not LANGFUSE_BASE_URL.
        self.client = Langfuse()
        self._logger = logging.getLogger("agentic_ai")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handlers = []

    def get_langgraph_handler(self, **kwargs) -> CallbackHandler:
        """
        Returns a Langfuse CallbackHandler for LangGraph. 
        Hooks directly into the graph's config to trace node execution.
        """
        handler = CallbackHandler(**kwargs)
        self._handlers.append(handler)
        return handler

    def flush(self):
        """
        Flushes queued events to the Langfuse backend. 
        Should be called before the process exits.
        """
        # Flush any created CallbackHandlers
        if hasattr(self, "_handlers"):
            for h in self._handlers:
                h.flush()
        
        # Flush traces from @observe decorators
        from langfuse.decorators import langfuse_context
        langfuse_context.flush()
        
        # Flush the main client
        self.client.flush()

    def log(self, message: str, level: str = "info") -> None:
        level_name = level.lower()
        if level_name == "debug":
            self._logger.debug(message)
        elif level_name == "warning":
            self._logger.warning(message)
        elif level_name == "error":
            self._logger.error(message)
        else:
            self._logger.info(message)

    def dump_json_artifact(
        self,
        file_name: str,
        payload: Any,
        subdir: str = "artifacts",
        run_id: str | None = None,
    ) -> str | None:
        try:
            base_dir = Path(subdir)
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = self._resolve_artifact_path(base_dir, file_name, run_id=run_id)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            return str(output_path)
        except Exception:
            return None

    def dump_validation_error(
        self,
        agent_display_name: str,
        attempt: int,
        max_attempts: int,
        validation_error: Exception,
        offending_json: str,
        model: str | None = None,
    ) -> Path | None:
        """Write a structured validation-error dump to VALIDATION_ERRORS_DIR.

        Returns the Path of the written file, or None on failure.
        """
        try:
            VALIDATION_ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            safe_agent = self._sanitize_file_part(agent_display_name) or "agent"
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{safe_agent}_{attempt + 1}of{max_attempts}_{ts}.json"
            out_path = VALIDATION_ERRORS_DIR / filename
            payload = {
                "agent": agent_display_name,
                "model": model,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_summary": str(validation_error),
                "offending_json": offending_json,
            }
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return out_path
        except Exception:
            return None

    def _resolve_artifact_path(
        self,
        base_dir: Path,
        file_name: str,
        run_id: str | None = None,
    ) -> Path:
        requested = Path(file_name)
        stem = requested.stem
        suffix = requested.suffix or ".json"
        safe_run_id = self._sanitize_file_part(run_id) if run_id else ""

        if safe_run_id:
            candidate = base_dir / f"{stem}_{safe_run_id}{suffix}"
            if not candidate.exists():
                return candidate
            idx = 1
            while True:
                candidate = base_dir / f"{stem}_{safe_run_id}_{idx}{suffix}"
                if not candidate.exists():
                    return candidate
                idx += 1

        candidate = base_dir / file_name
        if not candidate.exists():
            return candidate
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base_dir / f"{stem}_{ts}{suffix}"
        if not candidate.exists():
            return candidate
        idx = 1
        while True:
            candidate = base_dir / f"{stem}_{ts}_{idx}{suffix}"
            if not candidate.exists():
                return candidate
            idx += 1

    def _sanitize_file_part(self, value: str | None) -> str:
        if not value:
            return ""
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
