import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

class AgentLogger:
    """
    AgentLogger handles integration with Langfuse to provide observability 
    for the multi-agent graph and LLM responses natively.
    """
    def __init__(self):
        # Langfuse automatically grabs LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, 
        # and LANGFUSE_BASE_URL from environment variables.
        self.client = Langfuse()
        self._logger = logging.getLogger("agentic_ai")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def get_langgraph_handler(self, **kwargs) -> CallbackHandler:
        """
        Returns a Langfuse CallbackHandler for LangGraph. 
        Hooks directly into the graph's config to trace node execution.
        """
        return CallbackHandler(**kwargs)

    def flush(self):
        """
        Flushes queued events to the Langfuse backend. 
        Should be called before the process exits.
        """
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
