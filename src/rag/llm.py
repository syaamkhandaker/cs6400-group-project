"""Answer generation through a local Ollama server.

The model name lives in ``configs/default.yaml`` (``generation.model``).
Queries are HTTP calls to Ollama's chat API, by default
``http://127.0.0.1:11434``. Override the host with ``OLLAMA_HOST`` and the
model with ``OLLAMA_MODEL``.

Start the server and pull the model once before running QA::

    ollama serve
    ollama pull llama3.2
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rag.config import load_raw_config

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class LLMConfig:
    """Generation settings. ``model`` is the Ollama model we query."""

    provider: str = "ollama"
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    base_url: str = DEFAULT_BASE_URL


def _normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


def load_config(path: Path | None = None) -> LLMConfig:
    """Read ``generation`` from configs/default.yaml."""
    raw = load_raw_config(path)["generation"]
    provider = raw.get("provider", "ollama")
    if provider != "ollama":
        raise ValueError(f"Unsupported generation provider {provider!r}; expected 'ollama'.")

    base_url = os.environ.get("OLLAMA_HOST") or raw.get("base_url", DEFAULT_BASE_URL)
    return LLMConfig(
        provider=provider,
        model=os.environ.get("OLLAMA_MODEL") or raw.get("model", DEFAULT_MODEL),
        temperature=float(raw.get("temperature", 0.0)),
        base_url=_normalize_base_url(base_url),
    )


class OllamaLLM:
    """Query a local Ollama model for an answer.

    Construction::

        model = OllamaLLM.from_yaml()
        answer = model.generate("What does the paper report for Recall@10?")
    """

    def __init__(self, config: LLMConfig | None = None, timeout: float = 120):
        self.config = config or load_config()
        self.timeout = timeout

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> OllamaLLM:
        return cls(config=load_config(path))

    def generate(self, prompt: str, *, system_instruction: str | None = None) -> str:
        """Send ``prompt`` to Ollama and return the assistant text."""
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        url = f"{self.config.base_url}/api/chat"
        body = _post_json(url, payload, self.timeout)
        message = body.get("message") or {}
        return message.get("content") or ""


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama returned {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {url}. Start it with `ollama serve`."
        ) from exc
