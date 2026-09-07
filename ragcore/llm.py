"""Provider-agnostic streaming LLM client.

Everything downstream depends on one method — `stream(system, user) -> Iterator[str]`
— so the generation provider is a configuration choice, not an architectural
one. Three real providers are implemented plus a stub:

* **groq**    — free tier, OpenAI-compatible, very fast. The default.
* **gemini**  — free tier, different wire format, so it exercises the
                abstraction rather than pretending every provider is OpenAI.
* **openai**  — anything OpenAI-compatible: OpenAI itself, Together,
                OpenRouter, a local Ollama or vLLM server via `RAG_LLM_BASE_URL`.
* **echo**    — no network. Returns the retrieved context with citation
                markers so the retrieval half of the system can be
                demonstrated and tested without an API key.

Requests are retried on transient failures only. A 401 or a 400 is a
configuration error and retrying it just delays the useful error message.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class LLMError(RuntimeError):
    pass


class LLMConfigError(LLMError):
    """Bad key, bad model name, malformed request — do not retry."""


class LLMTransientError(LLMError):
    """Timeout, 429, 5xx — worth retrying."""


@runtime_checkable
class LLM(Protocol):
    model: str

    def stream(self, system: str, user: str) -> Iterator[str]: ...


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    body = resp.text[:400]
    if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
        raise LLMTransientError(f"{resp.status_code} from provider: {body}")
    raise LLMConfigError(f"{resp.status_code} from provider: {body}")


_RETRY = {
    "retry": retry_if_exception_type((LLMTransientError, httpx.TransportError)),
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=8),
    "reraise": True,
}


class OpenAICompatibleLLM:
    """Covers Groq, OpenAI, Together, OpenRouter, Ollama, vLLM."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
            raise LLMConfigError(
                "No API key set. Put RAG_LLM_API_KEY in your .env "
                "(see .env.example), or set RAG_LLM_PROVIDER=echo to run "
                "retrieval-only without a key."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    @retry(**_RETRY)
    def stream(self, system: str, user: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with (
            httpx.Client(timeout=self.timeout_s) as client,
            client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as resp,
        ):
            if resp.status_code >= 400:
                resp.read()
                _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta


class GeminiLLM:
    """Google Generative Language API — a different wire format on purpose.

    Included so the LLM abstraction is validated against something that is
    *not* OpenAI-shaped. Its streaming endpoint returns a JSON array delivered
    incrementally, which needs a small incremental parser rather than SSE.
    """

    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 900,
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMConfigError("RAG_LLM_API_KEY is required for provider=gemini.")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    @retry(**_RETRY)
    def stream(self, system: str, user: str) -> Iterator[str]:
        url = f"{self.BASE}/models/{self.model}:streamGenerateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        with (
            httpx.Client(timeout=self.timeout_s) as client,
            client.stream(
                "POST",
                url,
                json=payload,
                params={"key": self.api_key, "alt": "sse"},
                headers={"Content-Type": "application/json"},
            ) as resp,
        ):
            if resp.status_code >= 400:
                resp.read()
                _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                for cand in chunk.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if text := part.get("text"):
                            yield text


class EchoLLM:
    """Key-free stub. Streams a deterministic answer built from the context.

    This is not a toy for tests only — it is the mode the deployed demo falls
    back to if the LLM provider is unreachable, so a visitor still sees
    retrieval working instead of an error page.
    """

    model = "echo"

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield (
            "*(No generation provider configured — showing retrieved evidence only. "
            "Set RAG_LLM_PROVIDER and RAG_LLM_API_KEY to enable synthesis.)*\n\n"
        )
        body = user.split("QUESTION:")[0]
        shown = 0
        for block in body.split("\n\n"):
            block = block.strip()
            if block.startswith("[") and shown < 3:
                shown += 1
                marker = block.split("]")[0].lstrip("[")
                excerpt = " ".join(block.split()[1:60])
                yield f"Passage [{marker}]: {excerpt}…\n\n"
        if not shown:
            yield "No passages were retrieved for this question.\n"


def build_llm(settings) -> LLM:
    provider = (settings.llm_provider or "echo").lower()
    if provider == "echo":
        return EchoLLM()
    if provider == "groq":
        return OpenAICompatibleLLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or "https://api.groq.com/openai/v1",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "openai":
        return OpenAICompatibleLLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or "https://api.openai.com/v1",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "gemini":
        return GeminiLLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "gemini-2.0-flash",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    raise LLMConfigError(
        f"Unknown RAG_LLM_PROVIDER '{provider}'. Use groq, gemini, openai or echo."
    )
