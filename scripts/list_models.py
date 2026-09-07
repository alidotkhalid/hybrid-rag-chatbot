"""List the models your API key can actually call.

Provider documentation is not a reliable guide to what a given key may use:
Groq's public model table, for instance, includes Enterprise-only models that
a free-tier key cannot call, and a request for one fails at generation time
with a 404 that looks like a bug in your code. This asks the provider directly.

    python scripts/list_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from ragcore.config import settings  # noqa: E402

DEFAULT_BASES = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
}

# Models that exist but are not chat completions endpoints.
NON_CHAT = ("whisper", "tts", "guard", "embed", "distil-whisper", "playai")


def main() -> int:
    provider = (settings.llm_provider or "").lower()

    if provider == "echo":
        print("RAG_LLM_PROVIDER=echo — no provider to query.")
        return 0
    if provider == "gemini":
        print("Gemini model list: https://ai.google.dev/gemini-api/docs/models")
        return 0

    base = settings.llm_base_url or DEFAULT_BASES.get(provider)
    if not base:
        print(f"No base URL known for provider '{provider}'. Set RAG_LLM_BASE_URL.")
        return 1
    if not settings.llm_api_key:
        print("RAG_LLM_API_KEY is not set. Check your .env file.")
        return 1

    try:
        resp = httpx.get(
            f"{base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        print(f"Could not reach {base}: {exc}")
        return 1

    if resp.status_code == 401:
        print("401 Unauthorized — the API key in your .env is not valid.")
        return 1
    if resp.status_code != 200:
        print(f"{resp.status_code} from provider: {resp.text[:300]}")
        return 1

    models = sorted(m["id"] for m in resp.json().get("data", []))
    chat = [m for m in models if not any(x in m.lower() for x in NON_CHAT)]

    print(f"{len(models)} model(s) available to this key; {len(chat)} usable for chat.\n")
    print("Set RAG_LLM_MODEL in .env to one of these:\n")
    for m in chat:
        marker = "   <-- currently configured" if m == settings.llm_model else ""
        print(f"    {m}{marker}")

    if settings.llm_model not in models:
        print(
            f"\n  !! Your configured model '{settings.llm_model}' is NOT available "
            "to this key.\n     That is why generation fails with a 404. "
            "Pick one from the list above."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
