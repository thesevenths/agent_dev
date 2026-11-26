"""ARR 0.1 / R3: thin, dependency-light LLM client for the research runtime.

This reuses the SAME OpenAI-compatible gateway the rest of the system already
uses (the ``LLM_*`` env vars defined in ``config.py``) -- no new vendor, no new
endpoint. It exists so the generic ``research_runtime`` can offer an LLM-backed
hypothesis proposer without importing the heavy agent-runtime ``llm.py`` (which
pulls in ``tools`` / ``hitl`` / ``runlog`` / ``graph`` and is not part of the
research runtime's contract).

``langchain_openai`` is imported LAZILY: a unit test that injects a fake
``generate`` callable (via ``ResearchDependencies.llm_generate``) never triggers
an SDK or network import, and the deterministic ``RuleBasedProposer`` remains the
fallback when no gateway is configured.
"""
from __future__ import annotations

import os
from typing import Any, Callable

# Mirror the defaults in config.py so behaviour is identical without importing it.
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"


def llm_config() -> dict[str, Any]:
    """Read the gateway config from the same env vars the agent runtime uses."""
    verify_ssl = os.getenv("LLM_VERIFY_SSL", "true").strip().lower() in (
        "1", "true", "yes", "y", "on")
    return {
        "base_url": os.getenv("LLM_BASE_URL", _DEFAULT_BASE_URL),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", _DEFAULT_MODEL),
        "verify_ssl": verify_ssl,
    }


def is_configured() -> bool:
    """Cheap prerequisite check: a non-empty API key is the only thing we can
    verify without a network call. Returns False in SDK-free test environments
    or when no key is set, which routes the proposer to the deterministic
    RuleBasedProposer.
    """
    return bool(os.getenv("LLM_API_KEY", "").strip())


def build_research_llm(temperature: float = 0.0):
    """Lazily build a ``ChatOpenAI`` bound to the project's ``LLM_*`` gateway.

    Raises ``RuntimeError`` if the SDK is unavailable or no key is configured,
    so callers can fall back to the deterministic proposer instead of crashing.
    """
    if not is_configured():
        raise RuntimeError("LLM_API_KEY is not set; research LLM proposer disabled")
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - exercised only when SDK missing
        raise RuntimeError(f"langchain_openai unavailable: {exc}") from exc
    cfg = llm_config()
    if not cfg["verify_ssl"]:
        import httpx
        sync_http = httpx.Client(verify=False, timeout=60)
        async_http = httpx.AsyncClient(verify=False, timeout=60)
        return ChatOpenAI(
            model=cfg["model"], temperature=temperature,
            api_key=cfg["api_key"], base_url=cfg["base_url"],
            http_client=sync_http, http_async_client=async_http,
            max_retries=2,
        )
    return ChatOpenAI(
        model=cfg["model"], api_key=cfg["api_key"], base_url=cfg["base_url"],
        temperature=temperature, timeout=60, max_retries=2,
    )


def make_generate(llm=None) -> Callable[[str], str]:
    """Wrap a ``ChatOpenAI`` (or build one) into the ``generate(prompt) -> str``
    callable the ``LLMProposer`` expects. Returns raw text; the proposer parses
    the JSON itself.
    """
    if llm is None:
        llm = build_research_llm()
    from langchain_core.messages import HumanMessage

    def generate(prompt: str) -> str:
        resp = llm.invoke([HumanMessage(content=prompt)])
        if isinstance(resp, str):
            return resp
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            # Multi-part content -> join the text parts only.
            return "".join(str(part) for part in content
                           if isinstance(part, str))
        return content if isinstance(content, str) else str(content)

    return generate
