"""M6.1-fix-B / M7 -- REAL LLM Agent Provider Adapter.

The M6.1 deterministic planner is retained for ENGINE-FREE TESTS only; it is
never presented as a real LLM agent. This adapter performs an actual HTTP call
to an OpenAI-compatible gateway (the same LLM_* vars the project already uses)
and records EVERY identity/telemetry field the acceptance spec requires:

  provider, model id, model version, deployment, prompt SHA, toolset SHA,
  temperature, prompt/completion/total tokens, request-id, response-id,
  latency, cost, tool calls, errors/retries, full trajectory.

If no valid credentials are configured, ``is_available()`` returns False and
callers MUST mark the gate BLOCKED -- this adapter never fabricates a model
identity or a response.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AGENT_ADAPTER_VERSION = "m6_1_llm_agent_adapter_v1"


class AgentBlocked(RuntimeError):
    """Raised when no real model credentials are available."""


@dataclass
class AgentTrajectory:
    trajectory_id: str
    provider: str
    model_id: str
    model_version: str
    model_deployment: str
    prompt_sha256: str
    toolset_sha256: str
    temperature: float
    request_id: str
    response_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost: float
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retries: int = 0
    content: str = ""
    ok: bool = False
    adapter_version: str = AGENT_ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config() -> dict[str, Any]:
    # Load .env without requiring python-dotenv in production paths.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith(("LLM_", "DASHSCOPE_")) and k not in os.environ:
                os.environ[k] = v.strip().strip('"\'')
    return {"base_url": os.getenv("LLM_BASE_URL", ""),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model": os.getenv("LLM_MODEL", ""),
            "verify_ssl": os.getenv("LLM_VERIFY_SSL", "true").strip().lower()
            in ("1", "true", "yes", "y", "on")}


def is_available() -> bool:
    cfg = _config()
    return bool(cfg["api_key"].strip() and cfg["base_url"].strip()
                and cfg["model"].strip())


class LLMAgentAdapter:
    """Real model calls with full trajectory recording. No FakeRunner path."""

    def __init__(self, *, max_retries: int = 2, timeout_s: float = 60.0,
                 price_per_1k_tokens: float = 0.0):
        self.cfg = _config()
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.price_per_1k = price_per_1k_tokens
        self.calls: list[AgentTrajectory] = []

    def _provider(self) -> str:
        url = self.cfg["base_url"]
        if "dashscope" in url:
            return "dashscope-openai-compatible"
        if "vllm" in url or ":8000" in url:
            return "vllm-openai-compatible"
        return "openai-compatible"

    def call(self, *, prompt: str, system: str = "",
             toolset: list[str] | None = None,
             temperature: float = 0.0,
             max_tokens: int = 1024,
             response_format_json: bool = False) -> AgentTrajectory:
        if not is_available():
            raise AgentBlocked("no real LLM credentials; gate must be BLOCKED")
        tools = sorted(toolset or [])
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        toolset_sha = hashlib.sha256(
            json.dumps(tools, sort_keys=True).encode()).hexdigest()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": self.cfg["model"],
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": max_tokens}
        if response_format_json:
            body["response_format"] = {"type": "json_object"}
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        errors: list[str] = []
        retries = 0
        started = time.time()
        content = ""
        req_id = resp_id = ""
        pt = ct = 0
        tool_calls: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retries = attempt
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(body).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.cfg['api_key']}",
                             "Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                    req_id = r.headers.get("x-request-id", "") or str(
                        resp.get("id", ""))
                resp_id = str(resp.get("id", ""))
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = str(msg.get("content") or "")
                tool_calls = list(msg.get("tool_calls") or [])
                usage = resp.get("usage") or {}
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - record, retry, never fake
                errors.append(f"{type(exc).__name__}: {exc}")
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
        latency = (time.time() - started) * 1000.0
        total = pt + ct
        traj = AgentTrajectory(
            trajectory_id=hashlib.sha256(
                f"{prompt_sha}|{self.cfg['model']}|{retries}".encode()).hexdigest(),
            provider=self._provider(), model_id=self.cfg["model"],
            model_version=str(self.cfg["model"]),  # vLLM exposes no separate version
            model_deployment=self.cfg["base_url"].rstrip("/"),
            prompt_sha256=prompt_sha, toolset_sha256=toolset_sha,
            temperature=temperature, request_id=req_id, response_id=resp_id,
            prompt_tokens=pt, completion_tokens=ct, total_tokens=total,
            latency_ms=round(latency, 2),
            estimated_cost=round(total / 1000.0 * self.price_per_1k, 6),
            tool_calls=tool_calls, errors=errors, retries=retries,
            content=content, ok=(last_exc is None and bool(content)))
        self.calls.append(traj)
        return traj

    def export_trajectories(self, path: str | Path) -> dict[str, Any]:
        from research_runtime.harness_rsi.identity import freeze_json
        return freeze_json({"adapter_version": AGENT_ADAPTER_VERSION,
                            "config": {k: ("<set>" if k == "api_key" and v else v)
                                       for k, v in self.cfg.items()},
                            "trajectories": [t.to_dict() for t in self.calls]},
                           path)


__all__ = ["AGENT_ADAPTER_VERSION", "AgentBlocked", "AgentTrajectory",
           "LLMAgentAdapter", "is_available"]
