from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..qa_generator import (
    DEFAULT_MAX_RETRIES,
    MIN_REQUEST_INTERVAL_SECONDS,
)

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    return default


@dataclass
class AgentGenerationConfig:
    api_key: str
    base_url: str
    agent_question_generate_model: str
    classify_model: str
    core_agent_answer_generate_model: str
    review_model: str
    question_count: int = 20
    max_questions_per_context: int = 4
    top_k_tools: int = 5
    max_turns: int = 8
    langgraph_retry: int = 1
    checkpoint_path: Optional[str] = None
    resume_from_checkpoint: bool = True
    max_retries: int = DEFAULT_MAX_RETRIES
    min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS
    question_temperature: float = 0.5
    classify_temperature: float = 0.3
    agent_temperature: float = 0.7
    review_temperature: float = 0.2
    max_workers: int = 4
    debug: bool = False
    agent_backend: str = "langgraph"
    auth: Optional[Dict[str, Any]] = None
    default_tool_server: Optional[str] = None
    tool_request_timeout: float = 30.0
    require_auth_for_protected_tools: bool = True


@dataclass
class AgentScriptSettings:
    agent_backend: str
    agent_question_model: Optional[str]
    agent_classify_model: Optional[str]
    agent_answer_model: Optional[str]
    agent_review_model: Optional[str]
    question_count: int
    max_qps: float
    top_k_tools: int
    max_turns: int
    max_workers: int
    max_retries: int
    max_questions_per_context: int
    langgraph_retry: int
    resume_from_checkpoint: bool
    debug: bool
    min_request_interval: Optional[float]
    default_tool_server: Optional[str]
    tool_request_timeout: Optional[float]
    require_auth: bool


def load_agent_script_settings_from_env(
    default_question_model: Optional[str] = None,
) -> AgentScriptSettings:
    backend_default = os.getenv("AGENT_BACKEND", "openai") or "openai"
    agent_backend = backend_default.strip().lower() or "langgraph"

    fallback_model = default_question_model or "qwen3-max"
    question_model = os.getenv("AGENT_QUESTION_MODEL") or fallback_model
    classify_model = os.getenv("AGENT_CLASSIFY_MODEL") or fallback_model
    answer_model = os.getenv("AGENT_ANSWER_MODEL") or fallback_model
    review_model = os.getenv("AGENT_REVIEW_MODEL") or fallback_model

    question_count = max(1, _env_int("AGENT_QUESTION_COUNT", 20))
    max_qps = max(0.0, _env_float("AGENT_MAX_QPS", 10.0))
    top_k_tools = max(1, _env_int("AGENT_TOP_K_TOOLS", 5))
    max_turns = max(1, _env_int("AGENT_MAX_TURNS", 8))
    max_workers = max(1, _env_int("AGENT_MAX_WORKERS", 4))
    max_retries = max(1, _env_int("AGENT_MAX_RETRIES", 3))
    max_questions_per_context = max(1, _env_int("AGENT_MAX_QUESTIONS_PER_CONTEXT", 4))
    langgraph_retry = max(0, _env_int("AGENT_LANGGRAPH_RETRY", 1))
    resume_from_checkpoint = _env_bool("AGENT_RESUME_FROM_CHECKPOINT", True)
    debug = _env_bool("AGENT_DEBUG", True)

    min_request_interval: Optional[float] = None
    override = os.getenv("AGENT_MIN_REQUEST_INTERVAL")
    if override and override.strip():
        try:
            min_request_interval = float(override)
        except ValueError:
            min_request_interval = None
    if min_request_interval is None and max_qps > 0:
        min_request_interval = 1.0 / max_qps

    default_tool_server = os.getenv("AGENT_DEFAULT_TOOL_SERVER")
    tool_request_timeout = _env_float("AGENT_TOOL_TIMEOUT", 30.0)
    require_auth = _env_bool("AGENT_REQUIRE_AUTH", True)

    return AgentScriptSettings(
        agent_backend=agent_backend,
        agent_question_model=question_model,
        agent_classify_model=classify_model,
        agent_answer_model=answer_model,
        agent_review_model=review_model,
        question_count=question_count,
        max_qps=max_qps,
        top_k_tools=top_k_tools,
        max_turns=max_turns,
        max_workers=max_workers,
        max_retries=max_retries,
        max_questions_per_context=max_questions_per_context,
        langgraph_retry=langgraph_retry,
        resume_from_checkpoint=resume_from_checkpoint,
        debug=debug,
        min_request_interval=min_request_interval,
        default_tool_server=default_tool_server,
        tool_request_timeout=tool_request_timeout,
        require_auth=require_auth,
    )


__all__ = [
    "AgentGenerationConfig",
    "AgentScriptSettings",
    "load_agent_script_settings_from_env",
]
