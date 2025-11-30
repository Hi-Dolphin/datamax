from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin

from loguru import logger
import requests

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = "__langgraph_end__"
    StateGraph = None

from datamax.utils.performance_monitor import PerformanceMonitor

from .agent import (
    AgentGenerationConfig,
    AgentScriptSettings,
    load_agent_script_settings_from_env,
)
from .agent.classifier import ToolClassifier
from .agent.models import (
    AgentEpisode,
    AgentQuestion,
    AgentTurn,
    ApiEndpoint,
    ApiSpec,
    PromptContext,
    ReviewResult,
    ToolCall,
    ToolCandidate,
    ToolSpec,
)
from .agent.progress import AgentProgressTracker
from .agent.questions import AgentQuestionGenerator
from .agent.review import AgentReviewPipeline
from .agent.runners import (
    LangGraphAgent,
    OpenAIAgent,
    ToolRegistry,
    build_agent_plan,
    generate_agent_final_answer,
)
from .agent.spec import ApiGraph, ApiSpecLoader
from .qa_generator import (
    QAProgressTracker,
    extract_json_from_llm_output,
    llm_generator,
)
from .auth import AuthManager, AuthContext

SPEC_EXTENSIONS: Tuple[str, ...] = (".json", ".yaml", ".yml")
# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class AgentTrainingDataGenerator:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config
        self.loader = ApiSpecLoader()
        self.monitor = PerformanceMonitor()
        self.question_generator = AgentQuestionGenerator(config)
        self.classifier = ToolClassifier(config)
        self.review_pipeline = AgentReviewPipeline(config)

    def run(self, spec_sources: Sequence[Union[str, dict]]) -> dict:
        specs = self.loader.load(spec_sources)
        if not specs:
            raise ValueError("No API specifications could be loaded; aborting generation.")

        api_graph = ApiGraph(specs)
        tool_catalog = api_graph.tool_catalog()
        if self.config.require_auth_for_protected_tools and not self.config.auth:
            protected_schemes = sorted(
                {
                    scheme
                    for tool in tool_catalog
                    for requirement in tool.security
                    if isinstance(requirement, dict)
                    for scheme in requirement.keys()
                }
            )
            if protected_schemes:
                raise RuntimeError(
                    "Authentication is required for the loaded API specifications. "
                    "Provide credentials for the following security schemes via AgentGenerationConfig.auth: "
                    + ", ".join(protected_schemes)
                )
        auth_manager = AuthManager(self.config.auth)
        tool_registry = ToolRegistry(
            tool_catalog,
            auth_manager=auth_manager,
            default_server=self.config.default_tool_server,
            timeout=self.config.tool_request_timeout,
            require_auth=self.config.require_auth_for_protected_tools,
        )
        backend = (self.config.agent_backend or "langgraph").lower()
        if backend == "langgraph":
            agent = LangGraphAgent(self.config, tool_registry)
        elif backend == "openai":
            agent = OpenAIAgent(self.config, tool_registry)
        else:
            raise ValueError(f"Unsupported agent backend: {self.config.agent_backend}")

        tracker: Optional[AgentProgressTracker] = None
        if self.config.checkpoint_path:
            tracker = AgentProgressTracker(
                self.config.checkpoint_path, resume=self.config.resume_from_checkpoint
            )

        questions = self.question_generator.generate(api_graph, self.monitor)
        episodes: List[AgentEpisode] = []
        for question in questions:
            episode_id = question.question_id
            if tracker and episode_id in tracker.entries_by_key:
                continue
            candidates = self.classifier.classify(question, tool_catalog, self.monitor)
            turns, tool_calls, final_answer = agent.run(question, candidates, self.monitor)
            episode = AgentEpisode(
                episode_id=episode_id,
                question=question,
                tool_candidates=candidates,
                tool_calls=tool_calls,
                turns=turns,
                final_answer=final_answer,
                agent_name=agent.agent_name,
                metadata={
                    "difficulty": question.difficulty,
                    "agent_backend": backend,
                },
            )
            review = self.review_pipeline.review(episode, self.monitor)
            episode.review = review
            episodes.append(episode)
            if tracker:
                tracker.record(episode.to_dict())

        result = {
            "episodes": [episode.to_dict() for episode in episodes],
            "tool_catalog": [asdict(tool) for tool in tool_catalog],
            "questions": [question.to_dict() for question in questions],
            "metadata": {
                "spec_count": len(specs),
                "question_count": len(questions),
                "episode_count": len(episodes),
                "max_turns": self.config.max_turns,
                "agent_backend": backend,
            },
            "performance": self.monitor.build_report(),
        }
        return result


def generate_agent_training_data(
    spec_sources: Sequence[Union[str, dict]],
    config: AgentGenerationConfig,
) -> dict:
    generator = AgentTrainingDataGenerator(config)
    return generator.run(spec_sources)


def discover_spec_files(directory: Path, extensions: Sequence[str] = SPEC_EXTENSIONS) -> List[Path]:
    """Recursively discover API specification files under ``directory``."""
    if not directory.exists():
        return []

    files: Dict[Path, Path] = {}
    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            if path.is_file():
                relative = try_relative(path, directory)
                files.setdefault(relative, path)

    return [files[key] for key in sorted(files)]


def try_relative(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def remove_all_suffixes(path: Path) -> Path:
    result = path
    while result.suffix:
        try:
            result = result.with_suffix("")
        except ValueError:
            break
    return result


def split_relative_path(spec_path: Path, base_dir: Path) -> tuple[Path, str]:
    relative = try_relative(spec_path, base_dir)
    base = remove_all_suffixes(relative)
    return base.parent, base.name


def make_agent_output_stem(spec_path: Path, spec_root: Path, output_root: Path) -> Path:
    relative_dir, stem = split_relative_path(spec_path, spec_root)
    output_dir = output_root / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / stem


def make_agent_checkpoint_path(spec_path: Path, spec_root: Path, checkpoint_root: Path) -> Path:
    relative_dir, stem = split_relative_path(spec_path, spec_root)
    checkpoint_dir = checkpoint_root / relative_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{stem}.jsonl"


def build_agent_mode_config_for_spec(
    spec_path: Path,
    settings: AgentScriptSettings,
    *,
    checkpoint_path: Optional[Path] = None,
    auth_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Construct the agent_mode configuration payload for ``DataMax``."""
    config: Dict[str, Any] = {
        "enabled": True,
        "agent_backend": settings.agent_backend,
        "spec_sources": [str(spec_path)],
        "question_count": settings.question_count,
        "max_questions_per_context": settings.max_questions_per_context,
        "top_k_tools": settings.top_k_tools,
        "max_turns": settings.max_turns,
        "max_workers": settings.max_workers,
        "max_retries": settings.max_retries,
        "langgraph_retry": settings.langgraph_retry,
        "resume_from_checkpoint": settings.resume_from_checkpoint,
    }

    if checkpoint_path:
        config["checkpoint_path"] = str(checkpoint_path)

    if settings.min_request_interval and settings.min_request_interval > 0:
        config["min_request_interval_seconds"] = settings.min_request_interval

    if settings.agent_question_model:
        config["agent_question_generate_model"] = settings.agent_question_model
    if settings.agent_classify_model:
        config["classify_model"] = settings.agent_classify_model
    if settings.agent_answer_model:
        config["core_agent_answer_generate_model"] = settings.agent_answer_model
    if settings.agent_review_model:
        config["review_model"] = settings.agent_review_model

    if auth_config:
        config["auth"] = json.loads(json.dumps(auth_config))
    if settings.default_tool_server:
        config["default_tool_server"] = settings.default_tool_server
    if settings.tool_request_timeout is not None:
        config["tool_request_timeout"] = settings.tool_request_timeout

    config["require_auth_for_protected_tools"] = settings.require_auth
    return config


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def format_function_call_payload(tool_name: str, arguments: Any) -> str:
    payload = {"name": tool_name, "arguments": arguments}
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return json.dumps({"name": tool_name, "arguments": None}, ensure_ascii=False)


def episode_to_sharegpt(
    episode: Dict[str, Any],
    tool_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    conversations: List[Dict[str, str]] = []
    question = episode.get("question") or {}

    prompt = ensure_text(
        question.get("prompt")
        or question.get("question")
        or next(
            (
                turn.get("content")
                for turn in (episode.get("turns") or [])
                if isinstance(turn, dict)
                and turn.get("role") == "user"
                and turn.get("content")
            ),
            "",
        )
    )
    conversations.append({"from": "human", "value": prompt})

    used_tools: List[str] = []
    for call in episode.get("tool_calls") or []:
        tool_name = ensure_text(call.get("tool_name") or "tool")
        conversations.append(
            {
                "from": "function_call",
                "value": format_function_call_payload(tool_name, call.get("input")),
            }
        )
        conversations.append(
            {
                "from": "observation",
                "value": ensure_text(call.get("observation")),
            }
        )
        if tool_name not in used_tools:
            used_tools.append(tool_name)

    final_answer = ensure_text(episode.get("final_answer"))
    if not final_answer:
        fallback = next(
            (
                turn.get("content")
                for turn in reversed(episode.get("turns") or [])
                if isinstance(turn, dict)
                and turn.get("role") == "assistant"
                and turn.get("content")
                and not turn.get("tool_name")
            ),
            "",
        )
        final_answer = ensure_text(fallback)

    conversations.append({"from": "gpt", "value": final_answer})

    entry: Dict[str, Any] = {"conversations": conversations}
    if used_tools:
        tool_specs = [tool_lookup[name] for name in used_tools if name in tool_lookup]
        if tool_specs:
            entry["tools"] = tool_specs

    metadata = episode.get("metadata")
    backend = metadata.get("agent_backend") if isinstance(metadata, dict) else None
    if backend:
        entry["system"] = f"Agent backend: {backend}"

    return entry


def convert_episodes_to_sharegpt(
    episodes: Sequence[Dict[str, Any]],
    tool_catalog: Iterable[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    tool_lookup: Dict[str, Dict[str, Any]] = {}
    if tool_catalog:
        for tool in tool_catalog:
            if isinstance(tool, dict) and tool.get("name"):
                tool_lookup[str(tool["name"])] = tool
    return [episode_to_sharegpt(ep, tool_lookup) for ep in episodes if isinstance(ep, dict)]


def validate_auth_configuration_for_spec(spec_path: Path, auth_config: Optional[dict]) -> None:
    if not auth_config:
        return
    loader = ApiSpecLoader()
    specs = loader.load([str(spec_path)])
    if not specs:
        return
    manager = AuthManager(auth_config)
    for spec in specs:
        for endpoint in spec.endpoints:
            tool_spec = endpoint.to_tool_spec()
            manager.get_context(tool_spec)
