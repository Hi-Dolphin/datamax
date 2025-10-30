from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from loguru import logger

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = '__langgraph_end__'
    StateGraph = None

from datamax.utils.performance_monitor import PerformanceMonitor

from ..auth import AuthContext, AuthManager
from ..qa_generator import extract_json_from_llm_output, llm_generator
from .config import AgentGenerationConfig
from .models import AgentQuestion, AgentTurn, ToolCall, ToolCandidate, ToolSpec

# ---------------------------------------------------------------------------
# LangGraph-based agent execution
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(
        self,
        tool_specs: Sequence[ToolSpec],
        *,
        auth_manager: Optional[AuthManager] = None,
        default_server: Optional[str] = None,
        timeout: float = 30.0,
        require_auth: bool = True,
    ):
        self._by_name: Dict[str, ToolSpec] = {spec.name: spec for spec in tool_specs}
        self._auth_manager = auth_manager or AuthManager(None)
        self._default_server = default_server
        self._timeout = timeout
        self._require_auth = require_auth

    def resolve(self, name: str) -> Optional[ToolSpec]:
        return self._by_name.get(name)

    def list_tool_names(self) -> List[str]:
        return list(self._by_name.keys())

    def specs(self) -> List[ToolSpec]:
        return list(self._by_name.values())

    def simulate_invocation(self, tool: ToolSpec, params: Dict[str, Any]) -> str:
        fields = []
        schema = tool.output_schema or {}
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict):
            fields = list(properties.keys())
        dominant_fields = ", ".join(fields[:6]) if fields else "unspecified fields"
        param_desc = json.dumps(params, ensure_ascii=False) if params else "{}"
        return (
            f"Simulated response from {tool.method.upper()} {tool.path}. "
            f"Parameters: {param_desc}. Likely returns {dominant_fields}."
        )

    def invoke(
        self,
        tool: ToolSpec,
        params: Optional[Dict[str, Any]],
        monitor: Optional[PerformanceMonitor] = None,
    ) -> Tuple[str, bool, Optional[str], Optional[float]]:
        observation, success, error, latency = self._execute_request(tool, params or {})
        if monitor and latency is not None:
            monitor.record_request(stage="tool_invocation", duration_seconds=latency)
        return observation, success, error, latency

    def _execute_request(
        self,
        tool: ToolSpec,
        params: Dict[str, Any],
    ) -> Tuple[str, bool, Optional[str], Optional[float]]:
        if not isinstance(params, dict):
            raise TypeError(
                f"Tool invocation expects parameters as an object, received {type(params).__name__}."
            )
        method, url, headers, query_params, json_payload, data_payload = self._prepare_request(tool, params)
        start = time.perf_counter()
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers or None,
                params=query_params or None,
                json=json_payload if json_payload is not None else None,
                data=data_payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            latency = time.perf_counter() - start
            message = f"Request failure invoking {tool.name}: {exc}"
            logger.error(message)
            return message, False, str(exc), latency

        latency = time.perf_counter() - start
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            observation = self._format_error_response(response)
            logger.warning(
                "Tool %s responded with HTTP %s: %s",
                tool.name,
                response.status_code,
                response.reason,
            )
            return observation, False, str(exc), latency

        observation = self._summarise_response(response)
        return observation, True, None, latency

    def _prepare_request(
        self,
        tool: ToolSpec,
        params: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, str], Dict[str, Any], Optional[Any], Optional[Any]]:
        payload = {key: value for key, value in params.items() if value is not None}
        path_values, query_params, header_params, remaining = self._extract_parameter_values(tool, payload)
        url, remaining_after_path = self._resolve_url(tool, path_values, remaining)
        method = (tool.method or "get").lower()

        body_json: Optional[Any] = None
        body_data: Optional[Any] = None
        if method in {"get", "delete", "head", "options"}:
            for key, value in remaining_after_path.items():
                query_params.setdefault(key, value)
        else:
            body_candidate = None
            if "body" in remaining_after_path:
                body_candidate = remaining_after_path.pop("body")
            elif "json" in remaining_after_path:
                body_candidate = remaining_after_path.pop("json")
            elif remaining_after_path:
                body_candidate = remaining_after_path

            if isinstance(body_candidate, (str, bytes, bytearray)):
                body_data = body_candidate
            elif body_candidate is not None:
                body_json = body_candidate

        auth_context = self._resolve_auth_context(tool)

        headers: Dict[str, str] = {k: str(v) for k, v in header_params.items()}
        query_merged: Dict[str, Any] = dict(query_params)
        if auth_context.headers:
            headers.update(auth_context.headers)
        if auth_context.query_params:
            query_merged.update(auth_context.query_params)

        if body_json is not None:
            header_names = {key.lower(): key for key in headers.keys()}
            if "content-type" not in header_names:
                headers["Content-Type"] = "application/json"

        return method, url, headers, query_merged, body_json, body_data

    def _extract_parameter_values(
        self,
        tool: ToolSpec,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        working = dict(payload)
        path_values: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}
        header_params: Dict[str, Any] = {}

        for parameter in tool.parameters or []:
            if not isinstance(parameter, dict):
                continue
            name = parameter.get("name")
            if not name:
                continue
            location = (parameter.get("in") or "").lower()
            required = bool(parameter.get("required"))
            lookup_keys = [name]
            if "-" in name:
                lookup_keys.append(name.replace("-", "_"))

            value_found = False
            value = None
            for key in lookup_keys:
                if key in working:
                    value = working.pop(key)
                    value_found = True
                    break

            if location == "path":
                if not value_found:
                    if required:
                        raise RuntimeError(
                            f"Missing required path parameter '{name}' for tool '{tool.name}'."
                        )
                    continue
                path_values[name] = value
            elif value_found:
                if location == "query":
                    query_params[name] = value
                elif location == "header":
                    header_params[name] = value

        return path_values, query_params, header_params, working

    def _resolve_url(
        self,
        tool: ToolSpec,
        path_values: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        path_template = tool.path or "/"
        remaining = dict(payload)
        placeholders = re.findall(r"{([^}]+)}", path_template)

        for placeholder in placeholders:
            key = placeholder.split(":")[0]
            if key in path_values:
                value = path_values.pop(key)
            elif key in remaining:
                value = remaining.pop(key)
            elif key.replace("-", "_") in remaining:
                alt_key = key.replace("-", "_")
                value = remaining.pop(alt_key)
            else:
                raise RuntimeError(
                    f"Missing value for path placeholder '{key}' in tool '{tool.name}'."
                )
            path_template = path_template.replace(f"{{{placeholder}}}", str(value))

        base_url = self._select_base_url(tool)
        full_url = urljoin(base_url.rstrip("/") + "/", path_template.lstrip("/"))
        return full_url, remaining

    def _select_base_url(self, tool: ToolSpec) -> str:
        for candidate in tool.servers:
            if not isinstance(candidate, str):
                continue
            cleaned = candidate.strip()
            if not cleaned:
                continue
            if cleaned.startswith("http://") or cleaned.startswith("https://"):
                return cleaned
            if self._default_server:
                return urljoin(self._default_server.rstrip("/") + "/", cleaned.lstrip("/"))
        if self._default_server:
            return self._default_server
        raise RuntimeError(
            f"No server URL available for tool '{tool.name}'. Provide `default_tool_server` in the config."
        )

    def _resolve_auth_context(self, tool: ToolSpec) -> AuthContext:
        try:
            return self._auth_manager.get_context(tool)
        except RuntimeError:
            if self._require_auth:
                raise
            logger.warning("Auth disabled yet required for tool '%s'; continuing without credentials.", tool.name)
            return AuthContext()

    def _summarise_response(self, response: requests.Response) -> str:
        body_preview = self._extract_body_preview(response)
        return f"HTTP {response.status_code}: {body_preview}"

    def _format_error_response(self, response: requests.Response) -> str:
        body_preview = self._extract_body_preview(response)
        reason = response.reason or "error"
        return f"HTTP {response.status_code} {reason}: {body_preview}"

    @staticmethod
    def _extract_body_preview(response: requests.Response, limit: int = 2000) -> str:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                payload = response.json()
                text = json.dumps(payload, ensure_ascii=False)
            except ValueError:
                text = response.text or ""
        else:
            text = response.text or ""
        return ToolRegistry._truncate(text, limit=limit)

    @staticmethod
    def _truncate(value: str, limit: int = 2000) -> str:
        if value is None:
            return ""
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."


def build_agent_plan(
    config: AgentGenerationConfig,
    tool_registry: ToolRegistry,
    question: AgentQuestion,
    tool_candidates: Sequence[str],
    monitor: Optional[PerformanceMonitor],
) -> List[dict]:
    catalog_lines: List[str] = []
    for name in tool_candidates:
        tool = tool_registry.resolve(name)
        if tool:
            catalog_lines.append(tool.to_prompt_block())
    if not catalog_lines:
        catalog_lines = [spec.to_prompt_block() for spec in tool_registry.specs()]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert planner for an autonomous API agent. "
                "Given the question and available tools, produce a JSON array plan. "
                "Each item must include `tool`, `reason`, `inputs` (object), "
                "and optional `assistant_thought` summarising internal reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question.prompt}\n\nAvailable tools:\n"
                + "\n\n".join(catalog_lines)
            ),
        },
    ]

    raw = llm_generator(
        api_key=config.api_key,
        model=config.core_agent_answer_generate_model,
        base_url=config.base_url,
        message=messages,
        temperature=config.agent_temperature,
        top_p=0.9,
        type="agent_plan",
        debug=config.debug,
        max_retries=config.max_retries,
        min_interval_seconds=config.min_request_interval_seconds,
        perf_monitor=monitor,
        perf_stage="agent_answer_generation",
    )
    parsed = extract_json_from_llm_output(raw[0]) if raw else None
    tool_list = list(tool_candidates)
    fallback_tools = tool_list or tool_registry.list_tool_names()
    if not isinstance(parsed, list):
        if not fallback_tools:
            return []
        return [{"tool": fallback_tools[0], "reason": "Default fallback", "inputs": {}}]
    steps: List[dict] = []
    for entry in parsed[: config.max_turns]:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool")
        if not tool_name or not isinstance(tool_name, str):
            continue
        steps.append(
            {
                "tool": tool_name.strip(),
                "reason": entry.get("reason") or "",
                "inputs": entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {},
                "assistant_thought": entry.get("assistant_thought"),
            }
        )
    if not steps and fallback_tools:
        steps.append({"tool": fallback_tools[0], "reason": "Default fallback", "inputs": {}})
    return steps


def generate_agent_final_answer(
    config: AgentGenerationConfig,
    question: AgentQuestion,
    tool_calls: Sequence[ToolCall],
    monitor: Optional[PerformanceMonitor],
) -> str:
    call_summaries = []
    for call in tool_calls:
        call_summaries.append(
            f"Tool {call.tool_name} with input {json.dumps(call.input, ensure_ascii=False)} "
            f"produced observation: {call.observation}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are the final responding agent. Summarise the findings, "
                "provide the requested output, and offer any follow-up actions. "
                "Incorporate reflection on tool reliability when helpful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question.prompt}\n\n"
                f"Tool call summary:\n" + "\n".join(call_summaries)
            ),
        },
    ]
    raw = llm_generator(
        api_key=config.api_key,
        model=config.core_agent_answer_generate_model,
        base_url=config.base_url,
        message=messages,
        temperature=config.agent_temperature,
        top_p=0.9,
        type="agent_final_answer",
        debug=config.debug,
        max_retries=config.max_retries,
        min_interval_seconds=config.min_request_interval_seconds,
        perf_monitor=monitor,
        perf_stage="agent_answer_generation",
    )
    if raw:
        return raw[0]
    return "Unable to produce final answer due to generation failure."


def ensure_langgraph_available() -> None:
    if StateGraph is None:
        raise ImportError(
            "LangGraph is not installed. Install langgraph>=0.0.30 to run the agent pipeline."
        )


class LangGraphAgent:
    def __init__(self, config: AgentGenerationConfig, tool_registry: ToolRegistry):
        ensure_langgraph_available()
        self.config = config
        self.tool_registry = tool_registry
        self.agent_name = "langgraph_agent"
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(dict)

        def plan_node(state: dict) -> dict:
            state["plan"] = self._generate_plan(state)
            state["step_index"] = 0
            turns: List[AgentTurn] = state.get("turns") or []
            plan_explanation = "\n".join(
                f"{idx + 1}. {step.get('tool')} - {step.get('reason')}"
                for idx, step in enumerate(state["plan"])
            )
            turns.append(
                AgentTurn(
                    role="assistant",
                    content=f"Planning steps:\n{plan_explanation}",
                )
            )
            state["turns"] = turns
            return state

        def act_node(state: dict) -> dict:
            plan: List[dict] = state.get("plan") or []
            index = state.get("step_index", 0)
            if index >= len(plan):
                state["complete"] = True
                return state
            step = plan[index]
            tool_name = step.get("tool")
            if not tool_name:
                state.setdefault("issues", []).append("Plan step missing tool name")
                state["step_index"] = index + 1
                return state
            tool_spec = self.tool_registry.resolve(tool_name)
            if not tool_spec:
                state.setdefault("issues", []).append(f"Tool {tool_name} not found")
                state["step_index"] = index + 1
                return state
            params = step.get("inputs") or {}
            observation, success, error, latency = self.tool_registry.invoke(
                tool_spec, params, monitor
            )
            tool_calls: List[ToolCall] = state.get("tool_calls") or []
            sequence = len(tool_calls) + 1
            tool_calls.append(
                ToolCall(
                    sequence=sequence,
                    tool_name=tool_spec.name,
                    input=params if isinstance(params, dict) else {},
                    observation=observation,
                    success=success,
                    latency_seconds=latency,
                    error=error,
                    metadata={"reason": step.get("reason")},
                )
            )
            if not success and error:
                state.setdefault("issues", []).append(
                    f"Tool {tool_spec.name} invocation failed: {error}"
                )
            turns: List[AgentTurn] = state.get("turns") or []
            turns.append(
                AgentTurn(
                    role="assistant",
                    content=step.get("assistant_thought") or f"Calling tool {tool_spec.name}",
                    tool_name=tool_spec.name,
                    tool_input=params if isinstance(params, dict) else {},
                )
            )
            turns.append(
                AgentTurn(
                    role="tool",
                    content=observation,
                    tool_name=tool_spec.name,
                    observation=observation,
                )
            )
            state["turns"] = turns
            state["tool_calls"] = tool_calls
            state["step_index"] = index + 1
            return state

        def should_continue(state: dict) -> str:
            plan: List[dict] = state.get("plan") or []
            index = state.get("step_index", 0)
            if index >= len(plan) or index >= self.config.max_turns:
                return "finalise"
            return "act"

        def finalise_node(state: dict) -> dict:
            final_answer = self._generate_final_answer(state)
            state["final_answer"] = final_answer
            turns: List[AgentTurn] = state.get("turns") or []
            turns.append(AgentTurn(role="assistant", content=final_answer))
            state["turns"] = turns
            return state

        graph.add_node("plan", plan_node)
        graph.add_node("act", act_node)
        graph.add_node("finalise", finalise_node)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "act")
        graph.add_conditional_edges(
            "act",
            should_continue,
            {"act": "act", "finalise": "finalise"},
        )
        graph.add_edge("finalise", END)
        return graph.compile()

    def run(
        self,
        question: AgentQuestion,
        tool_candidates: Sequence[ToolCandidate],
        monitor: PerformanceMonitor,
    ) -> Tuple[List[AgentTurn], List[ToolCall], str]:
        tool_names = [candidate.tool_name for candidate in tool_candidates if candidate.tool_name]
        initial_state = {
            "question": question,
            "tool_candidates": tool_names,
            "turns": [AgentTurn(role="user", content=question.prompt)],
            "tool_calls": [],
            "monitor": monitor,
        }
        with monitor.stage("agent_answer_generation"):
            final_state = self.graph.invoke(initial_state)
        tool_calls = final_state.get("tool_calls") or []
        turns = final_state.get("turns") or []
        final_answer = final_state.get("final_answer") or ""
        monitor.add_stage_items("agent_answer_generation", 1)
        return turns, tool_calls, final_answer

    def _generate_plan(self, state: dict) -> List[dict]:
        question: AgentQuestion = state["question"]
        tool_candidates: Sequence[str] = state.get("tool_candidates") or []
        monitor: Optional[PerformanceMonitor] = state.get("monitor")
        return build_agent_plan(
            config=self.config,
            tool_registry=self.tool_registry,
            question=question,
            tool_candidates=tool_candidates,
            monitor=monitor,
        )

    def _generate_final_answer(self, state: dict) -> str:
        question: AgentQuestion = state["question"]
        tool_calls: List[ToolCall] = state.get("tool_calls") or []
        monitor: Optional[PerformanceMonitor] = state.get("monitor")
        return generate_agent_final_answer(
            config=self.config,
            question=question,
            tool_calls=tool_calls,
            monitor=monitor,
        )


# ---------------------------------------------------------------------------
# Native OpenAI-style agent
# ---------------------------------------------------------------------------


class OpenAIAgent:
    def __init__(self, config: AgentGenerationConfig, tool_registry: ToolRegistry):
        self.config = config
        self.tool_registry = tool_registry
        self.agent_name = "openai_native_agent"

    def run(
        self,
        question: AgentQuestion,
        tool_candidates: Sequence[ToolCandidate],
        monitor: PerformanceMonitor,
    ) -> Tuple[List[AgentTurn], List[ToolCall], str]:
        tool_names = [candidate.tool_name for candidate in tool_candidates if candidate.tool_name]
        turns: List[AgentTurn] = [AgentTurn(role="user", content=question.prompt)]
        tool_calls: List[ToolCall] = []

        with monitor.stage("agent_answer_generation"):
            plan = build_agent_plan(
                config=self.config,
                tool_registry=self.tool_registry,
                question=question,
                tool_candidates=tool_names,
                monitor=monitor,
            )
            if plan:
                plan_explanation = "\n".join(
                    f"{idx + 1}. {step.get('tool')} - {step.get('reason')}"
                    for idx, step in enumerate(plan)
                )
                turns.append(
                    AgentTurn(
                        role="assistant",
                        content=f"Planning steps:\n{plan_explanation}",
                    )
                )
            for step_index, step in enumerate(plan, 1):
                tool_name = step.get("tool")
                if not tool_name:
                    continue
                tool_spec = self.tool_registry.resolve(tool_name)
                if not tool_spec:
                    turns.append(
                        AgentTurn(
                            role="assistant",
                            content=f"Step {step_index}: tool '{tool_name}' unavailable, skipping.",
                        )
                    )
                    continue
                params = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
                assistant_thought = step.get("assistant_thought") or f"Calling tool {tool_spec.name}"
                turns.append(
                    AgentTurn(
                        role="assistant",
                        content=assistant_thought,
                        tool_name=tool_spec.name,
                        tool_input=params,
                    )
                )
                observation, success, error, latency = self.tool_registry.invoke(
                    tool_spec, params, monitor
                )
                tool_call = ToolCall(
                    sequence=len(tool_calls) + 1,
                    tool_name=tool_spec.name,
                    input=params,
                    observation=observation,
                    success=success,
                    error=error,
                    latency_seconds=latency,
                    metadata={"reason": step.get("reason")},
                )
                tool_calls.append(tool_call)
                if not success and error:
                    turns.append(
                        AgentTurn(
                            role="assistant",
                            content=f"Tool {tool_spec.name} failed: {error}",
                        )
                    )
                turns.append(
                    AgentTurn(
                        role="tool",
                        content=observation,
                        tool_name=tool_spec.name,
                        observation=observation,
                    )
                )
                if len(tool_calls) >= self.config.max_turns:
                    break

            final_answer = generate_agent_final_answer(
                config=self.config,
                question=question,
                tool_calls=tool_calls,
                monitor=monitor,
            )
            turns.append(AgentTurn(role="assistant", content=final_answer))

        monitor.add_stage_items("agent_answer_generation", 1)
        return turns, tool_calls, final_answer


