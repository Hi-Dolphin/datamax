from __future__ import annotations

import uuid
from typing import Any, List, Optional

from loguru import logger

from datamax.utils.performance_monitor import PerformanceMonitor

from ..qa_generator import extract_json_from_llm_output, llm_generator
from .config import AgentGenerationConfig
from .models import AgentQuestion, PromptContext
from .spec import ApiGraph


class AgentQuestionGenerator:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config

    def generate(
        self, api_graph: ApiGraph, monitor: PerformanceMonitor
    ) -> List[AgentQuestion]:
        questions: List[AgentQuestion] = []
        contexts = api_graph.build_prompt_contexts()
        if not contexts:
            logger.warning("No prompt contexts derived from API specifications")
            return questions

        with monitor.stage("agent_question_generation"):
            for context in contexts:
                if len(questions) >= self.config.question_count:
                    break
                batch_limit = min(
                    self.config.max_questions_per_context,
                    self.config.question_count - len(questions),
                )
                prompt = self._build_prompt(context, batch_limit, api_graph)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an API integration expert responsible for designing"
                            " realistic, high-coverage agent tasks for tool-augmented assistants."
                            " Create diverse user questions grounded in the provided interfaces."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
                logger.debug(f"Generating questions for context {context.name}")
                raw = llm_generator(
                    api_key=self.config.api_key,
                    model=self.config.agent_question_generate_model,
                    base_url=self.config.base_url,
                    message=messages,
                    temperature=self.config.question_temperature,
                    top_p=0.9,
                    type="agent_question",
                    debug=self.config.debug,
                    max_retries=self.config.max_retries,
                    min_interval_seconds=self.config.min_request_interval_seconds,
                    perf_monitor=monitor,
                    perf_stage="agent_question_generation",
                )
                if not raw:
                    continue
                parsed = extract_json_from_llm_output(raw[0])
                if not isinstance(parsed, list):
                    logger.warning(
                        "Question generation output not a list; skipping context"
                    )
                    continue
                for entry in parsed:
                    if len(questions) >= self.config.question_count:
                        break
                    question = self._normalise_question(entry, context, api_graph)
                    if question:
                        questions.append(question)
            monitor.add_stage_items("agent_question_generation", len(questions))
        return questions

    def _build_prompt(
        self, context: PromptContext, batch_limit: int, api_graph: ApiGraph
    ) -> str:
        endpoint_summary = api_graph.describe_endpoints(context.endpoints, limit=12)
        instructions = (
            "Generate realistic end-user questions/goals that require interacting with the provided APIs. "
            "Ensure coverage of single-tool and multi-tool chains (including failure handling when relevant). "
            "Return a JSON array where each item has keys: "
            "`question` (string), `required_tools` (list of tool names), `scenario_type` "
            "(one of ['single_call','multi_step','error_recovery','batch','analytics']), "
            "`difficulty` (easy|medium|hard), `prerequisites` (list of preconditions), "
            "`metadata` (object with helpful notes). "
            "Required tools must reference the tool names listed below. "
            "Prioritise questions that highlight dependencies or typical workflows described."
        )
        return (
            f"Context: {context.name}\n"
            f"Endpoints:\n{endpoint_summary}\n\n"
            f"Please generate up to {batch_limit} questions.\n"
            f"{instructions}\n"
            "Respond with JSON only."
        )

    def _normalise_question(
        self,
        entry: Any,
        context: PromptContext,
        api_graph: ApiGraph,
    ) -> Optional[AgentQuestion]:
        if not isinstance(entry, dict):
            return None
        question_text = entry.get("question") or entry.get("prompt")
        if not question_text or not isinstance(question_text, str):
            return None
        required_tools = entry.get("required_tools")
        if not isinstance(required_tools, list):
            required_tools = []
        normalised_tools: List[str] = []
        for tool in required_tools:
            if not isinstance(tool, str):
                continue
            tool = tool.strip()
            if tool in api_graph._endpoint_by_id or any(
                tool == ep.tool_name() for ep in context.endpoints
            ):
                normalised_tools.append(tool)
        if not normalised_tools and context.endpoints:
            normalised_tools = [context.endpoints[0].tool_name()]
        scenario_type = entry.get("scenario_type") or "multi_step"
        difficulty = entry.get("difficulty") or "medium"
        prerequisites = entry.get("prerequisites") or []
        metadata = entry.get("metadata") or {}
        question_id = str(uuid.uuid4())
        return AgentQuestion(
            question_id=question_id,
            prompt=question_text.strip(),
            target_tools=normalised_tools,
            scenario_type=scenario_type,
            difficulty=difficulty,
            prerequisites=prerequisites if isinstance(prerequisites, list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
        )


__all__ = ["AgentQuestionGenerator"]
