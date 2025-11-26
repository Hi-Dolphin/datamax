from __future__ import annotations

from typing import List, Sequence

from datamax.utils.performance_monitor import PerformanceMonitor

from ..qa_generator import extract_json_from_llm_output, llm_generator
from .config import AgentGenerationConfig
from .models import AgentQuestion, ToolCandidate, ToolSpec


class ToolClassifier:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config

    def classify(
        self,
        question: AgentQuestion,
        tool_catalog: Sequence[ToolSpec],
        monitor: PerformanceMonitor,
    ) -> List[ToolCandidate]:
        tool_blocks = [tool.to_prompt_block() for tool in tool_catalog]
        prompt = (
            "You are selecting the best tools for an autonomous agent. "
            "Given the question and tool catalog, return a JSON array with each item containing "
            "`tool_name`, `score` (0-1 float), and `rationale`. "
            f"Select at most {self.config.top_k_tools} tools sorted by descending score."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question.prompt}\n\n"
                    f"Candidate tools:\n" + "\n\n".join(tool_blocks)
                ),
            },
        ]
        raw = llm_generator(
            api_key=self.config.api_key,
            model=self.config.classify_model,
            base_url=self.config.base_url,
            message=messages,
            temperature=self.config.classify_temperature,
            top_p=0.9,
            type="tool_classification",
            debug=self.config.debug,
            max_retries=self.config.max_retries,
            min_interval_seconds=self.config.min_request_interval_seconds,
            perf_monitor=monitor,
            perf_stage="tool_classification",
        )
        if not raw:
            return []
        parsed = extract_json_from_llm_output(raw[0])
        candidates: List[ToolCandidate] = []
        if isinstance(parsed, list):
            for item in parsed[: self.config.top_k_tools]:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool_name")
                if not name or not isinstance(name, str):
                    continue
                score = item.get("score")
                try:
                    score_value = float(score) if score is not None else None
                except (TypeError, ValueError):
                    score_value = None
                rationale = item.get("rationale") or ""
                candidates.append(
                    ToolCandidate(
                        tool_name=name.strip(),
                        score=score_value,
                        rationale=str(rationale),
                    ),
                )
        monitor.add_stage_items("tool_classification", len(candidates))
        return candidates


__all__ = ["ToolClassifier"]
