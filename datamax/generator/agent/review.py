from __future__ import annotations

import json
from typing import List

from datamax.utils.performance_monitor import PerformanceMonitor

from ..qa_generator import extract_json_from_llm_output, llm_generator
from .config import AgentGenerationConfig
from .models import AgentEpisode, ReviewResult


class AgentReviewPipeline:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config

    def review(
        self, episode: AgentEpisode, monitor: PerformanceMonitor
    ) -> ReviewResult:
        issues: List[str] = []
        suggestions: List[str] = []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are auditing agent-generated training data. "
                    "Check for correctness, completeness, and tool alignment. "
                    "Return JSON with fields: success (bool), score (0-1), issues (list), suggestions (list)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": episode.question.to_dict(),
                        "tool_calls": [call.to_dict() for call in episode.tool_calls],
                        "final_answer": episode.final_answer,
                        "turns": [turn.to_dict() for turn in episode.turns],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        raw = llm_generator(
            api_key=self.config.api_key,
            model=self.config.review_model,
            base_url=self.config.base_url,
            message=messages,
            temperature=self.config.review_temperature,
            top_p=0.1,
            type="agent_review",
            debug=self.config.debug,
            max_retries=self.config.max_retries,
            min_interval_seconds=self.config.min_request_interval_seconds,
            perf_monitor=monitor,
            perf_stage="agent_review",
        )
        parsed = extract_json_from_llm_output(raw[0]) if raw else None
        success = False
        score = None
        if isinstance(parsed, dict):
            success = bool(parsed.get("success"))
            score = parsed.get("score")
            issues = (
                parsed.get("issues") if isinstance(parsed.get("issues"), list) else []
            )
            suggestions = (
                parsed.get("suggestions")
                if isinstance(parsed.get("suggestions"), list)
                else []
            )
        monitor.add_stage_items("agent_review", 1)
        return ReviewResult(
            success=success,
            score=float(score) if isinstance(score, (int, float)) else None,
            issues=[str(item) for item in issues],
            suggestions=[str(item) for item in suggestions],
            raw=parsed if isinstance(parsed, dict) else None,
        )


__all__ = ["AgentReviewPipeline"]
