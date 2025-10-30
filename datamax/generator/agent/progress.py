from __future__ import annotations

from typing import Optional

from ..qa_generator import QAProgressTracker


class AgentProgressTracker(QAProgressTracker):
    def _make_key(self, entry: dict) -> Optional[str]:
        if "episode_id" in entry:
            return entry["episode_id"]
        if "question" in entry and isinstance(entry["question"], dict):
            question_id = entry["question"].get("question_id")
            if question_id:
                return str(question_id)
        return super()._make_key(entry)


__all__ = ["AgentProgressTracker"]
