"""
Performance monitoring utilities for DataMax LLM workflows.

The PerformanceMonitor is designed to collect stage-level timings,
token usage, request rates, and detailed LLM invocations across the QA
generation pipeline. It can be shared across modules to provide a
consistent view of how each stage performs, making it easier to plug
metrics and call transcripts into automation or analytics tooling.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence


@dataclass
class StageStats:
    """Mutable stats container used internally by PerformanceMonitor."""

    name: str
    runs: int = 0
    total_duration: float = 0.0
    total_items: int = 0
    request_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_request_duration: float = 0.0


@dataclass
class StageReport:
    """Immutable snapshot returned to callers."""

    stage: str
    runs: int
    items: int
    stage_duration_seconds: float
    items_per_second: float
    items_per_minute: float
    request_count: int
    average_request_duration_seconds: float
    effective_qpm: float
    effective_concurrency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    tokens_per_request: float

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "runs": self.runs,
            "items": self.items,
            "stage_duration_seconds": self.stage_duration_seconds,
            "items_per_second": self.items_per_second,
            "items_per_minute": self.items_per_minute,
            "request_count": self.request_count,
            "average_request_duration_seconds": self.average_request_duration_seconds,
            "effective_qpm": self.effective_qpm,
            "effective_concurrency": self.effective_concurrency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_request": self.tokens_per_request,
        }


@dataclass
class LLMCallRecord:
    """Structured record of a single LLM invocation."""

    id: str
    stage: str
    call_type: str
    prompt: Optional[str]
    messages: List[Dict[str, str]]
    response: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stage": self.stage,
            "call_type": self.call_type,
            "prompt": self.prompt,
            "messages": self.messages,
            "response": self.response,
            "metadata": self.metadata,
        }


class PerformanceMonitor:
    """
    Collects stage-level performance metrics for LLM driven workflows.

    Usage example::

        monitor = PerformanceMonitor()
        with monitor.stage("question_generation"):
            ...
        monitor.add_stage_items("question_generation", len(questions))

        monitor.record_request(
            stage="question_generation",
            duration_seconds=0.8,
            prompt_tokens=1200,
            completion_tokens=350,
        )

        report = monitor.build_report()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stages: Dict[str, StageStats] = {}
        self._workflow_started_at = time.perf_counter()
        self._call_records: List[LLMCallRecord] = []

    def _stage(self, name: str) -> StageStats:
        with self._lock:
            if name not in self._stages:
                self._stages[name] = StageStats(name=name)
            return self._stages[name]

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """
        Context manager for timing a logical stage.

        Example::
            with monitor.stage("question_generation"):
                run_question_stage()
        """
        stage = self._stage(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                stage.total_duration += duration
                stage.runs += 1

    def add_stage_items(self, name: str, count: int) -> None:
        """Increment the processed item count for a stage."""
        if count <= 0:
            return
        stage = self._stage(name)
        with self._lock:
            stage.total_items += count

    def record_request(
        self,
        *,
        stage: Optional[str],
        duration_seconds: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """
        Record a single LLM request.

        Args:
            stage: Logical stage name; defaults to 'llm_calls' if None.
            duration_seconds: Wall clock duration for the request.
            prompt_tokens: Prompt token usage if available.
            completion_tokens: Completion token usage if available.
        """
        name = stage or "llm_calls"
        stage_stats = self._stage(name)
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = prompt_tokens + completion_tokens

        with self._lock:
            stage_stats.request_count += 1
            stage_stats.total_request_duration += max(duration_seconds, 0.0)
            stage_stats.total_prompt_tokens += prompt_tokens
            stage_stats.total_completion_tokens += completion_tokens
            stage_stats.total_tokens += total_tokens

    def record_call(
        self,
        *,
        stage: Optional[str],
        call_type: str,
        messages: Sequence[Dict[str, Any]] | None,
        response: Optional[str],
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist the transcript of an LLM request/response cycle.

        Args:
            stage: Logical stage name; defaults to 'llm_calls' if None.
            call_type: High level call category (e.g. 'question', 'answer').
            messages: Sequence of chat messages that were sent to the LLM.
            response: Raw textual response returned by the LLM.
            prompt: Optional prompt text used to build the request.
            metadata: Optional auxiliary metadata such as model name.
        """
        normalized_messages: List[Dict[str, str]] = []
        if messages:
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "") or "").strip()
                content = message.get("content", "")
                if content is None:
                    content = ""
                normalized_messages.append(
                    {"role": role, "content": str(content)}
                )

        record = LLMCallRecord(
            id=str(uuid.uuid4()),
            stage=stage or "llm_calls",
            call_type=call_type or "unknown",
            prompt=prompt,
            messages=normalized_messages,
            response=response or "",
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._call_records.append(record)

    def get_call_records(self) -> List[dict]:
        """Return a shallow copy of recorded LLM call transcripts."""
        with self._lock:
            return [record.to_dict() for record in self._call_records]

    def call_records_as_qa_pairs(self) -> List[dict]:
        """Convert recorded LLM calls into QA-style training entries."""
        records = self.get_call_records()
        qa_pairs: List[dict] = []
        for record in records:
            instruction = self._messages_to_instruction(record.get("messages", []))
            if not instruction and record.get("prompt"):
                instruction = str(record.get("prompt") or "")
            qa_pairs.append(
                {
                    "qid": record["id"],
                    "instruction": instruction,
                    "input": "",
                    "output": record.get("response", ""),
                    "label": "",
                    "tag-path": "",
                    "context": "",
                    "source_stage": record.get("stage"),
                    "call_type": record.get("call_type"),
                    "metadata": record.get("metadata", {}),
                }
            )
        return qa_pairs

    def build_report(self) -> dict:
        """Return a dictionary summarising all recorded metrics."""
        with self._lock:
            stage_snapshots = {
                name: self._create_stage_report(stats).to_dict()
                for name, stats in self._stages.items()
            }

            call_count = len(self._call_records)

            total_prompt_tokens = sum(
                stats.total_prompt_tokens for stats in self._stages.values()
            )
            total_completion_tokens = sum(
                stats.total_completion_tokens for stats in self._stages.values()
            )
            total_tokens = sum(stats.total_tokens for stats in self._stages.values())
            total_requests = sum(stats.request_count for stats in self._stages.values())

            overall_duration = max(
                time.perf_counter() - self._workflow_started_at, 0.0
            )

            overall_tokens_per_second = (
                total_tokens / overall_duration if overall_duration > 0 else 0.0
            )
            overall_qpm = (
                (total_requests / overall_duration) * 60
                if overall_duration > 0
                else 0.0
            )

            return {
                "stages": stage_snapshots,
                "totals": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                    "request_count": total_requests,
                    "workflow_duration_seconds": overall_duration,
                    "tokens_per_second": overall_tokens_per_second,
                    "overall_qpm": overall_qpm,
                    "llm_call_count": call_count,
                },
            }

    def _create_stage_report(self, stats: StageStats) -> StageReport:
        duration = max(stats.total_duration, 0.0)
        items = stats.total_items
        request_count = stats.request_count

        items_per_second = (items / duration) if duration > 0 else 0.0
        effective_qpm = (
            (request_count / duration) * 60 if duration > 0 else 0.0
        )
        tokens_per_second = (
            (stats.total_tokens / duration) if duration > 0 else 0.0
        )
        tokens_per_request = (
            (stats.total_tokens / request_count) if request_count > 0 else 0.0
        )
        average_request_duration = (
            stats.total_request_duration / request_count
            if request_count > 0
            else 0.0
        )
        effective_concurrency = (
            (stats.total_request_duration / duration) if duration > 0 else 0.0
        )

        return StageReport(
            stage=stats.name,
            runs=stats.runs,
            items=items,
            stage_duration_seconds=duration,
            items_per_second=items_per_second,
            items_per_minute=items_per_second * 60,
            request_count=request_count,
            average_request_duration_seconds=average_request_duration,
            effective_qpm=effective_qpm,
            effective_concurrency=effective_concurrency,
            prompt_tokens=stats.total_prompt_tokens,
            completion_tokens=stats.total_completion_tokens,
            total_tokens=stats.total_tokens,
            tokens_per_second=tokens_per_second,
            tokens_per_request=tokens_per_request,
        )

    @staticmethod
    def _messages_to_instruction(messages: Sequence[Dict[str, str]]) -> str:
        if not messages:
            return ""
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "").upper() or "UNKNOWN"
            content = message.get("content", "")
            if content is None:
                content = ""
            lines.append(f"{role}: {content}".strip())
        return "\n\n".join(line for line in lines if line) or ""
