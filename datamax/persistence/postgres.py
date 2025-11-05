from __future__ import annotations

import datetime as dt
import hashlib
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from datamax.generator.types import PersistenceConfig, QAGenerationResult
from datamax.utils.performance_monitor import PerformanceMonitor


class PostgresPersistence(AbstractContextManager):
    """Context-managed Postgres persistence handler."""

    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.conn: Optional[psycopg.Connection] = None
        self.run_id: Optional[int] = None
        self._accumulator = _RunAccumulator(
            source_key=config.source_key,
            model_key=config.model_key or "unknown-model",
        )
        self._source_id: Optional[int] = None
        self._model_id: Optional[int] = None

    def __enter__(self) -> "PostgresPersistence":
        if not self.conn:
            self.conn = psycopg.connect(self.config.dsn, row_factory=dict_row)
            self.conn.autocommit = False
        self._prepare_run()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn:
            if exc is not None:
                self.conn.rollback()
                if self.run_id:
                    self._finish_run(status="failed")
            else:
                self._finish_run(status="success")
                self.conn.commit()
            self.conn.close()
        self.conn = None
        self.run_id = None

    # Convenience API when context manager syntax is not desired
    def start(self) -> None:
        self.__enter__()

    def finish(self, exc_type=None, exc=None, tb=None) -> None:
        self.__exit__(exc_type, exc, tb)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def persist(self, result: QAGenerationResult, monitor: Optional[PerformanceMonitor]) -> None:
        if not self.conn:
            raise RuntimeError("PostgresPersistence must be entered before use.")

        if result.source_file:
            suffix = Path(result.source_file).suffix.lower()
            if suffix:
                self.config.tags["payload_type"] = suffix.lstrip(".") or "text"

        ctx = _PersistenceContext(
            config=self.config,
            performance=result.performance or {},
            monitor=monitor,
            source_id=self._source_id or 0,
            model_id=self._model_id or 0,
            run_id=self.run_id or 0,
        )

        payload_id = None
        if self.config.toggles.save_raw_payload:
            payload_id = self._insert_raw_payload(result, ctx)

        if self.config.toggles.save_pairs:
            self._insert_pairs(result, ctx, payload_id)

        if self.config.toggles.save_pipeline_events and result.call_records:
            self._insert_pipeline_events(result.call_records)

        self._accumulator.consume(result)

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _prepare_run(self) -> None:
        with self.conn.cursor() as cur:
            source_id = self._ensure_source_dim(cur)
            model_id = self._ensure_model_dim(cur)
            self.conn.commit()

        self._source_id = source_id
        self._model_id = model_id
        self.config.tags.setdefault("__source_id__", source_id)
        self.config.tags.setdefault("__model_id__", model_id)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sdc_ai.qa_generation_run
                    (run_name, trigger_type, model_id, model_name, status)
                VALUES (%s, %s, %s, %s, 'running')
                RETURNING run_id
                """,
                (
                    self.config.run_name,
                    self.config.trigger_type,
                    model_id,
                    self.config.model_key or "unknown-model",
                ),
            )
            self.run_id = cur.fetchone()["run_id"]
        self.config.tags["__run_id__"] = self.run_id
        self.conn.commit()

    def _finish_run(self, status: str) -> None:
        if not self.conn or not self.run_id:
            return

        hourly_row = self._accumulator.hourly_row()
        daily_row = self._accumulator.daily_row()

        with self.conn.cursor() as cur:
            if self.config.toggles.save_hourly_metrics:
                cur.execute(
                    """
                    INSERT INTO sdc_ai.qa_metrics_hourly (
                        bucket_start, bucket_end,
                        total_generated, success_count, failure_count,
                        total_prompt_tokens, total_completion_tokens, total_tokens,
                        tokens_per_second, overall_qpm, total_duration_seconds, tokens_per_request,
                        avg_latency_ms, p95_latency_ms,
                        avg_prompt_tokens, avg_completion_tokens,
                        avg_confidence, source_breakdown, model_breakdown
                    )
                    VALUES (
                        %(bucket_start)s, %(bucket_end)s,
                        %(total_generated)s, %(success_count)s, %(failure_count)s,
                        %(total_prompt_tokens)s, %(total_completion_tokens)s, %(total_tokens)s,
                        %(tokens_per_second)s, %(overall_qpm)s, %(total_duration_seconds)s, %(tokens_per_request)s,
                        %(avg_latency_ms)s, %(p95_latency_ms)s,
                        %(avg_prompt_tokens)s, %(avg_completion_tokens)s,
                        %(avg_confidence)s, %(source_breakdown)s, %(model_breakdown)s
                    )
                    ON CONFLICT (bucket_start) DO UPDATE
                    SET bucket_end = EXCLUDED.bucket_end,
                        total_generated = EXCLUDED.total_generated,
                        success_count = EXCLUDED.success_count,
                        failure_count = EXCLUDED.failure_count,
                        total_prompt_tokens = EXCLUDED.total_prompt_tokens,
                        total_completion_tokens = EXCLUDED.total_completion_tokens,
                        total_tokens = EXCLUDED.total_tokens,
                        tokens_per_second = EXCLUDED.tokens_per_second,
                        overall_qpm = EXCLUDED.overall_qpm,
                        total_duration_seconds = EXCLUDED.total_duration_seconds,
                        tokens_per_request = EXCLUDED.tokens_per_request,
                        avg_latency_ms = EXCLUDED.avg_latency_ms,
                        p95_latency_ms = EXCLUDED.p95_latency_ms,
                        avg_prompt_tokens = EXCLUDED.avg_prompt_tokens,
                        avg_completion_tokens = EXCLUDED.avg_completion_tokens,
                        avg_confidence = EXCLUDED.avg_confidence,
                        source_breakdown = EXCLUDED.source_breakdown,
                        model_breakdown = EXCLUDED.model_breakdown
                    """,
                    {
                        **hourly_row,
                        "source_breakdown": Json(hourly_row["source_breakdown"]),
                        "model_breakdown": Json(hourly_row["model_breakdown"]),
                    },
                )

            if self.config.toggles.save_daily_metrics:
                cur.execute(
                    """
                    INSERT INTO sdc_ai.qa_metrics_daily (
                        date_key, total_generated, success_count, failure_count,
                        total_prompt_tokens, total_completion_tokens, total_tokens,
                        tokens_per_second, overall_qpm, total_duration_seconds, tokens_per_request,
                        avg_latency_ms, p95_latency_ms,
                        avg_eval_score, avg_confidence,
                        source_breakdown, model_breakdown
                    )
                    VALUES (
                        %(date_key)s, %(total_generated)s, %(success_count)s, %(failure_count)s,
                        %(total_prompt_tokens)s, %(total_completion_tokens)s, %(total_tokens)s,
                        %(tokens_per_second)s, %(overall_qpm)s, %(total_duration_seconds)s, %(tokens_per_request)s,
                        %(avg_latency_ms)s, %(p95_latency_ms)s,
                        %(avg_eval_score)s, %(avg_confidence)s,
                        %(source_breakdown)s, %(model_breakdown)s
                    )
                    ON CONFLICT (date_key) DO UPDATE
                    SET total_generated = EXCLUDED.total_generated,
                        success_count = EXCLUDED.success_count,
                        failure_count = EXCLUDED.failure_count,
                        total_prompt_tokens = EXCLUDED.total_prompt_tokens,
                        total_completion_tokens = EXCLUDED.total_completion_tokens,
                        total_tokens = EXCLUDED.total_tokens,
                        tokens_per_second = EXCLUDED.tokens_per_second,
                        overall_qpm = EXCLUDED.overall_qpm,
                        total_duration_seconds = EXCLUDED.total_duration_seconds,
                        tokens_per_request = EXCLUDED.tokens_per_request,
                        avg_latency_ms = EXCLUDED.avg_latency_ms,
                        p95_latency_ms = EXCLUDED.p95_latency_ms,
                        avg_eval_score = EXCLUDED.avg_eval_score,
                        avg_confidence = EXCLUDED.avg_confidence,
                        source_breakdown = EXCLUDED.source_breakdown,
                        model_breakdown = EXCLUDED.model_breakdown
                    """,
                    {
                        **daily_row,
                        "source_breakdown": Json(daily_row["source_breakdown"]),
                        "model_breakdown": Json(daily_row["model_breakdown"]),
                    },
                )

            cur.execute(
                """
                UPDATE sdc_ai.qa_generation_run
                SET finished_at = NOW(),
                    status = %s,
                    total_items = %s,
                    success_count = %s,
                    failed_count = %s,
                    prompt_tokens = %s,
                    completion_tokens = %s,
                    total_tokens = %s,
                    tokens_per_second = %s,
                    overall_qpm = %s,
                    workflow_duration_seconds = %s,
                    tokens_per_request = %s,
                    metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                WHERE run_id = %s
                """,
                (
                    status,
                    self._accumulator.total_questions,
                    self._accumulator.success_pairs,
                    self._accumulator.failed_pairs,
                    self._accumulator.workflow_prompt_tokens,
                    self._accumulator.workflow_completion_tokens,
                    self._accumulator.workflow_total_tokens,
                    self._accumulator.workflow_tokens_per_second,
                    self._accumulator.workflow_overall_qpm,
                    self._accumulator.workflow_duration_seconds,
                    self._accumulator.workflow_tokens_per_request,
                    Json(
                        {
                            "source_key": self.config.source_key,
                            "model_key": self.config.model_key,
                            "files_processed": self._accumulator.files_processed,
                            "files_failed": self._accumulator.files_failed,
                        }
                    ),
                    self.run_id,
                ),
            )
        self.conn.commit()

    def _ensure_source_dim(self, cur) -> int:
        cur.execute(
            """
            INSERT INTO sdc_ai.qa_source_dim (source_key, display_name, owner_team, active)
            VALUES (%s, COALESCE(%s, %s), %s, TRUE)
            ON CONFLICT (source_key) DO UPDATE
            SET display_name = EXCLUDED.display_name,
                owner_team = COALESCE(EXCLUDED.owner_team, sdc_ai.qa_source_dim.owner_team),
                active = TRUE
            RETURNING source_id
            """,
            (
                self.config.source_key,
                self.config.source_name,
                self.config.source_key,
                self.config.owner_team,
            ),
        )
        return cur.fetchone()["source_id"]

    def _ensure_model_dim(self, cur) -> int:
        cur.execute(
            """
            INSERT INTO sdc_ai.qa_model_dim (model_key, provider, version, active)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT (model_key) DO UPDATE
            SET provider = EXCLUDED.provider,
                version = COALESCE(EXCLUDED.version, sdc_ai.qa_model_dim.version),
                active = TRUE
            RETURNING model_id
            """,
            (
                self.config.model_key or "unknown-model",
                self.config.model_provider or "unknown",
                self.config.model_version,
            ),
        )
        return cur.fetchone()["model_id"]

    def _insert_raw_payload(self, result: QAGenerationResult, ctx: "_PersistenceContext") -> Optional[int]:
        if not self.conn:
            return None
        checksum = hashlib.sha256(result.raw_content.encode("utf-8")).hexdigest()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sdc_ai.qa_raw_payload (
                    source_id, source_system, external_id,
                    payload_type, payload_content, payload_json,
                    checksum, created_by
                )
                VALUES (
                    %(source_id)s, %(source_system)s, %(external_id)s,
                    %(payload_type)s, %(payload_content)s, %(payload_json)s,
                    %(checksum)s, %(created_by)s
                )
                ON CONFLICT (source_system, checksum) DO UPDATE
                SET payload_content = EXCLUDED.payload_content,
                    payload_json = EXCLUDED.payload_json
                RETURNING payload_id
                """,
                {
                    "source_id": ctx.source_id,
                    "source_system": self.config.source_key,
                    "external_id": result.source_file or "unknown",
                    "payload_type": ctx.payload_type,
                    "payload_content": result.raw_content,
                    "payload_json": Json(
                        {
                            "metadata": result.metadata,
                            "domain_tree": result.domain_tree,
                            "tags": self.config.tags,
                        }
                    ),
                    "checksum": checksum,
                    "created_by": self.config.created_by,
                },
            )
            payload_id = cur.fetchone()["payload_id"]
        self.conn.commit()
        return payload_id

    def _insert_pairs(
        self,
        result: QAGenerationResult,
        ctx: "_PersistenceContext",
        payload_id: Optional[int],
    ) -> None:
        if not self.conn:
            return

        template = ctx.qa_template()
        generated_at = dt.datetime.utcnow()
        stage = ctx.answer_stage
        items = max(int(stage.get("items") or len(result.qa_pairs) or 1), 1)
        avg_prompt = _safe_div(stage.get("prompt_tokens"), items)
        avg_completion = _safe_div(stage.get("completion_tokens"), items)
        avg_latency_ms = _safe_div(stage.get("stage_duration_seconds"), items) * 1000.0
        call_metrics_by_question: Dict[tuple[str, str], Dict[str, Any]] = {}
        for record in result.call_records or []:
            if not isinstance(record, dict):
                continue
            if record.get("call_type") != "answer":
                continue
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            mapping_keys: list[tuple[str, str]] = []
            question_key = metadata.get("question")
            if isinstance(question_key, str):
                mapping_keys.append(("question", question_key.strip()))
            qid_key = metadata.get("qid")
            if qid_key:
                mapping_keys.append(("qid", str(qid_key)))
            if not mapping_keys:
                continue
            prompt_tokens_value = _coerce_int(metadata.get("prompt_tokens"))
            completion_tokens_value = _coerce_int(metadata.get("completion_tokens"))
            duration_seconds_value = _coerce_float(metadata.get("duration_seconds"))
            metrics_payload = {
                "prompt_tokens": prompt_tokens_value,
                "completion_tokens": completion_tokens_value,
                "latency_ms": (
                    duration_seconds_value * 1000.0 if duration_seconds_value is not None else None
                ),
            }
            for mapping_key in mapping_keys:
                call_metrics_by_question[mapping_key] = metrics_payload

        with self.conn.cursor() as cur:
            for pair in result.qa_pairs:
                question = pair.get("instruction") or pair.get("question") or ""
                answer = pair.get("output") or pair.get("answer") or ""
                if not question or not answer:
                    continue
                metrics = {}
                qid_value = pair.get("qid")
                if qid_value is not None:
                    metrics = call_metrics_by_question.get(("qid", str(qid_value))) or {}
                if not metrics:
                    metrics = call_metrics_by_question.get(("question", question.strip())) or {}
                per_pair_prompt = metrics.get("prompt_tokens")
                per_pair_completion = metrics.get("completion_tokens")
                per_pair_latency = metrics.get("latency_ms")
                if per_pair_prompt is None and avg_prompt:
                    per_pair_prompt = int(avg_prompt)
                if per_pair_completion is None and avg_completion:
                    per_pair_completion = int(avg_completion)
                if per_pair_latency is None and avg_latency_ms:
                    per_pair_latency = avg_latency_ms
                cur.execute(
                    """
                    INSERT INTO sdc_ai.qa_pair (
                        payload_id, run_id, source_id, model_id,
                        question, answer, answer_format, model_name,
                        prompt_tokens, completion_tokens, latency_ms,
                        confidence_score, status, generated_at, updated_at
                    )
                    VALUES (%(payload_id)s, %(run_id)s, %(source_id)s, %(model_id)s,
                            %(question)s, %(answer)s, %(answer_format)s, %(model_name)s,
                            %(prompt_tokens)s, %(completion_tokens)s, %(latency_ms)s,
                            %(confidence_score)s, %(status)s, %(generated_at)s, %(generated_at)s)
                    RETURNING qa_id
                    """,
                    {
                        **template,
                        "payload_id": payload_id,
                        "question": question,
                        "answer": answer,
                        "prompt_tokens": per_pair_prompt,
                        "completion_tokens": per_pair_completion,
                        "latency_ms": per_pair_latency,
                         "generated_at": generated_at,
                    },
                )
                qa_id = cur.fetchone()["qa_id"]

                metadata = {
                    "label": pair.get("label"),
                    "tag_path": pair.get("tag-path"),
                    "context": pair.get("context"),
                }
                if any(metadata.values()) and self.config.toggles.save_pipeline_events:
                    cur.execute(
                        """
                        INSERT INTO sdc_ai.qa_pipeline_event (
                            run_id, qa_id, event_type, status, message, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            self.run_id,
                            qa_id,
                            "qa_pair_metadata",
                            "success",
                            None,
                            Json(metadata),
                        ),
                    )
        self.conn.commit()

    def _insert_pipeline_events(self, records: Iterable[Dict[str, Any]]) -> None:
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            for record in records:
                cur.execute(
                    """
                    INSERT INTO sdc_ai.qa_pipeline_event (
                        run_id, event_type, status, message, metadata, occurred_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        self.run_id,
                        f"llm_call:{record.get('stage', 'unknown')}",
                        "success",
                        None,
                        Json(record),
                    ),
                )
        self.conn.commit()


# ---------------------------------------------------------------------- #
# Supporting classes                                                     #
# ---------------------------------------------------------------------- #


class _PersistenceContext:
    def __init__(
        self,
        config: PersistenceConfig,
        performance: Dict[str, Any],
        monitor: Optional[PerformanceMonitor],
        *,
        source_id: int,
        model_id: int,
        run_id: int,
    ) -> None:
        self.config = config
        self.performance = performance
        self.monitor = monitor
        self.source_id = source_id
        self.model_id = model_id
        self.run_id = run_id
        self.payload_type_value = config.tags.get("payload_type", "text")

    @property
    def payload_type(self) -> str:
        return self.payload_type_value or "text"

    def qa_template(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "model_id": self.model_id,
            "answer_format": "text",
            "model_name": self.config.model_key or "unknown-model",
            "confidence_score": None,
            "status": "success",
        }

    @property
    def answer_stage(self) -> Dict[str, Any]:
        return self.performance.get("stages", {}).get("answer_generation", {})


class _RunAccumulator:
    def __init__(self, source_key: str, model_key: str):
        self.source_key = source_key
        self.model_key = model_key
        self.reset()

    def reset(self) -> None:
        self.total_questions = 0
        self.success_pairs = 0
        self.failed_pairs = 0
        self.answer_prompt_tokens = 0
        self.answer_completion_tokens = 0
        self.answer_latency_seconds = 0.0
        self.answer_request_count = 0
        self.answer_items = 0
        self.workflow_prompt_tokens = 0
        self.workflow_completion_tokens = 0
        self.workflow_total_tokens = 0
        self.workflow_request_count = 0
        self.workflow_duration_seconds = 0.0
        self.workflow_tokens_per_second = 0.0
        self.workflow_overall_qpm = 0.0
        self.workflow_tokens_per_request = 0.0
        self.files_processed = 0
        self.files_failed = 0

    def consume(self, result: QAGenerationResult) -> None:
        qa_count = len(result.qa_pairs)
        self.files_processed += 1
        metadata = result.metadata or {}
        self.total_questions += metadata.get("total_questions_generated", qa_count)
        self.success_pairs += qa_count

        perf_stage = (result.performance or {}).get("stages", {}).get("answer_generation", {})
        prompt_tokens = int(perf_stage.get("prompt_tokens") or 0)
        completion_tokens = int(perf_stage.get("completion_tokens") or 0)
        duration = float(perf_stage.get("stage_duration_seconds") or 0.0)
        request_count = int(perf_stage.get("request_count") or 0)
        self.answer_prompt_tokens += prompt_tokens
        self.answer_completion_tokens += completion_tokens
        self.answer_latency_seconds += duration
        items = int(perf_stage.get("items") or qa_count or 1)
        self.answer_items += items
        if request_count:
            self.answer_request_count += request_count
        else:
            self.answer_request_count += items

        totals = (result.performance or {}).get("totals") or {}
        self.workflow_prompt_tokens = int(totals.get("prompt_tokens") or self.answer_prompt_tokens)
        self.workflow_completion_tokens = int(totals.get("completion_tokens") or self.answer_completion_tokens)
        self.workflow_total_tokens = int(
            totals.get("total_tokens")
            or (self.workflow_prompt_tokens + self.workflow_completion_tokens)
        )
        self.workflow_request_count = int(
            totals.get("request_count") or self.answer_request_count
        )
        if self.workflow_request_count <= 0 and self.success_pairs:
            self.workflow_request_count = self.success_pairs
        self.workflow_duration_seconds = float(
            totals.get("workflow_duration_seconds") or duration
        )
        self.workflow_tokens_per_second = float(totals.get("tokens_per_second") or 0.0)
        self.workflow_overall_qpm = float(totals.get("overall_qpm") or 0.0)
        self.workflow_tokens_per_request = _safe_div(
            self.workflow_total_tokens, self.workflow_request_count
        )
        if not self.workflow_tokens_per_second and self.workflow_duration_seconds:
            self.workflow_tokens_per_second = _safe_div(
                self.workflow_total_tokens, self.workflow_duration_seconds
            )
        if not self.workflow_overall_qpm and self.workflow_duration_seconds:
            self.workflow_overall_qpm = _safe_div(
                self.workflow_request_count, self.workflow_duration_seconds
            ) * 60.0

    def hourly_row(self) -> Dict[str, Any]:
        now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        avg_latency_ms = _safe_div(self.answer_latency_seconds, self.answer_items) * 1000.0
        return {
            "bucket_start": now,
            "bucket_end": now + dt.timedelta(hours=1),
            "total_generated": self.success_pairs + self.failed_pairs,
            "success_count": self.success_pairs,
            "failure_count": self.failed_pairs,
            "total_prompt_tokens": self.workflow_prompt_tokens,
            "total_completion_tokens": self.workflow_completion_tokens,
            "total_tokens": self.workflow_total_tokens,
            "tokens_per_second": self.workflow_tokens_per_second or None,
            "overall_qpm": self.workflow_overall_qpm or None,
            "total_duration_seconds": self.workflow_duration_seconds or None,
            "tokens_per_request": self.workflow_tokens_per_request or None,
            "avg_latency_ms": avg_latency_ms if avg_latency_ms else None,
            "p95_latency_ms": None,
            "avg_prompt_tokens": _safe_div(self.answer_prompt_tokens, max(self.success_pairs, 1)),
            "avg_completion_tokens": _safe_div(self.answer_completion_tokens, max(self.success_pairs, 1)),
            "avg_confidence": None,
            "source_breakdown": {self.source_key: self.success_pairs},
            "model_breakdown": {self.model_key: self.success_pairs},
        }

    def daily_row(self) -> Dict[str, Any]:
        today = dt.datetime.utcnow().date()
        avg_latency_ms = _safe_div(self.answer_latency_seconds, self.answer_items) * 1000.0
        return {
            "date_key": today,
            "total_generated": self.success_pairs + self.failed_pairs,
            "success_count": self.success_pairs,
            "failure_count": self.failed_pairs,
            "total_prompt_tokens": self.workflow_prompt_tokens,
            "total_completion_tokens": self.workflow_completion_tokens,
            "total_tokens": self.workflow_total_tokens,
            "tokens_per_second": self.workflow_tokens_per_second or None,
            "overall_qpm": self.workflow_overall_qpm or None,
            "total_duration_seconds": self.workflow_duration_seconds or None,
            "tokens_per_request": self.workflow_tokens_per_request or None,
            "avg_latency_ms": avg_latency_ms if avg_latency_ms else None,
            "p95_latency_ms": None,
            "avg_eval_score": None,
            "avg_confidence": None,
            "source_breakdown": {self.source_key: self.success_pairs},
            "model_breakdown": {self.model_key: self.success_pairs},
        }


def _safe_div(value, divisor):
    try:
        value = float(value or 0.0)
        divisor = float(divisor or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if divisor <= 0:
        return 0.0
    return value / divisor


def _coerce_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
