from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Alert,
    DataSource,
    Dataset,
    Pipeline,
    PipelineStage,
    Task,
    TaskRun,
)


async def seed_demo_data(session: AsyncSession) -> None:
    count = await session.scalar(select(DataSource).limit(1))
    if count:
        return

    datasource = DataSource(
        name="Postgres 主库",
        kind="database",
        description="生产数据源",
        tags=["prod", "postgres"],
        connection={
            "type": "postgres",
            "host": "postgres.internal",
            "port": 5432,
            "database": "datamax",
            "username": "service",
        },
        status="active",
    )

    dataset = Dataset(
        datasource=datasource,
        name="QA 训练集",
        description="QA 模型训练集合并数据",
        schema_version="v1",
        record_count=125_000,
        format="parquet",
        tags=["qa", "training"],
        last_refreshed_at=datetime.now(timezone.utc),
    )

    pipeline = Pipeline(
        name="QA 生产流程",
        description="生成 QA 数据流程",
        project="qa-prod",
        tags=["qa"],
        status="active",
    )

    pipeline.stages = [
        PipelineStage(
            name="问题生成",
            kind="question_generation",
            order=1,
            config={"model": "qwen-72b"},
            stats={
                "runs": 21,
                "items": 1275,
                "duration": 215.67,
                "reqs": 21,
                "qpm": 5.84,
                "total_tokens": 40_656,
                "tokens_per_sec": 188.51,
                "tokens_per_req": 1936.0,
            },
        ),
        PipelineStage(
            name="回答生成",
            kind="answer_generation",
            order=2,
            config={"model": "qwen-72b"},
            stats={
                "runs": 1,
                "items": 1266,
                "duration": 1041.24,
                "reqs": 1275,
                "qpm": 73.47,
                "total_tokens": 1_578_419,
                "tokens_per_sec": 1515.91,
                "tokens_per_req": 1237.98,
            },
        ),
        PipelineStage(
            name="质检评估",
            kind="quality_check",
            order=3,
            config={"validators": ["consistency", "toxicity"]},
            stats={
                "runs": 1,
                "items": 1266,
                "duration": 50.12,
                "reqs": 1275,
                "qpm": 73.47,
                "total_tokens": 5000,
                "tokens_per_sec": 99.8,
                "tokens_per_req": 3.92,
            },
        ),
    ]

    task = Task(
        pipeline=pipeline,
        dataset=dataset,
        name="QA 训练样本生成 #2025-11-05",
        description="批量生成 QA 样本用于训练",
        parameters={"batch_size": 256},
        status="running",
        started_at=datetime.now(timezone.utc),
    )

    run = TaskRun(
        task=task,
        status="succeeded",
        duration_seconds=1256.97,
        tokens_total=1_619_075,
        tokens_prompt=1_394_053,
        tokens_completion=225_022,
        requests=1_296,
        metrics={
            "qpm": 61.86,
            "throughput_series": [
                {"label": index + 1, "value": round(1288.08 * (0.9 + index * 0.015), 2)}
                for index in range(12)
            ],
            "history_series": [
                {"label": index + 1, "value": round(950 + index * 40, 2)}
                for index in range(12)
            ],
        },
        triggered_by="system",
    )

    alert = Alert(
        title="Pipeline 质检告警",
        description="质检失败率超过 5%",
        severity="warning",
        source="pipeline",
        source_id=pipeline.id,
    )

    session.add_all([datasource, dataset, pipeline, task, run, alert])
