from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Pipeline, PipelineStage


class PipelineRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> Sequence[Pipeline]:
        result = await self.session.execute(
            select(Pipeline).order_by(Pipeline.created_at.desc())
        )
        return result.scalars().unique().all()

    async def get(self, pipeline_id: UUID) -> Pipeline:
        pipeline = await self.session.get(Pipeline, pipeline_id)
        if not pipeline:
            raise KeyError(f"Pipeline {pipeline_id} not found")
        return pipeline

    async def create(self, pipeline: Pipeline) -> Pipeline:
        self.session.add(pipeline)
        await self.session.flush()
        return pipeline

    async def delete(self, pipeline_id: UUID) -> None:
        pipeline = await self.get(pipeline_id)
        await self.session.delete(pipeline)


class PipelineStageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(self, stages: list[PipelineStage]) -> None:
        self.session.add_all(stages)

