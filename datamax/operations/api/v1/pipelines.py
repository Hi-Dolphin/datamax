from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db_session
from ...models import Pipeline as PipelineORM, PipelineStage as PipelineStageORM
from ...repositories import PipelineRepository
from ...schemas.pipeline import Pipeline, PipelineCreate, PipelineStage, PipelineUpdate

router = APIRouter(prefix="/pipelines")


def _build_stages(payload_stages: list[PipelineStage], pipeline_id: UUID | None = None) -> list[PipelineStageORM]:
    stages: list[PipelineStageORM] = []
    for stage in payload_stages:
        stages.append(
            PipelineStageORM(
                id=stage.stage_id,
                pipeline_id=pipeline_id,
                name=stage.name,
                kind=stage.kind,
                order=stage.order,
                config=stage.config,
                is_optional=stage.is_optional,
                stats=stage.stats,
            )
        )
    return stages


@router.get("", response_model=List[Pipeline])
async def list_pipelines(session: AsyncSession = Depends(get_db_session)) -> List[Pipeline]:
    repo = PipelineRepository(session)
    pipelines = await repo.list()
    return [Pipeline.model_validate(pipeline) for pipeline in pipelines]


@router.post("", response_model=Pipeline, status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    payload: PipelineCreate,
    session: AsyncSession = Depends(get_db_session),
) -> Pipeline:
    repo = PipelineRepository(session)
    pipeline = PipelineORM(
        name=payload.name,
        description=payload.description,
        project=payload.project,
        tags=payload.tags,
        status="active",
    )
    pipeline.stages = _build_stages(payload.stages)
    await repo.create(pipeline)
    await session.commit()
    await session.refresh(pipeline)
    return Pipeline.model_validate(pipeline)


@router.get("/{pipeline_id}", response_model=Pipeline)
async def get_pipeline(
    pipeline_id: UUID, session: AsyncSession = Depends(get_db_session)
) -> Pipeline:
    repo = PipelineRepository(session)
    try:
        pipeline = await repo.get(pipeline_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")
    return Pipeline.model_validate(pipeline)


@router.put("/{pipeline_id}", response_model=Pipeline)
async def update_pipeline(
    pipeline_id: UUID,
    payload: PipelineUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> Pipeline:
    repo = PipelineRepository(session)
    try:
        pipeline = await repo.get(pipeline_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    for key, value in payload.model_dump(exclude_none=True, exclude={"stages"}).items():
        setattr(pipeline, key, value)

    if payload.stages is not None:
        pipeline.stages.clear()
        for stage in payload.stages:
            stage_orm = PipelineStageORM(
                name=stage.name,
                kind=stage.kind,
                order=stage.order,
                config=stage.config,
                is_optional=stage.is_optional,
                stats=stage.stats,
            )
            pipeline.stages.append(stage_orm)

    await session.commit()
    await session.refresh(pipeline)
    return Pipeline.model_validate(pipeline)


@router.delete(
    "/{pipeline_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_pipeline(
    pipeline_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    repo = PipelineRepository(session)
    try:
        await repo.delete(pipeline_id)
        await session.commit()
    except KeyError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")
