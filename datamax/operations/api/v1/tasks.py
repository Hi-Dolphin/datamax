from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db_session
from ...models import Task as TaskORM
from ...repositories import TaskRepository, TaskRunRepository
from ...schemas.task import Task, TaskCreate, TaskRun, TaskUpdate

router = APIRouter(prefix="/tasks")


@router.get("", response_model=List[Task])
async def list_tasks(session: AsyncSession = Depends(get_db_session)) -> List[Task]:
    repo = TaskRepository(session)
    tasks = await repo.list()
    return [Task.model_validate(task) for task in tasks]


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: TaskCreate,
    session: AsyncSession = Depends(get_db_session),
) -> Task:
    repo = TaskRepository(session)
    task = TaskORM(
        pipeline_id=payload.pipeline_id,
        dataset_id=payload.dataset_id,
        name=payload.name,
        description=payload.description,
        scheduled_at=payload.scheduled_at,
        parameters=payload.parameters or {},
        status="pending",
    )
    await repo.create(task)
    await session.commit()
    await session.refresh(task)
    return Task.model_validate(task)


@router.get("/{task_id}", response_model=Task)
async def get_task(
    task_id: UUID, session: AsyncSession = Depends(get_db_session)
) -> Task:
    repo = TaskRepository(session)
    try:
        task = await repo.get(task_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return Task.model_validate(task)


@router.put("/{task_id}", response_model=Task)
async def update_task(
    task_id: UUID,
    payload: TaskUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> Task:
    repo = TaskRepository(session)
    try:
        task = await repo.get(task_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(task, key, value)

    await session.commit()
    await session.refresh(task)
    return Task.model_validate(task)


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_task(
    task_id: UUID, session: AsyncSession = Depends(get_db_session)
) -> None:
    repo = TaskRepository(session)
    try:
        await repo.delete(task_id)
        await session.commit()
    except KeyError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")


@router.get("/{task_id}/runs", response_model=List[TaskRun])
async def list_task_runs(
    task_id: UUID, session: AsyncSession = Depends(get_db_session)
) -> List[TaskRun]:
    repo = TaskRunRepository(session)
    runs = await repo.list_for_task(task_id)
    return [TaskRun.model_validate(run) for run in runs]
