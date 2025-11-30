from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Task, TaskRun


class TaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> Sequence[Task]:
        result = await self.session.execute(select(Task).order_by(Task.created_at.desc()))
        return result.scalars().unique().all()

    async def get(self, task_id: UUID) -> Task:
        task = await self.session.get(Task, task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        return task

    async def create(self, task: Task) -> Task:
        self.session.add(task)
        await self.session.flush()
        return task

    async def delete(self, task_id: UUID) -> None:
        task = await self.get(task_id)
        await self.session.delete(task)


class TaskRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_for_task(self, task_id: UUID) -> Sequence[TaskRun]:
        result = await self.session.execute(
            select(TaskRun)
            .where(TaskRun.task_id == task_id)
            .order_by(TaskRun.created_at.desc())
        )
        return result.scalars().all()

    async def create(self, run: TaskRun) -> TaskRun:
        self.session.add(run)
        await self.session.flush()
        return run
