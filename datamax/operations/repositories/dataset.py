from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Dataset


class DatasetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> Sequence[Dataset]:
        result = await self.session.execute(
            select(Dataset).order_by(Dataset.created_at.desc())
        )
        return result.scalars().all()

    async def get(self, dataset_id: UUID) -> Dataset:
        dataset = await self.session.get(Dataset, dataset_id)
        if not dataset:
            raise KeyError(f"Dataset {dataset_id} not found")
        return dataset

    async def create(self, dataset: Dataset) -> Dataset:
        self.session.add(dataset)
        await self.session.flush()
        return dataset

    async def delete(self, dataset_id: UUID) -> None:
        dataset = await self.get(dataset_id)
        await self.session.delete(dataset)
