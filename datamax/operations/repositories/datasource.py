from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import DataSource


class DataSourceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> Sequence[DataSource]:
        result = await self.session.execute(
            select(DataSource).order_by(DataSource.created_at.desc())
        )
        return result.scalars().all()

    async def get(self, datasource_id: UUID) -> DataSource:
        datasource = await self.session.get(DataSource, datasource_id)
        if not datasource:
            raise KeyError(f"DataSource {datasource_id} not found")
        return datasource

    async def create(self, datasource: DataSource) -> DataSource:
        self.session.add(datasource)
        await self.session.flush()
        return datasource

    async def delete(self, datasource_id: UUID) -> None:
        datasource = await self.get(datasource_id)
        await self.session.delete(datasource)

