from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import MetricSummary


class MetricRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> Sequence[MetricSummary]:
        result = await self.session.execute(
            select(MetricSummary).order_by(MetricSummary.created_at.desc())
        )
        return result.scalars().all()

    async def upsert(self, summary: MetricSummary) -> MetricSummary:
        self.session.add(summary)
        await self.session.flush()
        return summary
