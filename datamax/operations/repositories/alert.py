from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Alert


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> Sequence[Alert]:
        result = await self.session.execute(select(Alert).order_by(Alert.created_at.desc()))
        return result.scalars().all()

    async def get(self, alert_id: UUID) -> Alert:
        alert = await self.session.get(Alert, alert_id)
        if not alert:
            raise KeyError(f"Alert {alert_id} not found")
        return alert

    async def mark_acknowledged(self, alert_id: UUID) -> Alert:
        alert = await self.get(alert_id)
        alert.acknowledged = True
        await self.session.flush()
        return alert
