from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db_session
from ...repositories import AlertRepository
from ...schemas.alert import Alert

router = APIRouter(prefix="/alerts")


@router.get("", response_model=List[Alert])
async def list_alerts(session: AsyncSession = Depends(get_db_session)) -> List[Alert]:
    repo = AlertRepository(session)
    alerts = await repo.list()
    return [Alert.model_validate(alert) for alert in alerts]


@router.post("/{alert_id}/ack", response_model=Alert)
async def acknowledge_alert(
    alert_id: UUID, session: AsyncSession = Depends(get_db_session)
) -> Alert:
    repo = AlertRepository(session)
    try:
        alert = await repo.mark_acknowledged(alert_id)
        await session.commit()
        await session.refresh(alert)
        return Alert.model_validate(alert)
    except KeyError:
        await session.rollback()
        raise HTTPException(status_code=404, detail="Alert not found")
