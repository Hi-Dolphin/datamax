from __future__ import annotations

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import AppSettings, get_settings
from ..core.database import get_session


def get_settings_dep() -> AppSettings:
    return get_settings()


async def get_db_session(
    settings: AppSettings = Depends(get_settings_dep),
) -> AsyncIterator[AsyncSession]:
    async for session in get_session(settings):
        yield session
