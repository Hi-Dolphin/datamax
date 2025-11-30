from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator, Callable

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import AppSettings


class Base(DeclarativeBase):
    pass


@lru_cache
def _build_engine(dsn: str) -> AsyncEngine:
    return create_async_engine(dsn, echo=False, future=True)


@lru_cache
def _build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def get_engine(settings: AppSettings) -> AsyncEngine:
    if not settings.postgres_dsn:
        raise RuntimeError("DATAMAX_POSTGRES_DSN is required when mock_mode is disabled.")
    return _build_engine(settings.postgres_dsn)


def get_sessionmaker(settings: AppSettings) -> async_sessionmaker[AsyncSession]:
    engine = get_engine(settings)
    return _build_sessionmaker(engine)


async def get_session(
    settings: AppSettings,
) -> AsyncIterator[AsyncSession]:
    session_factory = get_sessionmaker(settings)
    async with session_factory() as session:
        yield session
