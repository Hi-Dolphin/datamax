from __future__ import annotations

from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db_session
from ...models import DataSource as DataSourceORM
from ...repositories import DataSourceRepository
from ...schemas.datasource import DataSource, DataSourceCreate, DataSourceUpdate

router = APIRouter(prefix="/datasources")


def _sanitize_connection(connection: dict) -> dict:
    if not connection:
        return {}
    masked = dict(connection)
    if "password" in masked and masked["password"]:
        masked["password"] = "******"
    return masked


@router.get("", response_model=List[DataSource])
async def list_datasources(session: AsyncSession = Depends(get_db_session)) -> List[DataSource]:
    repo = DataSourceRepository(session)
    records = await repo.list()
    return [
        DataSource.model_validate(
            {**record.__dict__, "connection": _sanitize_connection(record.connection)}
        )
        for record in records
    ]


@router.post("", response_model=DataSource, status_code=status.HTTP_201_CREATED)
async def create_datasource(
    payload: DataSourceCreate,
    session: AsyncSession = Depends(get_db_session),
) -> DataSource:
    repo = DataSourceRepository(session)
    datasource = DataSourceORM(
        name=payload.name,
        kind=payload.kind,
        description=payload.description,
        tags=payload.tags,
        connection=payload.connection if isinstance(payload.connection, dict) else payload.connection.model_dump(),
        is_active=payload.is_active,
        status="active",
    )
    try:
        await repo.create(datasource)
        await session.commit()
    except IntegrityError as exc:  # pragma: no cover - DB-specific branch
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return DataSource.model_validate(
        {**datasource.__dict__, "connection": _sanitize_connection(datasource.connection)}
    )


@router.get("/{datasource_id}", response_model=DataSource)
async def get_datasource(
    datasource_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> DataSource:
    repo = DataSourceRepository(session)
    try:
        datasource = await repo.get(datasource_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
    return DataSource.model_validate(
        {**datasource.__dict__, "connection": _sanitize_connection(datasource.connection)}
    )


@router.put("/{datasource_id}", response_model=DataSource)
async def update_datasource(
    datasource_id: UUID,
    payload: DataSourceUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> DataSource:
    repo = DataSourceRepository(session)
    try:
        datasource = await repo.get(datasource_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")

    update_data = payload.model_dump(exclude_none=True)
    for key, value in update_data.items():
        if key == "connection" and value is not None:
            setattr(datasource, key, value if isinstance(value, dict) else value.model_dump())
        else:
            setattr(datasource, key, value)
    await session.commit()
    await session.refresh(datasource)
    return DataSource.model_validate(
        {**datasource.__dict__, "connection": _sanitize_connection(datasource.connection)}
    )


@router.delete(
    "/{datasource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_datasource(
    datasource_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    repo = DataSourceRepository(session)
    try:
        await repo.delete(datasource_id)
        await session.commit()
    except KeyError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
