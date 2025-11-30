from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db_session
from ...models import Dataset as DatasetORM
from ...repositories import DatasetRepository
from ...schemas.dataset import Dataset, DatasetCreate, DatasetUpdate

router = APIRouter(prefix="/datasets")


@router.get("", response_model=List[Dataset])
async def list_datasets(session: AsyncSession = Depends(get_db_session)) -> List[Dataset]:
    repo = DatasetRepository(session)
    datasets = await repo.list()
    return [Dataset.model_validate(dataset) for dataset in datasets]


@router.post("", response_model=Dataset, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    payload: DatasetCreate,
    session: AsyncSession = Depends(get_db_session),
) -> Dataset:
    repo = DatasetRepository(session)
    dataset = DatasetORM(
        datasource_id=payload.datasource_id,
        name=payload.name,
        description=payload.description,
        schema_version=payload.schema_version,
        record_count=payload.record_count,
        format=payload.format,
        tags=payload.tags,
    )
    await repo.create(dataset)
    await session.commit()
    await session.refresh(dataset)
    return Dataset.model_validate(dataset)


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> Dataset:
    repo = DatasetRepository(session)
    try:
        dataset = await repo.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return Dataset.model_validate(dataset)


@router.put("/{dataset_id}", response_model=Dataset)
async def update_dataset(
    dataset_id: UUID,
    payload: DatasetUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> Dataset:
    repo = DatasetRepository(session)
    try:
        dataset = await repo.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(dataset, key, value)

    await session.commit()
    await session.refresh(dataset)
    return Dataset.model_validate(dataset)


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_dataset(
    dataset_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    repo = DatasetRepository(session)
    try:
        await repo.delete(dataset_id)
        await session.commit()
    except KeyError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")


@router.post("/{dataset_id}/refresh", response_model=Dataset)
async def refresh_dataset(
    dataset_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> Dataset:
    repo = DatasetRepository(session)
    try:
        dataset = await repo.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    dataset.record_count = (dataset.record_count or 0) + 500
    dataset.last_refreshed_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(dataset)
    return Dataset.model_validate(dataset)
