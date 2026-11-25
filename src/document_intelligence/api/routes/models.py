"""API routes for model management and versioning."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from document_intelligence.db.models import ModelVersion
from document_intelligence.db.session import get_db_session
from document_intelligence.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Any
    version: str
    model_type: str
    model_path: str
    metrics: dict
    training_config: dict | None = None
    is_active: bool
    deployed_at: Any = None
    created_at: Any = None


class ModelVersionListResponse(BaseModel):
    models: list[ModelVersionResponse]
    total: int
    page: int
    page_size: int


class SetActiveModelRequest(BaseModel):
    version: str


class ModelMetricsResponse(BaseModel):
    version: str
    metrics: dict
    field_metrics: dict | None = None


@router.get("/models", response_model=ModelVersionListResponse)
async def list_models(
    model_type: str | None = Query(None),
    is_active: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """List model versions with filtering and pagination."""
    query = select(ModelVersion)

    if model_type:
        query = query.where(ModelVersion.model_type == model_type)
    if is_active is not None:
        query = query.where(ModelVersion.is_active == is_active)

    # Total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Pagination
    query = query.order_by(desc(ModelVersion.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    models = result.scalars().all()

    return ModelVersionListResponse(
        models=[ModelVersionResponse.model_validate(m) for m in models],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/models/{model_id}", response_model=ModelVersionResponse)
async def get_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get model version by ID."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(404, "Model not found")

    return ModelVersionResponse.model_validate(model)


@router.get("/models/active/{model_type}", response_model=ModelVersionResponse)
async def get_active_model(
    model_type: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get currently active model for a model type."""
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_type == model_type)
        .where(ModelVersion.is_active)
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(404, f"No active model found for type: {model_type}")

    return ModelVersionResponse.model_validate(model)


@router.post("/models/{model_id}/activate", response_model=ModelVersionResponse)
async def activate_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Activate a model version (deactivates others of same type)."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(404, "Model not found")

    # Deactivate all models of same type
    await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_type == model.model_type)
        .where(ModelVersion.is_active)
    )
    # Actually update them
    from sqlalchemy import update

    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.model_type == model.model_type)
        .where(ModelVersion.is_active)
        .values(is_active=False)
    )

    # Activate this model
    model.is_active = True
    from sqlalchemy import func

    model.deployed_at = func.now()

    await db.commit()
    await db.refresh(model)

    logger.info("Model activated", model_id=str(model.id), version=model.version)

    return ModelVersionResponse.model_validate(model)


@router.post("/models", response_model=ModelVersionResponse, status_code=201)
async def register_model(
    version: str = Form(...),
    model_type: str = Form(...),
    model_path: str = Form(...),
    metrics: str = Form(...),  # JSON string
    training_config: str | None = Form(None),  # JSON string
    is_active: bool = Form(False),
    db: AsyncSession = Depends(get_db_session),
):
    """Register a new model version."""
    import json

    # Check if version already exists for this type
    existing = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.version == version)
        .where(ModelVersion.model_type == model_type)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Model version {version} already exists for type {model_type}")

    model = ModelVersion(
        version=version,
        model_type=model_type,
        model_path=model_path,
        metrics=json.loads(metrics),
        training_config=json.loads(training_config) if training_config else None,
        is_active=is_active,
    )

    # If activating, deactivate others
    if is_active:
        from sqlalchemy import update

        await db.execute(
            update(ModelVersion)
            .where(ModelVersion.model_type == model_type)
            .where(ModelVersion.is_active)
            .values(is_active=False)
        )
        from sqlalchemy import func

        model.deployed_at = func.now()

    db.add(model)
    await db.commit()
    await db.refresh(model)

    logger.info("Model registered", model_id=str(model.id), version=version, type=model_type)

    return ModelVersionResponse.model_validate(model)


@router.get("/models/{model_id}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get detailed metrics for a model version."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(404, "Model not found")

    return ModelMetricsResponse(
        version=model.version,
        metrics=model.metrics,
        field_metrics=model.metrics.get("field_metrics") if model.metrics else None,
    )


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a model version (only if not active)."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(404, "Model not found")

    if model.is_active:
        raise HTTPException(400, "Cannot delete active model. Deactivate first.")

    await db.delete(model)
    await db.commit()

    return {"message": "Model deleted successfully"}
