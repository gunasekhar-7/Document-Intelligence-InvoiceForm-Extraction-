"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from document_intelligence.config import settings
from document_intelligence.db.models import Base

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create persistent in-memory test database engine with tables created."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide transactional test database session."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_invoice_data():
    """Sample invoice structured metadata for testing."""
    return {
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-01-15",
        "due_date": "2024-02-15",
        "vendor_name": "Acme Corporation",
        "vendor_address": "123 Main St\nNew York, NY 10001",
        "vendor_tax_id": "12-3456789",
        "currency": "USD",
        "subtotal": 1500.00,
        "tax_amount": 150.00,
        "total_amount": 1650.00,
        "line_items": [
            {
                "description": "Consulting Services",
                "quantity": 10,
                "unit_price": 150.00,
                "total": 1500.00,
            }
        ],
    }
