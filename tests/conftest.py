"""Common test fixtures for the application."""

import os
import warnings
from collections.abc import AsyncGenerator, Generator

os.environ.setdefault("ENV", "testing")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.app import app
from vectorize.config import settings
from vectorize.config.db import get_session
from vectorize.config.seed import seed_db


def pytest_configure(config: object) -> None:
    """Configure pytest to filter warnings for MPS pin_memory on Mac."""
    del config  # unused
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*not supported on MPS.*",
        category=UserWarning,
    )


@pytest.fixture(scope="session")
async def session() -> AsyncGenerator[AsyncSession]:
    """Create a test database engine.

    Returns:
        AsyncSession: SQLModel async session for database operations.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
        echo=False,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(test_engine) as session:
        await seed_db(session)

    async with AsyncSession(test_engine) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture(name="client")
def client_fixture(session: AsyncSession) -> Generator:
    """Create a test client for the FastAPI app.

    Args:
        session: Database session fixture.

    Returns:
        TestClient: Configured FastAPI test client.
    """

    def get_session_override() -> AsyncSession:
        return session

    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app, base_url=f"http://testserver{settings.prefix}")  # NOSONAR
    yield client

    app.dependency_overrides.clear()
