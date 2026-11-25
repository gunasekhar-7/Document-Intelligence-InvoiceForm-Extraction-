"""Database connection and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from document_intelligence.config import settings


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str = None, echo: bool = False):
        self.database_url = database_url or settings.database_url
        self.echo = echo
        self._engine: AsyncEngine = None
        self._session_factory: async_sessionmaker[AsyncSession] = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            engine_kwargs = {
                "echo": self.echo,
                "pool_pre_ping": True,
            }
            if "sqlite" not in self.database_url:
                engine_kwargs.update(
                    {
                        "pool_size": 10,
                        "max_overflow": 20,
                        "pool_recycle": 3600,
                    }
                )
            self._engine = create_async_engine(
                self.database_url,
                **engine_kwargs,
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._session_factory

    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception:
            return False


# Global instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with db_manager.session() as session:
        yield session


async def init_db():
    """Initialize database (create tables)."""
    import logging

    from document_intelligence.db.models import Base

    try:
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Database initialization skipped or deferred: %s", str(e)
        )


async def close_db():
    """Close database connections."""
    await db_manager.close()
