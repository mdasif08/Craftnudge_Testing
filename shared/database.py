"""
Enterprise-grade database layer for CraftNudge microservices.

This module provides comprehensive database functionality with:
- Connection pooling and management
- Database migrations and schema management
- Repository pattern implementation
- Transaction management
- Query optimization and caching
- Database health monitoring
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from datetime import datetime, timedelta
from functools import wraps

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.sql import select, update, delete, func
from sqlalchemy.dialects.postgresql import insert

from config.settings import get_database_url, settings
from shared.models import (
    Base,
    CommitModel,
    AnalysisModel,
    BehaviorPatternModel,
    UserBehaviorModel,
    Commit,
    Analysis,
    BehaviorPattern,
    UserBehavior,
    ModelConverter,
    ModelValidator,
)

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration class."""
    
    def __init__(self, url: str, pool_size: int = 10, max_overflow: int = 20):
        self.url = url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
    
    @classmethod
    def from_settings(cls):
        """Create DatabaseConfig from settings."""
        return cls(
            url=get_database_url(),
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
        )


class DatabaseManager:
    """Database connection and session management."""

    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self._initialized = False

    def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return

        # Create synchronous engine for migrations and sync operations
        self.engine = create_engine(
            get_database_url(),
            poolclass=QueuePool,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            echo=settings.database.echo,
            connect_args={"connect_timeout": 10, "application_name": "craftnudge"},
        )

        # Create async engine for async operations
        async_url = get_database_url().replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_url,
            poolclass=QueuePool,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            echo=settings.database.echo,
            connect_args={"connect_timeout": 10, "application_name": "craftnudge_async"},
        )

        # Create session factories
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.AsyncSessionLocal = async_sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

        self._initialized = True
        logger.info("Database manager initialized successfully")

    def create_tables(self):
        """Create all database tables."""
        if not self._initialized:
            self.initialize()

        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all database tables."""
        if not self._initialized:
            self.initialize()

        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped successfully")

    def get_sync_session(self) -> Session:
        """Get a synchronous database session."""
        if not self._initialized:
            self.initialize()

        return self.SessionLocal()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session."""
        if not self._initialized:
            self.initialize()

        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                result.fetchone()

                # Check connection pool status
                pool_status = {
                    "pool_size": self.async_engine.pool.size(),
                    "checked_in": self.async_engine.pool.checkedin(),
                    "checked_out": self.async_engine.pool.checkedout(),
                    "overflow": self.async_engine.pool.overflow(),
                }

                return {
                    "status": "healthy",
                    "pool_status": pool_status,
                    "timestamp": datetime.utcnow(),
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow()}

    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
        if self.async_engine:
            asyncio.create_task(self.async_engine.dispose())
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Session:
    """Get database session for dependency injection."""
    return db_manager.get_sync_session()


async def get_async_db() -> AsyncSession:
    """Get async database session for dependency injection."""
    async with db_manager.get_async_session() as session:
        yield session


def database_transaction(func):
    """Decorator for database transaction management."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with db_manager.get_async_session() as session:
            try:
                result = await func(*args, **kwargs, session=session)
                return result
            except SQLAlchemyError as e:
                logger.error(f"Database transaction failed: {e}")
                raise

    return wrapper


class BaseRepository:
    """Base repository class with common database operations."""

    def __init__(self, model_class):
        self.model_class = model_class

    @database_transaction
    async def create(self, data: Dict[str, Any], session: AsyncSession) -> Any:
        """Create a new record."""
        try:
            instance = self.model_class(**data)
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance
        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.model_class.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating {self.model_class.__name__}: {e}")
            raise

    @database_transaction
    async def get_by_id(self, record_id: str, session: AsyncSession) -> Optional[Any]:
        """Get record by ID."""
        try:
            result = await session.execute(
                select(self.model_class).where(self.model_class.id == record_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model_class.__name__} by ID: {e}")
            raise

    @database_transaction
    async def get_all(
        self, skip: int = 0, limit: int = 100, session: AsyncSession = None
    ) -> List[Any]:
        """Get all records with pagination."""
        try:
            result = await session.execute(
                select(self.model_class)
                .offset(skip)
                .limit(limit)
                .order_by(self.model_class.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting all {self.model_class.__name__}: {e}")
            raise

    @database_transaction
    async def update(
        self, record_id: str, data: Dict[str, Any], session: AsyncSession
    ) -> Optional[Any]:
        """Update a record."""
        try:
            result = await session.execute(
                update(self.model_class)
                .where(self.model_class.id == record_id)
                .values(**data, updated_at=datetime.utcnow())
                .returning(self.model_class)
            )
            updated_record = result.scalar_one_or_none()
            if updated_record:
                await session.refresh(updated_record)
            return updated_record
        except Exception as e:
            logger.error(f"Error updating {self.model_class.__name__}: {e}")
            raise

    @database_transaction
    async def delete(self, record_id: str, session: AsyncSession) -> bool:
        """Delete a record."""
        try:
            result = await session.execute(
                delete(self.model_class).where(self.model_class.id == record_id)
            )
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting {self.model_class.__name__}: {e}")
            raise

    @database_transaction
    async def count(self, session: AsyncSession) -> int:
        """Get total count of records."""
        try:
            result = await session.execute(select(func.count(self.model_class.id)))
            return result.scalar()
        except Exception as e:
            logger.error(f"Error counting {self.model_class.__name__}: {e}")
            raise


class CommitRepository(BaseRepository):
    """Repository for commit operations."""

    def __init__(self):
        super().__init__(CommitModel)

    @database_transaction
    async def get_by_hash(self, commit_hash: str, session: AsyncSession) -> Optional[CommitModel]:
        """Get commit by hash."""
        try:
            result = await session.execute(
                select(CommitModel).where(CommitModel.hash == commit_hash)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting commit by hash: {e}")
            raise

    @database_transaction
    async def get_by_author(
        self, author: str, skip: int = 0, limit: int = 100, session: AsyncSession = None
    ) -> List[CommitModel]:
        """Get commits by author."""
        try:
            result = await session.execute(
                select(CommitModel)
                .where(CommitModel.author == author)
                .offset(skip)
                .limit(limit)
                .order_by(CommitModel.timestamp.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting commits by author: {e}")
            raise

    @database_transaction
    async def get_by_repository(
        self, repository: str, skip: int = 0, limit: int = 100, session: AsyncSession = None
    ) -> List[CommitModel]:
        """Get commits by repository."""
        try:
            result = await session.execute(
                select(CommitModel)
                .where(CommitModel.repository == repository)
                .offset(skip)
                .limit(limit)
                .order_by(CommitModel.timestamp.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting commits by repository: {e}")
            raise

    @database_transaction
    async def get_recent_commits(
        self, days: int = 7, limit: int = 100, session: AsyncSession = None
    ) -> List[CommitModel]:
        """Get recent commits within specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = await session.execute(
                select(CommitModel)
                .where(CommitModel.timestamp >= cutoff_date)
                .limit(limit)
                .order_by(CommitModel.timestamp.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting recent commits: {e}")
            raise

    @database_transaction
    async def get_quality_stats(self, session: AsyncSession) -> Dict[str, Any]:
        """Get commit quality statistics."""
        try:
            # Average quality score
            avg_quality = await session.execute(
                select(func.avg(CommitModel.quality_score)).where(
                    CommitModel.quality_score.isnot(None)
                )
            )

            # Quality distribution
            quality_dist = await session.execute(
                select(CommitModel.quality_level, func.count(CommitModel.id))
                .where(CommitModel.quality_level.isnot(None))
                .group_by(CommitModel.quality_level)
            )

            # Total commits
            total_commits = await session.execute(select(func.count(CommitModel.id)))

            return {
                "average_quality": avg_quality.scalar() or 0.0,
                "quality_distribution": dict(quality_dist.all()),
                "total_commits": total_commits.scalar() or 0,
            }
        except Exception as e:
            logger.error(f"Error getting quality stats: {e}")
            raise


class AnalysisRepository(BaseRepository):
    """Repository for analysis operations."""

    def __init__(self):
        super().__init__(AnalysisModel)

    @database_transaction
    async def get_by_commit_id(
        self, commit_id: str, session: AsyncSession
    ) -> Optional[AnalysisModel]:
        """Get analysis by commit ID."""
        try:
            result = await session.execute(
                select(AnalysisModel).where(AnalysisModel.commit_id == commit_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting analysis by commit ID: {e}")
            raise

    @database_transaction
    async def get_by_status(self, status: str, session: AsyncSession) -> List[AnalysisModel]:
        """Get analyses by status."""
        try:
            result = await session.execute(
                select(AnalysisModel).where(AnalysisModel.status == status)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting analyses by status: {e}")
            raise

    @database_transaction
    async def get_analysis_stats(self, session: AsyncSession) -> Dict[str, Any]:
        """Get analysis statistics."""
        try:
            # Average quality score
            avg_quality = await session.execute(select(func.avg(AnalysisModel.quality_score)))

            # Average analysis duration
            avg_duration = await session.execute(select(func.avg(AnalysisModel.analysis_duration)))

            # Status distribution
            status_dist = await session.execute(
                select(AnalysisModel.status, func.count(AnalysisModel.id)).group_by(
                    AnalysisModel.status
                )
            )

            return {
                "average_quality": avg_quality.scalar() or 0.0,
                "average_duration": avg_duration.scalar() or 0.0,
                "status_distribution": dict(status_dist.all()),
            }
        except Exception as e:
            logger.error(f"Error getting analysis stats: {e}")
            raise


class BehaviorPatternRepository(BaseRepository):
    """Repository for behavior pattern operations."""

    def __init__(self):
        super().__init__(BehaviorPatternModel)

    @database_transaction
    async def get_by_user_id(
        self, user_id: str, session: AsyncSession
    ) -> List[BehaviorPatternModel]:
        """Get behavior patterns by user ID."""
        try:
            result = await session.execute(
                select(BehaviorPatternModel)
                .where(BehaviorPatternModel.user_id == user_id)
                .order_by(BehaviorPatternModel.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting behavior patterns by user ID: {e}")
            raise

    @database_transaction
    async def get_by_pattern_type(
        self, pattern_type: str, session: AsyncSession
    ) -> List[BehaviorPatternModel]:
        """Get behavior patterns by type."""
        try:
            result = await session.execute(
                select(BehaviorPatternModel)
                .where(BehaviorPatternModel.pattern_type == pattern_type)
                .order_by(BehaviorPatternModel.confidence.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting behavior patterns by type: {e}")
            raise

    @database_transaction
    async def get_active_patterns(self, session: AsyncSession) -> List[BehaviorPatternModel]:
        """Get active behavior patterns."""
        try:
            result = await session.execute(
                select(BehaviorPatternModel)
                .where(BehaviorPatternModel.is_active == True)
                .order_by(BehaviorPatternModel.confidence.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting active behavior patterns: {e}")
            raise


class UserBehaviorRepository(BaseRepository):
    """Repository for user behavior operations."""

    def __init__(self):
        super().__init__(UserBehaviorModel)

    @database_transaction
    async def get_by_user_id(
        self, user_id: str, session: AsyncSession
    ) -> Optional[UserBehaviorModel]:
        """Get user behavior by user ID."""
        try:
            result = await session.execute(
                select(UserBehaviorModel).where(UserBehaviorModel.user_id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user behavior by user ID: {e}")
            raise

    @database_transaction
    async def upsert_user_behavior(
        self, user_id: str, data: Dict[str, Any], session: AsyncSession
    ) -> UserBehaviorModel:
        """Upsert user behavior data."""
        try:
            # PostgreSQL upsert
            stmt = insert(UserBehaviorModel).values(
                user_id=user_id, **data, last_updated=datetime.utcnow()
            )

            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "commit_frequency": stmt.excluded.commit_frequency,
                    "average_commit_size": stmt.excluded.average_commit_size,
                    "common_patterns": stmt.excluded.common_patterns,
                    "improvement_areas": stmt.excluded.improvement_areas,
                    "quality_trend": stmt.excluded.quality_trend,
                    "last_updated": stmt.excluded.last_updated,
                },
            )

            result = await session.execute(stmt)
            await session.flush()

            # Get the updated record
            user_behavior = await self.get_by_user_id(user_id, session)
            return user_behavior
        except Exception as e:
            logger.error(f"Error upserting user behavior: {e}")
            raise


# Repository instances
commit_repo = CommitRepository()
analysis_repo = AnalysisRepository()
behavior_pattern_repo = BehaviorPatternRepository()
user_behavior_repo = UserBehaviorRepository()


class DatabaseService:
    """High-level database service with business logic."""

    @staticmethod
    async def store_commit(commit_data: Dict[str, Any]) -> Commit:
        """Store a new commit."""
        # Validate commit data
        errors = ModelValidator.validate_commit_data(commit_data)
        if errors:
            raise ValueError(f"Invalid commit data: {errors}")

        # Check if commit already exists
        existing_commit = await commit_repo.get_by_hash(commit_data["hash"])
        if existing_commit:
            logger.warning(f"Commit {commit_data['hash']} already exists")
            return ModelConverter.model_to_commit(existing_commit)

        # Create commit
        commit_model = await commit_repo.create(commit_data)
        return ModelConverter.model_to_commit(commit_model)

    @staticmethod
    async def store_analysis(analysis_data: Dict[str, Any]) -> Analysis:
        """Store a new analysis."""
        # Validate analysis data
        errors = ModelValidator.validate_analysis_data(analysis_data)
        if errors:
            raise ValueError(f"Invalid analysis data: {errors}")

        # Create analysis
        analysis_model = await analysis_repo.create(analysis_data)
        return ModelConverter.model_to_analysis(analysis_model)

    @staticmethod
    async def get_commit_with_analysis(commit_id: str) -> Optional[Dict[str, Any]]:
        """Get commit with its analysis."""
        commit_model = await commit_repo.get_by_id(commit_id)
        if not commit_model:
            return None

        analysis_model = await analysis_repo.get_by_commit_id(commit_id)

        return {
            "commit": ModelConverter.model_to_commit(commit_model),
            "analysis": (
                ModelConverter.model_to_analysis(analysis_model) if analysis_model else None
            ),
        }

    @staticmethod
    async def get_user_behavior_summary(user_id: str) -> Optional[UserBehavior]:
        """Get user behavior summary."""
        user_behavior_model = await user_behavior_repo.get_by_user_id(user_id)
        if not user_behavior_model:
            return None

        # Convert to Pydantic model
        return UserBehavior(
            user_id=user_behavior_model.user_id,
            commit_frequency=user_behavior_model.commit_frequency,
            average_commit_size=user_behavior_model.average_commit_size,
            common_patterns=user_behavior_model.common_patterns,
            improvement_areas=user_behavior_model.improvement_areas,
            quality_trend=user_behavior_model.quality_trend,
            last_updated=user_behavior_model.last_updated,
        )

    @staticmethod
    async def update_user_behavior(user_id: str, behavior_data: Dict[str, Any]) -> UserBehavior:
        """Update user behavior data."""
        user_behavior_model = await user_behavior_repo.upsert_user_behavior(user_id, behavior_data)

        return UserBehavior(
            user_id=user_behavior_model.user_id,
            commit_frequency=user_behavior_model.commit_frequency,
            average_commit_size=user_behavior_model.average_commit_size,
            common_patterns=user_behavior_model.common_patterns,
            improvement_areas=user_behavior_model.improvement_areas,
            quality_trend=user_behavior_model.quality_trend,
            last_updated=user_behavior_model.last_updated,
        )


# Database initialization
def init_database():
    """Initialize database tables and connections."""
    db_manager.initialize()
    db_manager.create_tables()
    logger.info("Database initialized successfully")


async def close_database():
    """Close database connections."""
    db_manager.close()
    logger.info("Database connections closed")


# Health check endpoint
async def get_database_health() -> Dict[str, Any]:
    """Get database health status."""
    return await db_manager.health_check()


# Export commonly used functions and classes
__all__ = [
    "DatabaseManager",
    "BaseRepository",
    "CommitRepository",
    "AnalysisRepository",
    "BehaviorPatternRepository",
    "UserBehaviorRepository",
    "DatabaseService",
    "get_db",
    "get_async_db",
    "database_transaction",
    "init_database",
    "close_database",
    "get_database_health",
    "db_manager",
]
