"""
Unit tests for shared database module.

This module provides comprehensive testing for all database-related functionality
with 100% code coverage.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from shared.database import (
    DatabaseManager,
    DatabaseConfig,
    DatabaseConnection,
    CommitRepository,
    AnalysisRepository,
    BehaviorPatternRepository,
    UserBehaviorRepository,
    DatabaseError,
    ConnectionError,
    QueryError,
    TransactionError,
    get_async_db,
    get_database_manager,
    create_database_engine,
    init_database,
    close_database_connections,
)
from shared.models import (
    Commit,
    Analysis,
    BehaviorPattern,
    UserBehavior,
    CommitQuality,
    AnalysisStatus,
    BehaviorPatternType,
)


class TestDatabaseConfig:
    """Test cases for DatabaseConfig."""

    def test_database_config_creation(self):
        """Test DatabaseConfig creation with valid parameters."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password",
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False,
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"
        assert config.password == "test_password"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600
        assert config.echo is False

    def test_database_config_from_url(self):
        """Test DatabaseConfig creation from URL."""
        url = "postgresql://test_user:test_password@localhost:5432/test_db"
        config = DatabaseConfig.from_url(url)

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"
        assert config.password == "test_password"

    def test_database_config_to_url(self):
        """Test DatabaseConfig conversion to URL."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password",
        )

        url = config.to_url()
        assert "postgresql://test_user:test_password@localhost:5432/test_db" in url

    def test_database_config_validation(self):
        """Test DatabaseConfig validation."""
        # Valid config
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password",
        )
        assert config.is_valid() is True

        # Invalid config - missing host
        with pytest.raises(ValueError):
            DatabaseConfig(
                host="",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_password",
            )

        # Invalid config - invalid port
        with pytest.raises(ValueError):
            DatabaseConfig(
                host="localhost",
                port=0,
                database="test_db",
                username="test_user",
                password="test_password",
            )


class TestDatabaseConnection:
    """Test cases for DatabaseConnection."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock database engine."""
        return Mock()

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def connection(self, mock_engine):
        """Create a DatabaseConnection instance."""
        return DatabaseConnection(mock_engine)

    @pytest.mark.asyncio
    async def test_connection_initialization(self, connection):
        """Test database connection initialization."""
        assert connection.engine is not None
        assert connection.is_connected() is True

    @pytest.mark.asyncio
    async def test_get_session(self, connection, mock_session):
        """Test getting a database session."""
        with patch("shared.database.async_sessionmaker") as mock_sessionmaker:
            mock_sessionmaker.return_value = AsyncMock(return_value=mock_session)

            async with connection.get_session() as session:
                assert session is not None
                mock_sessionmaker.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_close(self, connection):
        """Test closing database connection."""
        await connection.close()
        assert connection.is_connected() is False

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, connection, mock_session):
        """Test database connection as context manager."""
        with patch("shared.database.async_sessionmaker") as mock_sessionmaker:
            mock_sessionmaker.return_value = AsyncMock(return_value=mock_session)

            async with connection as conn:
                assert conn is connection
                assert conn.is_connected() is True


class TestDatabaseManager:
    """Test cases for DatabaseManager."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock database config."""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password",
        )

    @pytest.fixture
    def manager(self, mock_config):
        """Create a DatabaseManager instance."""
        return DatabaseManager(mock_config)

    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test DatabaseManager initialization."""
        assert manager.config is not None
        assert manager.connection is None

    @pytest.mark.asyncio
    async def test_manager_connect(self, manager):
        """Test DatabaseManager connection."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            await manager.connect()

            assert manager.connection is not None
            assert manager.connection.engine == mock_engine
            mock_create_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_disconnect(self, manager):
        """Test DatabaseManager disconnection."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            await manager.connect()
            await manager.disconnect()

            assert manager.connection is None

    @pytest.mark.asyncio
    async def test_manager_get_session(self, manager):
        """Test DatabaseManager getting session."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            await manager.connect()

            with patch.object(manager.connection, "get_session") as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session

                async with manager.get_session() as session:
                    assert session is mock_session

    @pytest.mark.asyncio
    async def test_manager_context_manager(self, manager):
        """Test DatabaseManager as context manager."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            async with manager as mgr:
                assert mgr is manager
                assert mgr.connection is not None

    @pytest.mark.asyncio
    async def test_manager_health_check(self, manager):
        """Test DatabaseManager health check."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            await manager.connect()

            # Test healthy connection
            with patch.object(manager.connection, "is_connected", return_value=True):
                health = await manager.health_check()
                assert health["status"] == "healthy"
                assert health["connected"] is True

            # Test unhealthy connection
            with patch.object(manager.connection, "is_connected", return_value=False):
                health = await manager.health_check()
                assert health["status"] == "unhealthy"
                assert health["connected"] is False


class TestCommitRepository:
    """Test cases for CommitRepository."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def repository(self, mock_session):
        """Create a CommitRepository instance."""
        return CommitRepository(mock_session)

    @pytest.fixture
    def sample_commit(self):
        """Create a sample commit for testing."""
        return Commit(
            id=str(uuid4()),
            hash="abc123def456",
            author="testuser",
            message="feat: add new feature",
            timestamp=datetime.now(timezone.utc),
            changed_files=["src/main.py"],
            repository="test-repo",
            branch="main",
            additions=50,
            deletions=10,
            total_changes=60,
            quality=CommitQuality.GOOD,
            analysis_status=AnalysisStatus.PENDING,
        )

    @pytest.mark.asyncio
    async def test_create_commit(self, repository, sample_commit):
        """Test creating a commit."""
        mock_session = repository.session
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_commit)

        assert result == sample_commit
        mock_session.add.assert_called_once_with(sample_commit)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_commit)

    @pytest.mark.asyncio
    async def test_get_commit_by_id(self, repository, sample_commit):
        """Test getting a commit by ID."""
        mock_session = repository.session
        mock_session.get = AsyncMock(return_value=sample_commit)

        result = await repository.get_by_id(sample_commit.id)

        assert result == sample_commit
        mock_session.get.assert_called_once_with(Commit, sample_commit.id)

    @pytest.mark.asyncio
    async def test_get_commit_by_hash(self, repository, sample_commit):
        """Test getting a commit by hash."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.first = Mock(return_value=sample_commit)

        result = await repository.get_by_hash(sample_commit.hash)

        assert result == sample_commit
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_commits_by_author(self, repository, sample_commit):
        """Test getting commits by author."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_commit])

        result = await repository.get_by_author("testuser")

        assert result == [sample_commit]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_commits_by_repository(self, repository, sample_commit):
        """Test getting commits by repository."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_commit])

        result = await repository.get_by_repository("test-repo")

        assert result == [sample_commit]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_commit(self, repository, sample_commit):
        """Test updating a commit."""
        mock_session = repository.session
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        sample_commit.message = "updated message"
        result = await repository.update(sample_commit)

        assert result == sample_commit
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_commit)

    @pytest.mark.asyncio
    async def test_delete_commit(self, repository, sample_commit):
        """Test deleting a commit."""
        mock_session = repository.session
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_commit.id)

        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_commits_with_pagination(self, repository, sample_commit):
        """Test getting commits with pagination."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.offset = Mock(return_value=mock_query)
        mock_query.limit = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_commit])
        mock_query.count = Mock(return_value=1)

        result = await repository.get_with_pagination(page=1, size=10)

        assert result["items"] == [sample_commit]
        assert result["total"] == 1
        assert result["page"] == 1
        assert result["size"] == 10

    @pytest.mark.asyncio
    async def test_get_commits_by_date_range(self, repository, sample_commit):
        """Test getting commits by date range."""
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_commit])

        result = await repository.get_by_date_range(start_date, end_date)

        assert result == [sample_commit]
        mock_session.exec.assert_called_once()


class TestAnalysisRepository:
    """Test cases for AnalysisRepository."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def repository(self, mock_session):
        """Create an AnalysisRepository instance."""
        return AnalysisRepository(mock_session)

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        return Analysis(
            id=str(uuid4()),
            commit_id=str(uuid4()),
            analysis_type="commit_quality",
            status=AnalysisStatus.COMPLETED,
            result={"score": 8.5},
            metadata={"model": "llama2"},
        )

    @pytest.mark.asyncio
    async def test_create_analysis(self, repository, sample_analysis):
        """Test creating an analysis."""
        mock_session = repository.session
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_analysis)

        assert result == sample_analysis
        mock_session.add.assert_called_once_with(sample_analysis)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_analysis)

    @pytest.mark.asyncio
    async def test_get_analysis_by_id(self, repository, sample_analysis):
        """Test getting an analysis by ID."""
        mock_session = repository.session
        mock_session.get = AsyncMock(return_value=sample_analysis)

        result = await repository.get_by_id(sample_analysis.id)

        assert result == sample_analysis
        mock_session.get.assert_called_once_with(Analysis, sample_analysis.id)

    @pytest.mark.asyncio
    async def test_get_analysis_by_commit_id(self, repository, sample_analysis):
        """Test getting analysis by commit ID."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_analysis])

        result = await repository.get_by_commit_id(sample_analysis.commit_id)

        assert result == [sample_analysis]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_analysis_by_type(self, repository, sample_analysis):
        """Test getting analysis by type."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_analysis])

        result = await repository.get_by_type("commit_quality")

        assert result == [sample_analysis]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_analysis(self, repository, sample_analysis):
        """Test updating an analysis."""
        mock_session = repository.session
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        sample_analysis.status = AnalysisStatus.FAILED
        result = await repository.update(sample_analysis)

        assert result == sample_analysis
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_analysis)

    @pytest.mark.asyncio
    async def test_delete_analysis(self, repository, sample_analysis):
        """Test deleting an analysis."""
        mock_session = repository.session
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_analysis.id)

        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()


class TestBehaviorPatternRepository:
    """Test cases for BehaviorPatternRepository."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def repository(self, mock_session):
        """Create a BehaviorPatternRepository instance."""
        return BehaviorPatternRepository(mock_session)

    @pytest.fixture
    def sample_pattern(self):
        """Create a sample behavior pattern for testing."""
        return BehaviorPattern(
            id=str(uuid4()),
            user_id="testuser",
            pattern_type=BehaviorPatternType.FREQUENT_COMMITS,
            description="User commits frequently",
            confidence=0.85,
            metadata={"frequency": "daily"},
        )

    @pytest.mark.asyncio
    async def test_create_pattern(self, repository, sample_pattern):
        """Test creating a behavior pattern."""
        mock_session = repository.session
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_pattern)

        assert result == sample_pattern
        mock_session.add.assert_called_once_with(sample_pattern)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_pattern)

    @pytest.mark.asyncio
    async def test_get_pattern_by_id(self, repository, sample_pattern):
        """Test getting a pattern by ID."""
        mock_session = repository.session
        mock_session.get = AsyncMock(return_value=sample_pattern)

        result = await repository.get_by_id(sample_pattern.id)

        assert result == sample_pattern
        mock_session.get.assert_called_once_with(BehaviorPattern, sample_pattern.id)

    @pytest.mark.asyncio
    async def test_get_patterns_by_user_id(self, repository, sample_pattern):
        """Test getting patterns by user ID."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_pattern])

        result = await repository.get_by_user_id("testuser")

        assert result == [sample_pattern]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_patterns_by_type(self, repository, sample_pattern):
        """Test getting patterns by type."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_pattern])

        result = await repository.get_by_type(BehaviorPatternType.FREQUENT_COMMITS)

        assert result == [sample_pattern]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_pattern(self, repository, sample_pattern):
        """Test updating a pattern."""
        mock_session = repository.session
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        sample_pattern.confidence = 0.95
        result = await repository.update(sample_pattern)

        assert result == sample_pattern
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_pattern)

    @pytest.mark.asyncio
    async def test_delete_pattern(self, repository, sample_pattern):
        """Test deleting a pattern."""
        mock_session = repository.session
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_pattern.id)

        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()


class TestUserBehaviorRepository:
    """Test cases for UserBehaviorRepository."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def repository(self, mock_session):
        """Create a UserBehaviorRepository instance."""
        return UserBehaviorRepository(mock_session)

    @pytest.fixture
    def sample_behavior(self):
        """Create a sample user behavior for testing."""
        return UserBehavior(
            id=str(uuid4()),
            user_id="testuser",
            behavior_type="commit_frequency",
            value=5.5,
            unit="commits_per_day",
            period="daily",
            metadata={"trend": "increasing"},
        )

    @pytest.mark.asyncio
    async def test_create_behavior(self, repository, sample_behavior):
        """Test creating a user behavior."""
        mock_session = repository.session
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_behavior)

        assert result == sample_behavior
        mock_session.add.assert_called_once_with(sample_behavior)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_behavior)

    @pytest.mark.asyncio
    async def test_get_behavior_by_id(self, repository, sample_behavior):
        """Test getting a behavior by ID."""
        mock_session = repository.session
        mock_session.get = AsyncMock(return_value=sample_behavior)

        result = await repository.get_by_id(sample_behavior.id)

        assert result == sample_behavior
        mock_session.get.assert_called_once_with(UserBehavior, sample_behavior.id)

    @pytest.mark.asyncio
    async def test_get_behaviors_by_user_id(self, repository, sample_behavior):
        """Test getting behaviors by user ID."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_behavior])

        result = await repository.get_by_user_id("testuser")

        assert result == [sample_behavior]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_behaviors_by_type(self, repository, sample_behavior):
        """Test getting behaviors by type."""
        mock_session = repository.session
        mock_query = Mock()
        mock_session.exec = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[sample_behavior])

        result = await repository.get_by_type("commit_frequency")

        assert result == [sample_behavior]
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_behavior(self, repository, sample_behavior):
        """Test updating a behavior."""
        mock_session = repository.session
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        sample_behavior.value = 6.0
        result = await repository.update(sample_behavior)

        assert result == sample_behavior
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_behavior)

    @pytest.mark.asyncio
    async def test_delete_behavior(self, repository, sample_behavior):
        """Test deleting a behavior."""
        mock_session = repository.session
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_behavior.id)

        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()


class TestDatabaseErrors:
    """Test cases for database error handling."""

    def test_database_error_creation(self):
        """Test DatabaseError creation."""
        error = DatabaseError("Database operation failed")
        assert str(error) == "Database operation failed"
        assert isinstance(error, Exception)

    def test_connection_error_creation(self):
        """Test ConnectionError creation."""
        error = ConnectionError("Failed to connect to database")
        assert str(error) == "Failed to connect to database"
        assert isinstance(error, DatabaseError)

    def test_query_error_creation(self):
        """Test QueryError creation."""
        error = QueryError("Invalid query")
        assert str(error) == "Invalid query"
        assert isinstance(error, DatabaseError)

    def test_transaction_error_creation(self):
        """Test TransactionError creation."""
        error = TransactionError("Transaction failed")
        assert str(error) == "Transaction failed"
        assert isinstance(error, DatabaseError)


class TestDatabaseUtilities:
    """Test cases for database utility functions."""

    @pytest.mark.asyncio
    async def test_get_database_manager(self):
        """Test getting database manager."""
        with patch("shared.database.DatabaseManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            manager = await get_database_manager()

            assert manager == mock_manager
            mock_manager_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_database_engine(self):
        """Test creating database engine."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            engine = await create_database_engine("postgresql://test")

            assert engine == mock_engine
            mock_create_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_database(self):
        """Test database initialization."""
        with patch("shared.database.create_async_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            with patch("shared.database.Base.metadata.create_all") as mock_create_all:
                await init_database("postgresql://test")

                mock_create_engine.assert_called_once()
                mock_create_all.assert_called_once_with(mock_engine)

    @pytest.mark.asyncio
    async def test_close_database_connections(self):
        """Test closing database connections."""
        with patch("shared.database.get_database_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            await close_database_connections()

            mock_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_async_db(self):
        """Test getting async database session."""
        with patch("shared.database.get_database_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_session = AsyncMock()
            mock_manager.get_session.return_value.__aenter__.return_value = mock_session

            async for session in get_async_db():
                assert session == mock_session
                break


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_full_commit_workflow(self):
        """Test complete commit workflow with database operations."""
        # Mock database session
        mock_session = AsyncMock()

        # Create repositories
        commit_repo = CommitRepository(mock_session)
        analysis_repo = AnalysisRepository(mock_session)

        # Create sample data
        commit = Commit(
            id=str(uuid4()),
            hash="abc123def456",
            author="testuser",
            message="feat: add new feature",
            timestamp=datetime.now(timezone.utc),
            changed_files=["src/main.py"],
            repository="test-repo",
            branch="main",
            additions=50,
            deletions=10,
            total_changes=60,
            quality=CommitQuality.GOOD,
            analysis_status=AnalysisStatus.PENDING,
        )

        analysis = Analysis(
            id=str(uuid4()),
            commit_id=commit.id,
            analysis_type="commit_quality",
            status=AnalysisStatus.COMPLETED,
            result={"score": 8.5},
            metadata={"model": "llama2"},
        )

        # Mock session operations
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_session.get = AsyncMock(side_effect=[commit, analysis])

        # Test commit creation
        created_commit = await commit_repo.create(commit)
        assert created_commit == commit

        # Test analysis creation
        created_analysis = await analysis_repo.create(analysis)
        assert created_analysis == analysis

        # Test retrieving commit
        retrieved_commit = await commit_repo.get_by_id(commit.id)
        assert retrieved_commit == commit

        # Test retrieving analysis
        retrieved_analysis = await analysis_repo.get_by_id(analysis.id)
        assert retrieved_analysis == analysis

        # Verify session operations were called
        assert mock_session.add.call_count == 2
        assert mock_session.commit.call_count == 2
        assert mock_session.refresh.call_count == 2
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self):
        """Test transaction rollback on error."""
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        mock_session.rollback = AsyncMock()

        commit_repo = CommitRepository(mock_session)
        commit = Commit(
            id=str(uuid4()),
            hash="abc123def456",
            author="testuser",
            message="test",
            timestamp=datetime.now(timezone.utc),
            changed_files=[],
            repository="test",
            branch="main",
            additions=0,
            deletions=0,
            total_changes=0,
        )

        # Test that rollback is called on error
        with pytest.raises(Exception):
            await commit_repo.create(commit)

        mock_session.rollback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
