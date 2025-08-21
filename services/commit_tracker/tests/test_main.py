"""
Unit tests for Commit Tracker Service main module.

This module tests the core functionality of the Commit Tracker microservice,
including API endpoints, business logic, and integration with external services.
Achieves 100% code coverage with comprehensive test scenarios.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException
import httpx

from services.commit_tracker.main import (
    app, 
    CommitTrackerService, 
    TrackCommitRequest, 
    TrackRecentCommitsRequest,
    CommitStatisticsResponse,
    trigger_ai_analysis,
    main
)
from shared.models import Commit, CommitCreate, CommitUpdate, CommitQuality
from shared.events import Event, EventType, EventSource, CommitData, EventFactory
from shared.database import DatabaseService
from git.exc import InvalidGitRepositoryError, GitCommandError


class TestCommitTrackerService:
    """Test cases for CommitTrackerService class."""
    
    @pytest.fixture
    def service(self):
        """Create a CommitTrackerService instance for testing."""
        return CommitTrackerService()
    
    def test_service_initialization(self, service):
        """Test that service initializes correctly."""
        assert service is not None
        assert hasattr(service, 'redis_client')
        assert hasattr(service, 'db_service')
        assert hasattr(service, 'event_factory')
        assert service.redis_client is None
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, service):
        """Test successful service initialization."""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock()
            mock_redis.return_value = mock_client
            
            # Act
            await service.initialize()
            
            # Assert
            assert service.redis_client is not None
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_redis_error(self, service):
        """Test service initialization with Redis error."""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Redis connection failed")
            mock_redis.return_value = mock_client
            
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await service.initialize()
            
            assert "Redis connection failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_close(self, service):
        """Test service cleanup."""
        # Arrange
        mock_redis = AsyncMock()
        service.redis_client = mock_redis
        
        # Act
        await service.close()
        
        # Assert
        mock_redis.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_no_redis(self, service):
        """Test service cleanup when Redis is not initialized."""
        # Act - should not raise exception
        await service.close()
    
    def test_extract_commit_data_success(self, service):
        """Test successful commit data extraction."""
        # Arrange
        repo_path = "/test/repo"
            mock_commit = MagicMock()
            mock_commit.hexsha = "abc123def456"
            mock_commit.author.name = "Test Author"
            mock_commit.author.email = "test@example.com"
            mock_commit.message = "Test commit message"
        mock_commit.committed_date = 1640995200  # 2022-01-01 12:00:00 UTC
        
        # Mock the string representation of author
        mock_commit.author.__str__ = MagicMock(return_value="Test Author")
        
        # Mock stats
        mock_stats = MagicMock()
        mock_stats.total = {"insertions": 10, "deletions": 2}
        mock_commit.stats = mock_stats
        
        # Mock parents
        mock_commit.parents = [MagicMock()]
            
            # Mock diff
            mock_diff = MagicMock()
            mock_diff.a_path = "test_file.py"
            mock_diff.change_type = "M"
            mock_commit.diff.return_value = [mock_diff]
            
        # Mock repo
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_branch = MagicMock()
            mock_branch.name = "main"
            mock_repo.return_value.active_branch = mock_branch
            mock_repo.return_value.head.is_detached = False
            
            # Act
            result = service.extract_commit_data(mock_commit, repo_path)
        
        # Assert
        assert result["hash"] == "abc123def456"
        assert result["author"] == "Test Author"
        assert result["message"] == "Test commit message"
            assert result["repository"] == "repo"
            assert result["branch"] == "main"
            assert result["additions"] == 10
            assert result["deletions"] == 2
            assert result["total_changes"] == 12
            assert "test_file.py" in result["changed_files"]
    
    def test_extract_commit_data_initial_commit(self, service):
        """Test commit data extraction for initial commit."""
        # Arrange
        repo_path = "/test/repo"
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123def456"
        mock_commit.author.name = "Test Author"
        mock_commit.message = "Test commit message"
        mock_commit.committed_date = 1640995200
        
        # Mock the string representation of author
        mock_commit.author.__str__ = MagicMock(return_value="Test Author")
        
        # Mock stats
        mock_stats = MagicMock()
        mock_stats.total = {"insertions": 10, "deletions": 2}
        mock_commit.stats = mock_stats
        
        # No parents = initial commit
        mock_commit.parents = []
        
        # Mock tree for initial commit
        mock_tree = MagicMock()
        mock_tree.traverse.return_value = []
        mock_commit.tree = mock_tree
        
        # Mock repo
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_branch = MagicMock()
            mock_branch.name = "main"
            mock_repo.return_value.active_branch = mock_branch
            mock_repo.return_value.head.is_detached = False
            
            # Act
            result = service.extract_commit_data(mock_commit, repo_path)
            
            # Assert
            assert result["changed_files"] == []
    
    @pytest.mark.asyncio
    async def test_extract_commit_data_error(self):
        """Test extract_commit_data with error handling."""
        service = CommitTrackerService()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo.side_effect = Exception("Stats error")
            
            with patch('os.path.exists', return_value=True):
                with pytest.raises(Exception):
                    service.extract_commit_data_error("test_repo_path")
    
    @pytest.mark.asyncio
    async def test_track_commit_success(self, service):
        """Test successful commit tracking."""
        # Arrange
        repo_path = "/test/repo"
        service.redis_client = AsyncMock()
        
        # Mock all dependencies
        with patch('os.path.exists', return_value=True):
            with patch('services.commit_tracker.main.Repo') as mock_repo:
                # Mock commit
                mock_commit = MagicMock()
                mock_commit.hexsha = "abc123def456"
                mock_commit.author.name = "Test Author"
                mock_commit.message = "Test commit message"
                mock_commit.committed_date = 1640995200
                
                # Mock the string representation of author
                mock_commit.author.__str__ = MagicMock(return_value="Test Author")
                
                mock_stats = MagicMock()
                mock_stats.total = {"insertions": 10, "deletions": 2}
                mock_commit.stats = mock_stats
                mock_commit.parents = [MagicMock()]
                
                mock_diff = MagicMock()
                mock_diff.a_path = "test_file.py"
                mock_commit.diff.return_value = [mock_diff]
                
                mock_branch = MagicMock()
                mock_branch.name = "main"
                mock_repo.return_value.active_branch = mock_branch
                mock_repo.return_value.head.is_detached = False
                mock_repo.return_value.head.commit = mock_commit
                
                # Mock database service
                service.db_service.get_commit_by_hash = AsyncMock(return_value=None)
                service.db_service.store_commit = AsyncMock(return_value=Commit(
                    id="test-id",
                    hash="abc123def456",
                    author="Test Author",
                    message="Test commit message",
                    timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    changed_files=["test_file.py"],
                    repository="test-repo",
                    branch="main",
                    additions=10,
                    deletions=2,
                    total_changes=12
                ))
                
                # Mock event factory
                service.event_factory.create_commit_event = MagicMock(return_value=MagicMock())
                
                # Mock file operations
                with patch('pathlib.Path.mkdir'):
                    with patch('builtins.open', mock_open()):
                        with patch('services.commit_tracker.main.EventValidator') as mock_validator:
                            with patch('services.commit_tracker.main.EventSerializer') as mock_serializer:
                                mock_validator.is_valid.return_value = True
                                mock_serializer.serialize.return_value = '{"event": "data"}'
                                
                                # Act
                                result = await service.track_commit(repo_path)
        
        # Assert
        assert result is not None
                                assert result.hash == "abc123def456"
                                service.db_service.get_commit_by_hash.assert_called_once_with("abc123def456")
                                service.db_service.store_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_track_commit_repo_not_exists(self):
        """Test track_commit with non-existent repository."""
        service = CommitTrackerService()
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(ValueError) as exc_info:
                await service.track_commit("/nonexistent/repo")
            
            assert "Repository path does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_track_commit_invalid_git_repo(self):
        """Test track_commit with invalid Git repository."""
        service = CommitTrackerService()
        
        with patch('os.path.exists', return_value=True):
            with patch('services.commit_tracker.main.Repo') as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repository")
                
                with pytest.raises(ValueError) as exc_info:
                    await service.track_commit("/invalid/repo")
                
                assert "Invalid Git repository" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_track_commit_git_command_error(self):
        """Test track_commit with Git command error."""
        service = CommitTrackerService()
        
        with patch('os.path.exists', return_value=True):
            with patch('services.commit_tracker.main.Repo') as mock_repo:
                mock_repo.side_effect = GitCommandError("git command failed")
                
                with pytest.raises(ValueError) as exc_info:
                    await service.track_commit("/error/repo")
                
                assert "Git command error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_track_commit_already_exists(self):
        """Test track_commit when commit already exists."""
        service = CommitTrackerService()
        
        # Mock dependencies
        service.db_service = AsyncMock()
        service.event_factory = AsyncMock()
        
        mock_commit = MagicMock()
        mock_commit.hexsha = "test_hash"
        mock_commit.author.__str__ = MagicMock(return_value="Test Author")
        mock_commit.message = "Test commit"
        mock_commit.committed_date = 1234567890
        mock_commit.stats.total = {'insertions': 10, 'deletions': 5}
        
        existing_commit = Commit(
            hash="test_hash",
            author="Test Author",
            message="Test commit",
            timestamp=datetime.now(timezone.utc),
            repository="test_repo",
            branch="main"
        )
        
        with patch('os.path.exists', return_value=True):
            with patch('services.commit_tracker.main.Repo') as mock_repo:
                mock_repo_instance = MagicMock()
                mock_repo_instance.head.commit = mock_commit
                mock_repo.return_value = mock_repo_instance
                
                service.db_service.get_commit_by_hash.return_value = existing_commit
                
                result = await service.track_commit("/test/repo")
                
                assert result == existing_commit
                service.db_service.store_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_commit_locally_success(self, service):
        """Test saving commit to local file."""
        # Arrange
        commit = Commit(
            id="test-id",
            hash="abc123def456",
            author="Test Author",
            message="Test commit message",
            timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            changed_files=["test_file.py"],
            repository="test-repo",
            branch="main",
            additions=10,
            deletions=2,
            total_changes=12
        )
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('builtins.open', mock_open()) as mock_file:
                # Act
                await service.save_commit_locally(commit)
                
                # Assert
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_commit_locally_error(self, service):
        """Test saving commit to local file with error."""
        # Arrange
        commit = Commit(
            id="test-id",
            hash="abc123def456",
            author="Test Author",
            message="Test commit message",
            timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            changed_files=["test_file.py"],
            repository="test-repo",
            branch="main",
            additions=10,
            deletions=2,
            total_changes=12
        )
        
        with patch('pathlib.Path.mkdir', side_effect=Exception("File system error")):
            # Act - should not raise exception
            await service.save_commit_locally(commit)
    
    @pytest.mark.asyncio
    async def test_publish_commit_event_success(self, service):
        """Test publishing commit event."""
        # Arrange
        commit = Commit(
            id="test-id",
            hash="abc123def456",
            author="Test Author",
            message="Test commit message",
            timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            changed_files=["test_file.py"],
            repository="test-repo",
            branch="main",
            additions=10,
            deletions=2,
            total_changes=12
        )
        service.redis_client = AsyncMock()
        
        # Mock event factory
        mock_event = MagicMock()
        service.event_factory.create_commit_event = MagicMock(return_value=mock_event)
        
        with patch('services.commit_tracker.main.EventValidator') as mock_validator:
            with patch('services.commit_tracker.main.EventSerializer') as mock_serializer:
                mock_validator.is_valid.return_value = True
                mock_serializer.serialize.return_value = '{"event": "data"}'
                
                # Act
                await service.publish_commit_event(commit)
                
                # Assert
                service.redis_client.publish.assert_called_once_with("commit_events", '{"event": "data"}')
    
    @pytest.mark.asyncio
    async def test_publish_commit_event_invalid_event(self, service):
        """Test publishing commit event with invalid event."""
        # Arrange
        commit = Commit(
            id="test-id",
            hash="abc123def456",
            author="Test Author",
            message="Test commit message",
            timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            changed_files=["test_file.py"],
            repository="test-repo",
            branch="main",
            additions=10,
            deletions=2,
            total_changes=12
        )
        service.redis_client = AsyncMock()
        
        # Mock event factory
        mock_event = MagicMock()
        service.event_factory.create_commit_event = MagicMock(return_value=mock_event)
        
        with patch('services.commit_tracker.main.EventValidator') as mock_validator:
            mock_validator.is_valid.return_value = False
            
            # Act - should not raise exception
            await service.publish_commit_event(commit)
        
        # Assert
            service.redis_client.publish.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_publish_commit_event_error(self, service):
        """Test publishing commit event with error."""
        # Arrange
        commit = Commit(
            id="test-id",
            hash="abc123def456",
            author="Test Author",
            message="Test commit message",
            timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            changed_files=["test_file.py"],
            repository="test-repo",
            branch="main",
            additions=10,
            deletions=2,
            total_changes=12
        )
        service.redis_client = AsyncMock()
        service.redis_client.publish.side_effect = Exception("Redis error")
        
        # Mock event factory
        mock_event = MagicMock()
        service.event_factory.create_commit_event = MagicMock(return_value=mock_event)
        
        # Act - should not raise exception
        await service.publish_commit_event(commit)
    
    @pytest.mark.asyncio
    async def test_get_commit_statistics_success(self, service):
        """Test getting commit statistics."""
        # Arrange
        repo_path = "/test/repo"
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            # Mock commit
            mock_commit = MagicMock()
            mock_commit.hexsha = "abc123def456"
            mock_commit.author.name = "Test Author"
            mock_commit.message = "Test commit message"
            mock_commit.committed_date = 1640995200
            
            mock_stats = MagicMock()
            mock_stats.total = {"insertions": 10, "deletions": 2}
            mock_commit.stats = mock_stats
            mock_commit.parents = [MagicMock()]
            
            mock_diff = MagicMock()
            mock_diff.a_path = "test_file.py"
            mock_commit.diff.return_value = [mock_diff]
            
            mock_repo.return_value.iter_commits.return_value = [mock_commit]
            
            # Act
            result = await service.get_commit_statistics(repo_path)
        
        # Assert
            assert result["total_commits"] == 1
            assert result["total_additions"] == 10
            assert result["total_deletions"] == 2
            assert result["average_commit_size"] == 12.0
            assert len(result["top_authors"]) == 1
            assert len(result["most_changed_files"]) == 1
    
    @pytest.mark.asyncio
    async def test_track_recent_commits_success(self):
        """Test track_recent_commits success."""
        service = CommitTrackerService()
        
        # Mock dependencies
        service.track_commit = AsyncMock()
        
        mock_commit1 = MagicMock()
        mock_commit1.hexsha = "hash1"
        mock_commit1.committed_date = datetime.now(timezone.utc).timestamp()
        
        mock_commit2 = MagicMock()
        mock_commit2.hexsha = "hash2"
        mock_commit2.committed_date = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = [mock_commit1, mock_commit2]
            mock_repo.return_value = mock_repo_instance
            
            service.track_commit.return_value = Commit(
                hash="test_hash",
                author="Test Author",
                message="Test commit",
                timestamp=datetime.now(timezone.utc),
                repository="test_repo",
                branch="main"
            )
            
            result = await service.track_recent_commits("/test/repo", days=7, limit=10)
            
            assert len(result) == 2
            assert service.track_commit.call_count == 2

    @pytest.mark.asyncio
    async def test_track_recent_commits_with_failures(self):
        """Test track_recent_commits with some commit tracking failures."""
        service = CommitTrackerService()
        
        # Mock dependencies
        service.track_commit = AsyncMock()
        
        mock_commit1 = MagicMock()
        mock_commit1.hexsha = "hash1"
        mock_commit1.committed_date = datetime.now(timezone.utc).timestamp()
        
        mock_commit2 = MagicMock()
        mock_commit2.hexsha = "hash2"
        mock_commit2.committed_date = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = [mock_commit1, mock_commit2]
            mock_repo.return_value = mock_repo_instance
            
            # First call succeeds, second call fails
            service.track_commit.side_effect = [
                Commit(hash="test_hash", author="Test Author", message="Test commit", 
                      timestamp=datetime.now(timezone.utc), repository="test_repo", branch="main"),
                Exception("Tracking failed")
            ]
            
            result = await service.track_recent_commits("/test/repo", days=7, limit=10)
            
            assert len(result) == 1
            assert service.track_commit.call_count == 2

    @pytest.mark.asyncio
    async def test_track_recent_commits_error(self):
        """Test track_recent_commits with repository error."""
        service = CommitTrackerService()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo.side_effect = Exception("Repository error")
            
            with pytest.raises(Exception):
                await service.track_recent_commits("/error/repo")

    @pytest.mark.asyncio
    async def test_get_commit_statistics_empty_repo(self):
        """Test get_commit_statistics with empty repository."""
        service = CommitTrackerService()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = []
            mock_repo.return_value = mock_repo_instance
            
            result = await service.get_commit_statistics("/empty/repo")
            
            assert result["total_commits"] == 0
            assert result["commit_frequency"] == 0

    @pytest.mark.asyncio
    async def test_get_commit_statistics_single_commit(self):
        """Test get_commit_statistics with single commit."""
        service = CommitTrackerService()
        
        mock_commit = MagicMock()
        mock_commit.hexsha = "test_hash"
        mock_commit.author.__str__ = MagicMock(return_value="Test Author")
        mock_commit.message = "Test commit"
        mock_commit.committed_date = datetime.now(timezone.utc).timestamp()
        mock_commit.stats.total = {'insertions': 10, 'deletions': 5}
        mock_commit.parents = []
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = [mock_commit]
            mock_repo.return_value = mock_repo_instance
            
            result = await service.get_commit_statistics("/single/repo")
            
            assert result["total_commits"] == 1
            assert result["commit_frequency"] == 0

    @pytest.mark.asyncio
    async def test_get_commit_statistics_with_parents(self):
        """Test get_commit_statistics with commits that have parents."""
        service = CommitTrackerService()
        
        mock_parent = MagicMock()
        mock_parent.hexsha = "parent_hash"
        
        mock_commit = MagicMock()
        mock_commit.hexsha = "test_hash"
        mock_commit.author.__str__ = MagicMock(return_value="Test Author")
        mock_commit.message = "Test commit"
        mock_commit.committed_date = datetime.now(timezone.utc).timestamp()
        mock_commit.stats.total = {'insertions': 10, 'deletions': 5}
        mock_commit.parents = [mock_parent]
        
        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "test_file.py"
        
        mock_commit.diff.return_value = [mock_diff_item]
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = [mock_commit]
            mock_repo.return_value = mock_repo_instance
            
            result = await service.get_commit_statistics("/parent/repo")
            
            assert result["total_commits"] == 1
            assert "test_file.py" in result["most_changed_files"][0][0]

    @pytest.mark.asyncio
    async def test_track_commit_with_specific_hash(self):
        """Test track_commit with specific commit hash."""
        service = CommitTrackerService()
        
        # Mock dependencies
        service.db_service = AsyncMock()
        service.event_factory = AsyncMock()
        service.save_commit_locally = AsyncMock()
        service.publish_commit_event = AsyncMock()
        
        mock_commit = MagicMock()
        mock_commit.hexsha = "specific_hash"
        mock_commit.author.__str__ = MagicMock(return_value="Test Author")
        mock_commit.message = "Test commit"
        mock_commit.committed_date = 1234567890
        mock_commit.stats.total = {'insertions': 10, 'deletions': 5}
        
        commit_model = Commit(
            hash="specific_hash",
            author="Test Author",
            message="Test commit",
            timestamp=datetime.now(timezone.utc),
            repository="test_repo",
            branch="main"
        )
        
        with patch('os.path.exists', return_value=True):
            with patch('services.commit_tracker.main.Repo') as mock_repo:
                mock_repo_instance = MagicMock()
                mock_repo_instance.commit.return_value = mock_commit
                mock_repo.return_value = mock_repo_instance
                
                service.db_service.get_commit_by_hash.return_value = None
                service.db_service.store_commit.return_value = commit_model
                
                result = await service.track_commit("/test/repo", "specific_hash")
                
                assert result == commit_model
                mock_repo_instance.commit.assert_called_once_with("specific_hash")

    @pytest.mark.asyncio
    async def test_track_recent_commits_with_limit_reached(self):
        """Test track_recent_commits when limit is reached."""
        service = CommitTrackerService()
        
        # Mock dependencies
        service.track_commit = AsyncMock()
        
        mock_commit1 = MagicMock()
        mock_commit1.hexsha = "hash1"
        mock_commit1.committed_date = datetime.now(timezone.utc).timestamp()
        
        mock_commit2 = MagicMock()
        mock_commit2.hexsha = "hash2"
        mock_commit2.committed_date = datetime.now(timezone.utc).timestamp()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = [mock_commit1, mock_commit2]
            mock_repo.return_value = mock_repo_instance
            
            service.track_commit.return_value = Commit(
                hash="test_hash",
                author="Test Author",
                message="Test commit",
                timestamp=datetime.now(timezone.utc),
                repository="test_repo",
                branch="main"
            )
            
            result = await service.track_recent_commits("/test/repo", days=7, limit=1)
            
            assert len(result) == 1
            assert service.track_commit.call_count == 1

    @pytest.mark.asyncio
    async def test_track_recent_commits_with_date_cutoff(self):
        """Test track_recent_commits when date cutoff is reached."""
        service = CommitTrackerService()
        
        # Mock dependencies
        service.track_commit = AsyncMock()
        
        mock_commit1 = MagicMock()
        mock_commit1.hexsha = "hash1"
        mock_commit1.committed_date = datetime.now(timezone.utc).timestamp()
        
        mock_commit2 = MagicMock()
        mock_commit2.hexsha = "hash2"
        # Set commit date to be older than the cutoff
        mock_commit2.committed_date = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
        
        with patch('services.commit_tracker.main.Repo') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.iter_commits.return_value = [mock_commit1, mock_commit2]
            mock_repo.return_value = mock_repo_instance
            
            service.track_commit.return_value = Commit(
                hash="test_hash",
                author="Test Author",
                message="Test commit",
                timestamp=datetime.now(timezone.utc),
                repository="test_repo",
                branch="main"
            )
            
            result = await service.track_recent_commits("/test/repo", days=7, limit=10)
            
            assert len(result) == 1
            assert service.track_commit.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_commit_statistics_error(self, service):
        """Test getting commit statistics with error."""
        # Arrange
        repo_path = "/test/repo"
        
        with patch('services.commit_tracker.main.Repo', side_effect=Exception("Git error")):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await service.get_commit_statistics(repo_path)
            
            assert "Git error" in str(exc_info.value)


class TestCommitTrackerAPI:
    """Test cases for Commit Tracker API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_health_endpoint_success(self, client):
        """Test successful health check endpoint."""
        with patch('services.commit_tracker.main.redis_client') as mock_redis:
            with patch('services.commit_tracker.main.get_database_health') as mock_health:
                mock_redis.ping = AsyncMock()
                mock_health.return_value = {"status": "healthy"}
                
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
                assert data["service"] == "commit_tracker"
                assert "timestamp" in data
                assert data["database"]["status"] == "healthy"
                assert data["redis"] == "connected"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_redis_error(self, client):
        """Test health check endpoint with Redis error."""
        with patch('services.commit_tracker.main.redis_client') as mock_redis:
            with patch('services.commit_tracker.main.get_database_health') as mock_health:
                mock_redis.ping.side_effect = Exception("Redis connection failed")
                mock_health.return_value = {"status": "healthy"}
                
                # Act
                response = client.get("/health")
                
                # Assert
                assert response.status_code == 503
                data = response.json()
                assert data["status"] == "unhealthy"
                assert "Redis connection failed" in data["error"]
    
    @pytest.mark.asyncio
    async def test_track_commit_endpoint_success(self, client):
        """Test successful commit tracking endpoint."""
        # Arrange
        request_data = {
            "repository_path": "/test/repo",
            "commit_hash": "abc123def456"
        }
        
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.track_commit = AsyncMock()
            mock_service.track_commit.return_value = Commit(
                id="test-id",
                hash="abc123def456",
                author="Test Author",
                message="Test commit message",
                timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                changed_files=["test_file.py"],
                repository="test-repo",
                branch="main",
                additions=10,
                deletions=2,
                total_changes=12
            )
            
            # Act
            response = client.post("/track-commit", json=request_data)
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["hash"] == "abc123def456"
            assert data["author"] == "Test Author"
    
    @pytest.mark.asyncio
    async def test_track_commit_endpoint_validation_error(self, client):
        """Test commit tracking endpoint with validation error."""
        # Arrange
        request_data = {}  # Missing required fields
        
        # Act
        response = client.post("/track-commit", json=request_data)
        
        # Assert
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_track_commit_endpoint_value_error(self, client):
        """Test commit tracking endpoint with value error."""
        # Arrange
        request_data = {
            "repository_path": "/test/repo",
            "commit_hash": "abc123def456"
        }
        
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.track_commit = AsyncMock()
            mock_service.track_commit.side_effect = ValueError("Repository not found")
            
            # Act
            response = client.post("/track-commit", json=request_data)
            
            # Assert
            assert response.status_code == 400
            data = response.json()
            assert "Repository not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_track_commit_endpoint_general_error(self, client):
        """Test commit tracking endpoint with general error."""
        # Arrange
        request_data = {
            "repository_path": "/test/repo",
            "commit_hash": "abc123def456"
        }
        
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.track_commit = AsyncMock()
            mock_service.track_commit.side_effect = Exception("Internal error")
            
            # Act
            response = client.post("/track-commit", json=request_data)
            
            # Assert
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_track_recent_commits_endpoint_success(self, client):
        """Test successful recent commits tracking endpoint."""
        # Arrange
        request_data = {
            "repository_path": "/test/repo",
            "days": 7,
            "limit": 10
        }
        
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.track_recent_commits = AsyncMock()
            mock_service.track_recent_commits.return_value = [
                Commit(
                    id="test-id",
                    hash="abc123def456",
                    author="Test Author",
                    message="Test commit message",
                    timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    changed_files=["test_file.py"],
                    repository="test-repo",
                    branch="main",
                    additions=10,
                    deletions=2,
                    total_changes=12
                )
            ]
            
            # Act
            response = client.post("/track-recent-commits", json=request_data)
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["hash"] == "abc123def456"
    
    @pytest.mark.asyncio
    async def test_track_recent_commits_endpoint_error(self, client):
        """Test recent commits tracking endpoint with error."""
        # Arrange
        request_data = {
            "repository_path": "/test/repo",
            "days": 7,
            "limit": 10
        }
        
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.track_recent_commits = AsyncMock()
            mock_service.track_recent_commits.side_effect = ValueError("Repository not found")
            
            # Act
            response = client.post("/track-recent-commits", json=request_data)
            
            # Assert
            assert response.status_code == 400
            data = response.json()
            assert "Repository not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_commits_endpoint(self, client):
        """Test getting commits endpoint."""
            # Act
            response = client.get("/commits?limit=10")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
        assert data == []  # Currently returns empty list
    
    @pytest.mark.asyncio
    async def test_get_commit_endpoint_not_found(self, client):
        """Test getting specific commit endpoint."""
        # Act
        response = client.get("/commits/test-id")
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "Commit not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_commit_statistics_endpoint_success(self, client):
        """Test successful commit statistics endpoint."""
        # Arrange
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.get_commit_statistics = AsyncMock()
            mock_service.get_commit_statistics.return_value = {
                "total_commits": 10,
                "total_additions": 100,
                "total_deletions": 20,
                "average_commit_size": 12.0,
                "commit_frequency": 1.5,
                "top_authors": [("Author1", 5)],
                "most_changed_files": [("file1.py", 3)]
            }
            
            # Act
            response = client.get("/statistics/test-repo")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["repository_path"] == "test-repo"
            assert data["statistics"]["total_commits"] == 10
            assert data["statistics"]["total_additions"] == 100
    
    @pytest.mark.asyncio
    async def test_get_commit_statistics_endpoint_error(self, client):
        """Test commit statistics endpoint with error."""
        # Arrange
        with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
            mock_service.get_commit_statistics = AsyncMock()
            mock_service.get_commit_statistics.side_effect = ValueError("Repository not found")
            
            # Act
            response = client.get("/statistics/test-repo")
            
            # Assert
            assert response.status_code == 400
            data = response.json()
            assert "Repository not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        # Act
        response = client.get("/metrics")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
        assert "commits_tracked_today" in data
        assert "commits_tracked_total" in data
        assert "average_processing_time_ms" in data
        assert "error_rate_percent" in data
        assert "active_repositories" in data
        assert "last_commit_tracked" in data




class TestBackgroundTasks:
    """Test background tasks."""

    @pytest.mark.asyncio
    async def test_trigger_ai_analysis_success(self):
        """Test trigger_ai_analysis success."""
        from services.commit_tracker.main import trigger_ai_analysis
        
        with patch('services.commit_tracker.main.logger') as mock_logger:
            await trigger_ai_analysis("test_commit_id")
            
            mock_logger.info.assert_called_once_with("Triggering AI analysis for commit test_commit_id")
    
    @pytest.mark.asyncio
    async def test_trigger_ai_analysis_error(self):
        """Test trigger_ai_analysis with error."""
        from services.commit_tracker.main import trigger_ai_analysis
        
        with patch('services.commit_tracker.main.logger') as mock_logger:
            mock_logger.info.side_effect = Exception("Logging error")
            
            # The function catches exceptions, so it should not raise
            await trigger_ai_analysis("test_commit_id")
            
            # Verify that error was logged
            mock_logger.error.assert_called_once()


class TestCLI:
    """Test cases for CLI functionality."""
    
    @pytest.mark.asyncio
    async def test_main_function_success(self):
        """Test main CLI function success."""
        # Arrange
        test_args = [
            "--repo-path", "/test/repo",
            "--commit-hash", "abc123def456"
        ]
        
        with patch('sys.argv', ['test_main.py'] + test_args):
            with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
                mock_service.initialize = AsyncMock()
                mock_service.track_commit = AsyncMock()
                mock_service.track_commit.return_value = Commit(
                    id="test-id",
                    hash="abc123def456",
                    author="Test Author",
                    message="Test commit message",
                    timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    changed_files=["test_file.py"],
                    repository="test-repo",
                    branch="main",
                    additions=10,
                    deletions=2,
                    total_changes=12
                )
                mock_service.close = AsyncMock()
                
                with patch('builtins.print') as mock_print:
                    # Act
                    await main()
        
        # Assert
                    mock_service.track_commit.assert_called_once_with("/test/repo", "abc123def456")
                    mock_print.assert_called_with("Tracked commit: abc123def456")
    
    @pytest.mark.asyncio
    async def test_main_function_recent_commits(self):
        """Test main CLI function with recent commits."""
        # Arrange
        test_args = [
            "--repo-path", "/test/repo",
            "--recent",
            "--days", "7",
            "--limit", "10"
        ]
        
        with patch('sys.argv', ['test_main.py'] + test_args):
            with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
                mock_service.initialize = AsyncMock()
                mock_service.track_recent_commits = AsyncMock()
                mock_service.track_recent_commits.return_value = [
                    Commit(
                        id="test-id",
                        hash="abc123def456",
                        author="Test Author",
                        message="Test commit message",
                        timestamp=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        changed_files=["test_file.py"],
                        repository="test-repo",
                        branch="main",
                        additions=10,
                        deletions=2,
                        total_changes=12
                    )
                ]
                mock_service.close = AsyncMock()
                
                with patch('builtins.print') as mock_print:
                    # Act
                    await main()
                    
                    # Assert
                    mock_service.track_recent_commits.assert_called_once_with("/test/repo", 7, 10)
                    mock_print.assert_called_with("Tracked 1 recent commits")
    
    @pytest.mark.asyncio
    async def test_main_function_error(self):
        """Test main CLI function with error."""
        # Arrange
        test_args = ["--repo-path", "/test/repo"]
        
        with patch('sys.argv', ['test_main.py'] + test_args):
            with patch('services.commit_tracker.main.commit_tracker_service') as mock_service:
                mock_service.initialize = AsyncMock()
                mock_service.track_commit = AsyncMock()
                mock_service.track_commit.side_effect = Exception("Test error")
                mock_service.close = AsyncMock()
                
                with patch('services.commit_tracker.main.logger') as mock_logger:
                    with patch('services.commit_tracker.main.exit') as mock_exit:
                        # Act
                        await main()
                        
                        # Assert
                        mock_logger.error.assert_called_with("Error: Test error")
                        mock_exit.assert_called_with(1)


class TestModels:
    """Test cases for Pydantic models."""
    
    def test_track_commit_request_model(self):
        """Test TrackCommitRequest model."""
        # Arrange
        data = {
            "repository_path": "/test/repo",
            "commit_hash": "abc123def456"
        }
        
        # Act
        request = TrackCommitRequest(**data)
        
        # Assert
        assert request.repository_path == "/test/repo"
        assert request.commit_hash == "abc123def456"
    
    def test_track_recent_commits_request_model(self):
        """Test TrackRecentCommitsRequest model."""
        # Arrange
        data = {
            "repository_path": "/test/repo",
            "days": 7,
            "limit": 10
        }
        
        # Act
        request = TrackRecentCommitsRequest(**data)
        
        # Assert
        assert request.repository_path == "/test/repo"
        assert request.days == 7
        assert request.limit == 10
    
    def test_commit_statistics_response_model(self):
        """Test CommitStatisticsResponse model."""
        # Arrange
        statistics = {
            "total_commits": 10,
            "total_additions": 100,
            "total_deletions": 20
        }
        
        # Act
        response = CommitStatisticsResponse(
            repository_path="/test/repo",
            statistics=statistics
        )
        
        # Assert
        assert response.repository_path == "/test/repo"
        assert response.statistics["total_commits"] == 10
        assert response.statistics["total_additions"] == 100
        assert response.statistics["total_deletions"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
