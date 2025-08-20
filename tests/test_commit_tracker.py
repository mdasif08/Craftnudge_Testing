"""
Unit tests for CraftNudge Commit Tracker Service.

This module provides comprehensive testing for:
- Commit tracking functionality
- Event publishing
- Data validation
- Error handling
- CLI interface
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from shared.models import Commit, CommitCreate, CommitQuality, AnalysisStatus
from shared.events import EventType, EventSource, CommitData
from services.commit_tracker.main import CommitTrackerService


class TestCommitTrackerService:
    """Test cases for CommitTrackerService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CommitTrackerService()
    
    @pytest.fixture
    def sample_commit_data(self):
        """Sample commit data for testing."""
        return {
            "hash": "abc123def456",
            "author": "Test User <test@example.com>",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py", "tests/test_main.py"],
            "repository": "test-repo",
            "branch": "main",
            "additions": 50,
            "deletions": 10,
            "total_changes": 60
        }
    
    @pytest.mark.asyncio
    async def test_initialize_service(self, service):
        """Test service initialization."""
        with patch('redis.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            
            await service.initialize()
            
            mock_redis.assert_called_once()
            mock_redis.return_value.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_commit_data(self, service, sample_commit_data):
        """Test commit data extraction."""
        # Mock Git commit object
        mock_commit = Mock()
        mock_commit.hexsha = sample_commit_data["hash"]
        mock_commit.author = sample_commit_data["author"]
        mock_commit.message = sample_commit_data["message"]
        mock_commit.committed_date = sample_commit_data["timestamp"].timestamp()
        
        # Mock commit stats
        mock_stats = Mock()
        mock_stats.total = {
            'insertions': sample_commit_data["additions"],
            'deletions': sample_commit_data["deletions"]
        }
        mock_commit.stats = mock_stats
        
        # Mock repository
        mock_repo = Mock()
        mock_repo.active_branch.name = sample_commit_data["branch"]
        mock_repo.head.is_detached = False
        
        with patch('git.Repo', return_value=mock_repo):
            with patch('os.path.basename', return_value=sample_commit_data["repository"]):
                # Mock diff
                mock_diff_item = Mock()
                mock_diff_item.a_path = "src/main.py"
                mock_commit.diff.return_value = [mock_diff_item]
                mock_commit.parents = [Mock()]  # Has parent
                
                result = service.extract_commit_data(mock_commit, "/path/to/repo")
                
                assert result["hash"] == sample_commit_data["hash"]
                assert result["author"] == sample_commit_data["author"]
                assert result["message"] == sample_commit_data["message"]
                assert result["repository"] == sample_commit_data["repository"]
                assert result["branch"] == sample_commit_data["branch"]
                assert result["additions"] == sample_commit_data["additions"]
                assert result["deletions"] == sample_commit_data["deletions"]
                assert result["total_changes"] == sample_commit_data["total_changes"]
                assert "src/main.py" in result["changed_files"]
    
    @pytest.mark.asyncio
    async def test_track_commit_success(self, service, sample_commit_data):
        """Test successful commit tracking."""
        with patch('os.path.exists', return_value=True):
            with patch('git.Repo') as mock_repo_class:
                # Mock repository
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                
                # Mock commit
                mock_commit = Mock()
                mock_commit.hexsha = sample_commit_data["hash"]
                mock_commit.author = sample_commit_data["author"]
                mock_commit.message = sample_commit_data["message"]
                mock_commit.committed_date = sample_commit_data["timestamp"].timestamp()
                
                # Mock commit stats
                mock_stats = Mock()
                mock_stats.total = {
                    'insertions': sample_commit_data["additions"],
                    'deletions': sample_commit_data["deletions"]
                }
                mock_commit.stats = mock_stats
                
                # Mock repository info
                mock_repo.active_branch.name = sample_commit_data["branch"]
                mock_repo.head.is_detached = False
                mock_repo.head.commit = mock_commit
                
                # Mock diff
                mock_diff_item = Mock()
                mock_diff_item.a_path = "src/main.py"
                mock_commit.diff.return_value = [mock_diff_item]
                mock_commit.parents = [Mock()]
                
                with patch('os.path.basename', return_value=sample_commit_data["repository"]):
                    with patch.object(service, 'db_service') as mock_db:
                        with patch.object(service, 'save_commit_locally') as mock_save:
                            with patch.object(service, 'publish_commit_event') as mock_publish:
                                # Mock database service
                                mock_commit_model = Commit(**sample_commit_data)
                                mock_db.store_commit.return_value = mock_commit_model
                                mock_db.get_commit_by_hash.return_value = None  # Commit doesn't exist
                                
                                result = await service.track_commit("/path/to/repo")
                                
                                assert result.hash == sample_commit_data["hash"]
                                mock_db.store_commit.assert_called_once()
                                mock_save.assert_called_once_with(result)
                                mock_publish.assert_called_once_with(result)
    
    @pytest.mark.asyncio
    async def test_track_commit_repository_not_found(self, service):
        """Test commit tracking with non-existent repository."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(ValueError, match="Repository path does not exist"):
                await service.track_commit("/non/existent/path")
    
    @pytest.mark.asyncio
    async def test_track_commit_invalid_git_repo(self, service):
        """Test commit tracking with invalid Git repository."""
        with patch('os.path.exists', return_value=True):
            with patch('git.Repo', side_effect=Exception("Invalid Git repository")):
                with pytest.raises(ValueError, match="Invalid Git repository"):
                    await service.track_commit("/path/to/repo")
    
    @pytest.mark.asyncio
    async def test_save_commit_locally(self, service, sample_commit_data):
        """Test local commit storage."""
        commit = Commit(**sample_commit_data)
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('builtins.open', create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                await service.save_commit_locally(commit)
                
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_open.assert_called_once()
                mock_file.write.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_commit_event(self, service, sample_commit_data):
        """Test commit event publishing."""
        commit = Commit(**sample_commit_data)
        
        with patch.object(service, 'redis_client') as mock_redis:
            with patch.object(service, 'event_factory') as mock_factory:
                # Mock event factory
                mock_event = Mock()
                mock_factory.create_commit_event.return_value = mock_event
                mock_event.json.return_value = '{"test": "event"}'
                
                await service.publish_commit_event(commit)
                
                mock_factory.create_commit_event.assert_called_once()
                mock_redis.publish.assert_called_once_with(
                    "commit_events",
                    '{"test": "event"}'
                )
    
    @pytest.mark.asyncio
    async def test_get_commit_statistics(self, service):
        """Test commit statistics calculation."""
        with patch('git.Repo') as mock_repo_class:
            # Mock repository
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            # Mock commits
            mock_commit1 = Mock()
            mock_commit1.stats.total = {'insertions': 10, 'deletions': 5}
            mock_commit1.author = "User 1"
            mock_commit1.committed_date = datetime.now(timezone.utc).timestamp()
            mock_commit1.parents = [Mock()]
            
            mock_commit2 = Mock()
            mock_commit2.stats.total = {'insertions': 20, 'deletions': 10}
            mock_commit2.author = "User 2"
            mock_commit2.committed_date = datetime.now(timezone.utc).timestamp()
            mock_commit2.parents = [Mock()]
            
            mock_repo.iter_commits.return_value = [mock_commit1, mock_commit2]
            
            # Mock diff
            mock_diff_item = Mock()
            mock_diff_item.a_path = "test.py"
            mock_commit1.diff.return_value = [mock_diff_item]
            mock_commit2.diff.return_value = [mock_diff_item]
            
            result = await service.get_commit_statistics("/path/to/repo")
            
            assert result["total_commits"] == 2
            assert result["total_additions"] == 30
            assert result["total_deletions"] == 15
            assert result["average_commit_size"] == 22.5
            assert len(result["top_authors"]) == 2


class TestCommitModels:
    """Test cases for commit models."""
    
    def test_commit_creation(self):
        """Test commit model creation."""
        commit_data = {
            "hash": "abc123def456",
            "author": "Test User <test@example.com>",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
            "repository": "test-repo",
            "branch": "main",
            "additions": 50,
            "deletions": 10,
            "total_changes": 60
        }
        
        commit = Commit(**commit_data)
        
        assert commit.hash == commit_data["hash"]
        assert commit.author == commit_data["author"]
        assert commit.message == commit_data["message"]
        assert commit.repository == commit_data["repository"]
        assert commit.branch == commit_data["branch"]
        assert commit.additions == commit_data["additions"]
        assert commit.deletions == commit_data["deletions"]
        assert commit.total_changes == commit_data["total_changes"]
        assert commit.analysis_status == AnalysisStatus.PENDING
    
    def test_commit_validation(self):
        """Test commit validation."""
        # Test invalid hash
        with pytest.raises(ValueError, match="Commit hash must be at least 7 characters"):
            CommitCreate(
                hash="abc",
                author="Test User",
                message="test",
                timestamp=datetime.now(timezone.utc),
                repository="test",
                branch="main"
            )
        
        # Test empty message
        with pytest.raises(ValueError, match="Commit message cannot be empty"):
            CommitCreate(
                hash="abc123def456",
                author="Test User",
                message="",
                timestamp=datetime.now(timezone.utc),
                repository="test",
                branch="main"
            )
    
    def test_commit_computed_fields(self):
        """Test computed fields in commit model."""
        commit_data = {
            "hash": "abc123def456",
            "author": "Test User <test@example.com>",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
            "repository": "test-repo",
            "branch": "main",
            "additions": 50,
            "deletions": 10,
            "total_changes": 60
        }
        
        commit = Commit(**commit_data)
        
        # Test change ratio
        assert commit.change_ratio == 5.0  # 50 additions / 10 deletions
        
        # Test commit type detection
        assert commit.commit_type.value == "feature"  # "feat" in message
    
    def test_quality_score_validation(self):
        """Test quality score validation and level assignment."""
        commit_data = {
            "hash": "abc123def456",
            "author": "Test User <test@example.com>",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
            "repository": "test-repo",
            "branch": "main",
            "additions": 50,
            "deletions": 10,
            "total_changes": 60,
            "quality_score": 9.5
        }
        
        commit = Commit(**commit_data)
        
        assert commit.quality_score == 9.5
        assert commit.quality_level == CommitQuality.EXCELLENT
        
        # Test invalid quality score
        with pytest.raises(ValueError, match="Quality score must be between 0 and 10"):
            Commit(
                **commit_data,
                quality_score=15.0
            )


class TestEventSystem:
    """Test cases for event system integration."""
    
    def test_commit_data_creation(self):
        """Test CommitData model creation."""
        commit_data = CommitData(
            hash="abc123def456",
            author="Test User <test@example.com>",
            message="feat: add new feature",
            timestamp=datetime.now(timezone.utc),
            changed_files=["src/main.py"],
            repository="test-repo",
            branch="main",
            additions=50,
            deletions=10,
            total_changes=60
        )
        
        assert commit_data.hash == "abc123def456"
        assert commit_data.author == "Test User <test@example.com>"
        assert commit_data.message == "feat: add new feature"
        assert len(commit_data.changed_files) == 1
        assert commit_data.changed_files[0] == "src/main.py"
    
    def test_commit_data_validation(self):
        """Test CommitData validation."""
        # Test invalid hash
        with pytest.raises(ValueError, match="Commit hash must be at least 7 characters"):
            CommitData(
                hash="abc",
                author="Test User",
                message="test",
                timestamp=datetime.now(timezone.utc),
                repository="test",
                branch="main"
            )
        
        # Test empty message
        with pytest.raises(ValueError, match="Commit message cannot be empty"):
            CommitData(
                hash="abc123def456",
                author="Test User",
                message="",
                timestamp=datetime.now(timezone.utc),
                repository="test",
                branch="main"
            )


if __name__ == "__main__":
    pytest.main([__file__])
