"""
Unit tests for the Commit Quality Coaching Service.

This module tests the AI-powered coaching and feedback functionality
for commit quality improvement.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from shared.models import Commit, CommitQuality, AnalysisStatus
from shared.events import EventType, EventSource, EventFactory
from services.commit_quality_coaching.main import (
    CommitQualityCoachingService,
    CoachingFeedback,
    CoachingSession,
    UserProgress,
    CoachingRequest,
    CoachingResponse
)


class TestCommitQualityCoachingService:
    """Test cases for the CommitQualityCoachingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CommitQualityCoachingService()
    
    @pytest.fixture
    def sample_commit(self):
        """Create a sample commit for testing."""
        return Commit(
            id="test-commit-123",
            hash="abc123def456",
            author="testuser",
            message="Add new feature for user authentication",
            timestamp=datetime.now(timezone.utc),
            changed_files=["src/auth.py", "tests/test_auth.py", "docs/auth.md"],
            repository="test-repo",
            branch="main",
            additions=150,
            deletions=20,
            total_changes=170
        )
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis result."""
        return {
            "quality_score": 7.5,
            "quality_level": CommitQuality.GOOD,
            "message_score": 8.0,
            "file_score": 7.0,
            "size_score": 7.5,
            "analysis_details": {
                "message_analysis": {
                    "length": 45,
                    "has_imperative": True,
                    "has_issue_reference": False,
                    "is_descriptive": True
                },
                "file_analysis": {
                    "total_files": 3,
                    "unique_extensions": 3,
                    "extension_distribution": {"py": 2, "md": 1}
                },
                "size_analysis": {
                    "file_count": 3,
                    "size_category": "small"
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_initialize_service(self, service):
        """Test service initialization."""
        with patch('services.commit_quality_coaching.main.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            await service.initialize()
            
            assert service.redis_client is not None
            assert service.event_factory is not None
            assert service.event_serializer is not None
    
    @pytest.mark.asyncio
    async def test_get_commit_data(self, service):
        """Test getting commit data from database."""
        with patch('services.commit_quality_coaching.main.AsyncSession') as mock_session:
            # For now, this returns None as it's a placeholder
            result = await service.get_commit_data("test-commit-id", mock_session)
            assert result is None
    
    def test_analyze_message_quality_excellent(self, service):
        """Test message quality analysis for excellent commit message."""
        message = "Add comprehensive user authentication system with OAuth2 support and role-based access control"
        score = service._analyze_message_quality(message)
        
        assert score >= 8.0
        assert score <= 10.0
    
    def test_analyze_message_quality_poor(self, service):
        """Test message quality analysis for poor commit message."""
        message = "fix"
        score = service._analyze_message_quality(message)
        
        assert score < 5.0
    
    def test_analyze_message_quality_with_issue_reference(self, service):
        """Test message quality analysis with issue reference."""
        message = "Fix authentication bug #123"
        score = service._analyze_message_quality(message)
        
        # Should get bonus for issue reference
        assert score > 5.0
    
    def test_analyze_file_changes_small_commit(self, service):
        """Test file changes analysis for small commit."""
        files = ["src/main.py", "tests/test_main.py"]
        score = service._analyze_file_changes(files)
        
        assert score >= 6.0  # Should score well for small, focused commit
    
    def test_analyze_file_changes_large_commit(self, service):
        """Test file changes analysis for large commit."""
        files = [f"file_{i}.py" for i in range(25)]  # 25 files
        score = service._analyze_file_changes(files)
        
        assert score < 5.0  # Should score poorly for large commit
    
    def test_analyze_commit_size_small(self, service):
        """Test commit size analysis for small commit."""
        files = ["file1.py", "file2.py"]
        score = service._analyze_commit_size(files)
        
        assert score >= 8.0  # Should score well for small commit
    
    def test_analyze_commit_size_large(self, service):
        """Test commit size analysis for large commit."""
        files = [f"file_{i}.py" for i in range(15)]  # 15 files
        score = service._analyze_commit_size(files)
        
        assert score < 6.0  # Should score poorly for large commit
    
    @pytest.mark.asyncio
    async def test_analyze_commit_quality(self, service, sample_commit):
        """Test complete commit quality analysis."""
        analysis = await service.analyze_commit_quality(sample_commit)
        
        assert "quality_score" in analysis
        assert "quality_level" in analysis
        assert "message_score" in analysis
        assert "file_score" in analysis
        assert "size_score" in analysis
        assert "analysis_details" in analysis
        
        assert 0 <= analysis["quality_score"] <= 10
        assert isinstance(analysis["quality_level"], CommitQuality)
    
    @pytest.mark.asyncio
    async def test_generate_coaching_feedback(self, service, sample_commit, sample_analysis):
        """Test coaching feedback generation."""
        feedback = await service.generate_coaching_feedback(sample_commit, sample_analysis)
        
        assert isinstance(feedback, CoachingFeedback)
        assert feedback.commit_id == sample_commit.id
        assert feedback.user_id == sample_commit.author
        assert feedback.quality_score == sample_analysis["quality_score"]
        assert feedback.quality_level == sample_analysis["quality_level"]
        assert isinstance(feedback.strengths, list)
        assert isinstance(feedback.areas_for_improvement, list)
        assert isinstance(feedback.specific_recommendations, list)
        assert isinstance(feedback.coaching_tips, list)
        assert isinstance(feedback.next_steps, list)
    
    @pytest.mark.asyncio
    async def test_get_user_progress(self, service):
        """Test user progress retrieval."""
        progress = await service.get_user_progress("testuser", days=30)
        
        assert isinstance(progress, UserProgress)
        assert progress.user_id == "testuser"
        assert progress.total_commits > 0
        assert 0 <= progress.average_quality_score <= 10
        assert isinstance(progress.quality_distribution, dict)
        assert isinstance(progress.improvement_areas, list)
        assert isinstance(progress.regression_areas, list)
        assert 0 <= progress.consistency_score <= 10
    
    @pytest.mark.asyncio
    async def test_publish_coaching_event(self, service):
        """Test publishing coaching events to Redis."""
        feedback = CoachingFeedback(
            commit_id="test-commit-123",
            user_id="testuser",
            quality_score=7.5,
            quality_level=CommitQuality.GOOD,
            strengths=["Good commit message structure"],
            areas_for_improvement=["Commit size management"],
            specific_recommendations=["Keep commits smaller"],
            coaching_tips=["Review before pushing"],
            next_steps=["Practice smaller commits"]
        )
        
        with patch.object(service, 'redis_client') as mock_redis:
            mock_redis.publish = AsyncMock()
            await service.publish_coaching_event(feedback)
            
            mock_redis.publish.assert_called_once()
            call_args = mock_redis.publish.call_args
            assert call_args[0][0] == "coaching_events"
    
    def test_get_message_analysis(self, service):
        """Test detailed message analysis."""
        message = "Add user authentication with OAuth2 support #123"
        analysis = service._get_message_analysis(message)
        
        assert analysis["length"] == len(message)
        assert analysis["has_imperative"] == True
        assert analysis["has_issue_reference"] == True
        assert analysis["is_descriptive"] == True
    
    def test_get_file_analysis(self, service):
        """Test detailed file analysis."""
        files = ["src/auth.py", "tests/test_auth.py", "docs/auth.md", "src/utils.py"]
        analysis = service._get_file_analysis(files)
        
        assert analysis["total_files"] == 4
        assert analysis["unique_extensions"] == 3  # py, py, md, py
        assert "py" in analysis["extension_distribution"]
        assert "md" in analysis["extension_distribution"]
        assert analysis["extension_distribution"]["py"] == 3
        assert analysis["extension_distribution"]["md"] == 1
    
    def test_get_size_analysis(self, service):
        """Test detailed size analysis."""
        files = ["file1.py", "file2.py", "file3.py", "file4.py", "file5.py"]
        analysis = service._get_size_analysis(files)
        
        assert analysis["file_count"] == 5
        assert analysis["size_category"] == "medium"


class TestCoachingModels:
    """Test cases for coaching data models."""
    
    def test_coaching_feedback_creation(self):
        """Test CoachingFeedback model creation."""
        feedback = CoachingFeedback(
            commit_id="test-commit-123",
            user_id="testuser",
            quality_score=8.5,
            quality_level=CommitQuality.EXCELLENT,
            strengths=["Clear commit message", "Good file organization"],
            areas_for_improvement=["Consider smaller commits"],
            specific_recommendations=["Break down large changes"],
            coaching_tips=["Review before pushing"],
            next_steps=["Practice atomic commits"]
        )
        
        assert feedback.commit_id == "test-commit-123"
        assert feedback.user_id == "testuser"
        assert feedback.quality_score == 8.5
        assert feedback.quality_level == CommitQuality.EXCELLENT
        assert len(feedback.strengths) == 2
        assert len(feedback.areas_for_improvement) == 1
        assert len(feedback.specific_recommendations) == 1
        assert len(feedback.coaching_tips) == 1
        assert len(feedback.next_steps) == 1
    
    def test_coaching_session_creation(self):
        """Test CoachingSession model creation."""
        session = CoachingSession(
            user_id="testuser",
            commits_reviewed=["commit1", "commit2"],
            insights_generated=["Improved message clarity"],
            goals_set=["Write clearer commit messages"]
        )
        
        assert session.user_id == "testuser"
        assert session.session_id is not None
        assert session.start_time is not None
        assert session.end_time is None
        assert len(session.commits_reviewed) == 2
        assert len(session.insights_generated) == 1
        assert len(session.goals_set) == 1
    
    def test_user_progress_creation(self):
        """Test UserProgress model creation."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        progress = UserProgress(
            user_id="testuser",
            period_start=start_date,
            period_end=end_date,
            total_commits=25,
            average_quality_score=7.2,
            quality_distribution={
                "EXCELLENT": 5,
                "GOOD": 12,
                "AVERAGE": 6,
                "POOR": 2
            },
            improvement_areas=["Commit message clarity"],
            regression_areas=["Commit size consistency"],
            consistency_score=6.8
        )
        
        assert progress.user_id == "testuser"
        assert progress.total_commits == 25
        assert progress.average_quality_score == 7.2
        assert progress.consistency_score == 6.8
        assert len(progress.quality_distribution) == 4
        assert len(progress.improvement_areas) == 1
        assert len(progress.regression_areas) == 1
    
    def test_coaching_request_creation(self):
        """Test CoachingRequest model creation."""
        request = CoachingRequest(
            commit_id="test-commit-123",
            user_id="testuser",
            include_historical_context=True,
            focus_areas=["commit_message", "file_organization"]
        )
        
        assert request.commit_id == "test-commit-123"
        assert request.user_id == "testuser"
        assert request.include_historical_context == True
        assert request.focus_areas == ["commit_message", "file_organization"]
    
    def test_coaching_response_creation(self):
        """Test CoachingResponse model creation."""
        feedback = CoachingFeedback(
            commit_id="test-commit-123",
            user_id="testuser",
            quality_score=7.5,
            quality_level=CommitQuality.GOOD
        )
        
        response = CoachingResponse(
            feedback=feedback,
            recommendations=["Focus on clear messages", "Keep commits small"]
        )
        
        assert response.feedback == feedback
        assert len(response.recommendations) == 2
        assert response.session is None
        assert response.progress is None


class TestQualityAnalysis:
    """Test cases for quality analysis algorithms."""
    
    @pytest.fixture
    def service(self):
        return CommitQualityCoachingService()
    
    def test_message_quality_imperative_mood(self, service):
        """Test message quality scoring for imperative mood."""
        good_messages = [
            "Add user authentication",
            "Fix authentication bug",
            "Update documentation",
            "Remove unused code",
            "Refactor authentication module"
        ]
        
        for message in good_messages:
            score = service._analyze_message_quality(message)
            assert score >= 6.0  # Should score well for imperative mood
    
    def test_message_quality_descriptive_content(self, service):
        """Test message quality scoring for descriptive content."""
        descriptive_messages = [
            "Add OAuth2 authentication because users need secure login",
            "Fix authentication bug when user session expires",
            "Update documentation to explain the new API endpoints"
        ]
        
        for message in descriptive_messages:
            score = service._analyze_message_quality(message)
            assert score >= 7.0  # Should score well for descriptive content
    
    def test_message_quality_issue_references(self, service):
        """Test message quality scoring for issue references."""
        messages_with_issues = [
            "Fix authentication bug #123",
            "Add feature requested in GH-456",
            "Update docs for JIRA-789"
        ]
        
        for message in messages_with_issues:
            score = service._analyze_message_quality(message)
            assert score >= 5.5  # Should get bonus for issue references
    
    def test_file_changes_focused_commit(self, service):
        """Test file changes scoring for focused commits."""
        focused_files = [
            ["src/auth.py", "tests/test_auth.py"],  # Related files
            ["docs/api.md", "docs/installation.md"],  # Documentation
            ["src/models.py", "src/serializers.py", "src/views.py"]  # Related modules
        ]
        
        for files in focused_files:
            score = service._analyze_file_changes(files)
            assert score >= 6.0  # Should score well for focused commits
    
    def test_file_changes_mixed_concerns(self, service):
        """Test file changes scoring for mixed concerns."""
        mixed_files = [
            "src/auth.py",
            "src/models.py", 
            "docs/api.md",
            "tests/test_auth.py",
            "scripts/deploy.sh",
            "config/database.yml"
        ]
        
        score = service._analyze_file_changes(mixed_files)
        assert score < 6.0  # Should score poorly for mixed concerns
    
    def test_commit_size_optimal(self, service):
        """Test commit size scoring for optimal sizes."""
        optimal_sizes = [
            ["file1.py"],  # 1 file
            ["file1.py", "file2.py"],  # 2 files
            ["file1.py", "file2.py", "file3.py"]  # 3 files
        ]
        
        for files in optimal_sizes:
            score = service._analyze_commit_size(files)
            assert score >= 8.0  # Should score well for optimal sizes
    
    def test_commit_size_too_large(self, service):
        """Test commit size scoring for large commits."""
        large_commits = [
            [f"file_{i}.py" for i in range(15)],  # 15 files
            [f"file_{i}.py" for i in range(25)],  # 25 files
            [f"file_{i}.py" for i in range(50)]   # 50 files
        ]
        
        for files in large_commits:
            score = service._analyze_commit_size(files)
            assert score < 6.0  # Should score poorly for large commits


class TestErrorHandling:
    """Test cases for error handling."""
    
    @pytest.fixture
    def service(self):
        return CommitQualityCoachingService()
    
    @pytest.mark.asyncio
    async def test_analyze_commit_quality_with_error(self, service):
        """Test commit quality analysis with error handling."""
        # Create a commit that might cause issues
        problematic_commit = Commit(
            id="problematic-commit",
            hash="abc123",
            author="testuser",
            message="",  # Empty message
            timestamp=datetime.now(timezone.utc),
            changed_files=[],  # Empty file list
            repository="test-repo",
            branch="main",
            additions=0,
            deletions=0,
            total_changes=0
        )
        
        analysis = await service.analyze_commit_quality(problematic_commit)
        
        # Should still return a valid analysis with default values
        assert "quality_score" in analysis
        assert "quality_level" in analysis
        assert analysis["quality_level"] == CommitQuality.AVERAGE
    
    @pytest.mark.asyncio
    async def test_publish_coaching_event_with_error(self, service):
        """Test publishing coaching events with error handling."""
        feedback = CoachingFeedback(
            commit_id="test-commit-123",
            user_id="testuser",
            quality_score=7.5,
            quality_level=CommitQuality.GOOD
        )
        
        # Mock Redis to raise an exception
        with patch.object(service, 'redis_client') as mock_redis:
            mock_redis.publish = AsyncMock(side_effect=Exception("Redis error"))
            
            # Should not raise an exception
            await service.publish_coaching_event(feedback)
            
            # Should have attempted to publish
            mock_redis.publish.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
