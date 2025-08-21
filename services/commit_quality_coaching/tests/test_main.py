"""
Unit tests for Commit Quality Coaching Service main module.

This module tests the core functionality of the Commit Quality Coaching microservice,
including API endpoints, business logic, and AI-powered coaching features.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from services.commit_quality_coaching.main import app, CommitQualityCoachingService
from shared.models import CoachingRequest, CoachingResponse, CoachingSession
from shared.events import Event, EventType, EventSource


class TestCommitQualityCoachingService:
    """Test cases for CommitQualityCoachingService class."""
    
    @pytest.fixture
    def service(self):
        """Create a CommitQualityCoachingService instance for testing."""
        return CommitQualityCoachingService()
    
    @pytest.fixture
    def mock_commit_data(self):
        """Mock commit data for testing."""
        return {
            "hash": "abc123def456",
            "author": "Test Author",
            "message": "feat: add new feature with comprehensive testing",
            "changed_files": [
                {"path": "src/feature.py", "status": "A"},
                {"path": "tests/test_feature.py", "status": "A"}
            ],
            "stats": {"insertions": 150, "deletions": 20}
        }
    
    def test_service_initialization(self, service):
        """Test that service initializes correctly."""
        assert service is not None
        assert hasattr(service, 'settings')
        assert hasattr(service, 'event_bus')
        assert hasattr(service, 'ollama_client')
    
    @pytest.mark.asyncio
    async def test_analyze_commit_quality(self, service, mock_commit_data):
        """Test commit quality analysis."""
        # Act
        result = await service.analyze_commit_quality(mock_commit_data)
        
        # Assert
        assert result is not None
        assert "message_quality" in result
        assert "file_changes" in result
        assert "commit_size" in result
        
        # Check message quality analysis
        message_quality = result["message_quality"]
        assert "rating" in message_quality
        assert "comment" in message_quality
        assert isinstance(message_quality["rating"], (int, float))
        assert message_quality["rating"] >= 0 and message_quality["rating"] <= 10
        
        # Check file changes analysis
        file_changes = result["file_changes"]
        assert "rating" in file_changes
        assert "comment" in file_changes
        assert isinstance(file_changes["rating"], (int, float))
        
        # Check commit size analysis
        commit_size = result["commit_size"]
        assert "rating" in commit_size
        assert "comment" in commit_size
        assert isinstance(commit_size["rating"], (int, float))
    
    @pytest.mark.asyncio
    async def test_generate_coaching_feedback(self, service, mock_commit_data):
        """Test coaching feedback generation."""
        # Arrange
        quality_analysis = {
            "message_quality": {"rating": 8, "comment": "Good commit message"},
            "file_changes": {"rating": 7, "comment": "Reasonable file changes"},
            "commit_size": {"rating": 6, "comment": "Slightly large commit"}
        }
        
        # Act
        result = await service.generate_coaching_feedback(mock_commit_data, quality_analysis)
        
        # Assert
        assert result is not None
        assert "session_id" in result
        assert "commit_hash" in result
        assert "quality_score" in result
        assert "overall_rating" in result
        assert "feedback" in result
        assert "suggestions" in result
        assert "next_steps" in result
        
        # Check feedback structure
        feedback = result["feedback"]
        assert "message_quality" in feedback
        assert "file_changes" in feedback
        assert "commit_size" in feedback
        
        # Check suggestions and next steps
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["next_steps"], list)
        assert len(result["suggestions"]) > 0
        assert len(result["next_steps"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_user_progress(self, service):
        """Test user progress retrieval."""
        # Arrange
        user_id = "test-user-123"
        
        # Act
        result = await service.get_user_progress(user_id)
        
        # Assert
        assert result is not None
        assert "user_id" in result
        assert "overall_progress" in result
        assert "average_quality_score" in result
        assert "total_commits" in result
        assert "metrics" in result
        assert "insights" in result
        
        # Check metrics structure
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check insights
        insights = result["insights"]
        assert isinstance(insights, list)
        assert len(insights) > 0
    
    @pytest.mark.asyncio
    async def test_create_coaching_session(self, service):
        """Test coaching session creation."""
        # Arrange
        user_id = "test-user-123"
        goals = "Improve commit message quality and reduce commit size"
        
        # Act
        result = await service.create_coaching_session(user_id, goals)
        
        # Assert
        assert result is not None
        assert "session_id" in result
        assert "user_id" in result
        assert "goals" in result
        assert "created_at" in result
        assert "progress" in result
        assert "activities" in result
        
        assert result["user_id"] == user_id
        assert result["goals"] == goals
        assert result["progress"] == 0  # New session starts at 0%
        assert isinstance(result["activities"], list)
    
    @pytest.mark.asyncio
    async def test_get_coaching_session(self, service):
        """Test coaching session retrieval."""
        # Arrange
        session_id = "test-session-123"
        
        # Act
        result = await service.get_coaching_session(session_id)
        
        # Assert
        assert result is not None
        assert "session_id" in result
        assert "user_id" in result
        assert "goals" in result
        assert "created_at" in result
        assert "progress" in result
        assert "activities" in result
        
        assert result["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_publish_coaching_event(self, service):
        """Test publishing coaching events."""
        # Arrange
        coaching_data = {
            "session_id": "test-session-123",
            "user_id": "test-user-123",
            "commit_hash": "abc123",
            "quality_score": 85
        }
        
        # Mock event bus
        service.event_bus.publish = AsyncMock()
        
        # Act
        await service.publish_coaching_event(coaching_data)
        
        # Assert
        service.event_bus.publish.assert_called_once()
        call_args = service.event_bus.publish.call_args[0][0]
        assert isinstance(call_args, Event)
        assert call_args.event_type == EventType.COACHING_PROVIDED
        assert call_args.source == EventSource.COMMIT_QUALITY_COACHING


class TestCommitQualityCoachingAPI:
    """Test cases for Commit Quality Coaching API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_coach_commit_endpoint_success(self, client):
        """Test successful commit coaching endpoint."""
        # Arrange
        with patch('services.commit_quality_coaching.main.CommitQualityCoachingService') as mock_service:
            mock_instance = AsyncMock()
            mock_instance.analyze_commit_quality.return_value = {
                "message_quality": {"rating": 8, "comment": "Good message"},
                "file_changes": {"rating": 7, "comment": "Good changes"},
                "commit_size": {"rating": 6, "comment": "Reasonable size"}
            }
            mock_instance.generate_coaching_feedback.return_value = {
                "session_id": "test-session",
                "commit_hash": "abc123",
                "quality_score": 85,
                "overall_rating": "Good",
                "feedback": {},
                "suggestions": ["Write more descriptive commit messages"],
                "next_steps": ["Practice with smaller commits"]
            }
            mock_service.return_value = mock_instance
            
            # Act
            response = client.post("/coach-commit", json={"commit_hash": "abc123"})
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert "commit_hash" in data
            assert "quality_score" in data
            assert "suggestions" in data
            assert "next_steps" in data
    
    @pytest.mark.asyncio
    async def test_coach_commit_endpoint_with_repo_path(self, client):
        """Test commit coaching with specific repository path."""
        # Arrange
        with patch('services.commit_quality_coaching.main.CommitQualityCoachingService') as mock_service:
            mock_instance = AsyncMock()
            mock_instance.analyze_commit_quality.return_value = {
                "message_quality": {"rating": 8, "comment": "Good message"},
                "file_changes": {"rating": 7, "comment": "Good changes"},
                "commit_size": {"rating": 6, "comment": "Reasonable size"}
            }
            mock_instance.generate_coaching_feedback.return_value = {
                "session_id": "test-session",
                "commit_hash": "abc123",
                "quality_score": 85,
                "overall_rating": "Good",
                "feedback": {},
                "suggestions": [],
                "next_steps": []
            }
            mock_service.return_value = mock_instance
            
            # Act
            response = client.post("/coach-commit", json={
                "commit_hash": "abc123",
                "repo_path": "/test/repo"
            })
            
            # Assert
            assert response.status_code == 200
            mock_instance.analyze_commit_quality.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_coach_commit_endpoint_error(self, client):
        """Test commit coaching endpoint with error."""
        # Arrange
        with patch('services.commit_quality_coaching.main.CommitQualityCoachingService') as mock_service:
            mock_instance = AsyncMock()
            mock_instance.analyze_commit_quality.side_effect = Exception("Commit not found")
            mock_service.return_value = mock_instance
            
            # Act
            response = client.post("/coach-commit", json={"commit_hash": "invalid"})
            
            # Assert
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Commit not found" in data["error"]
    
    @pytest.mark.asyncio
    async def test_get_user_progress_endpoint(self, client):
        """Test getting user progress endpoint."""
        # Arrange
        with patch('services.commit_quality_coaching.main.CommitQualityCoachingService') as mock_service:
            mock_instance = AsyncMock()
            mock_instance.get_user_progress.return_value = {
                "user_id": "test-user",
                "overall_progress": 75,
                "average_quality_score": 82,
                "total_commits": 25,
                "metrics": {"message_quality": 8.5, "commit_size": 7.2},
                "insights": ["You're improving steadily", "Consider smaller commits"]
            }
            mock_service.return_value = mock_instance
            
            # Act
            response = client.get("/user-progress?user_id=test-user")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "user_id" in data
            assert "overall_progress" in data
            assert "average_quality_score" in data
            assert "metrics" in data
            assert "insights" in data
    
    @pytest.mark.asyncio
    async def test_create_coaching_session_endpoint(self, client):
        """Test creating coaching session endpoint."""
        # Arrange
        with patch('services.commit_quality_coaching.main.CommitQualityCoachingService') as mock_service:
            mock_instance = AsyncMock()
            mock_instance.create_coaching_session.return_value = {
                "session_id": "test-session",
                "user_id": "test-user",
                "goals": "Improve commit quality",
                "created_at": "2024-01-01T12:00:00",
                "progress": 0,
                "activities": []
            }
            mock_service.return_value = mock_instance
            
            # Act
            response = client.post("/coaching-sessions", json={
                "user_id": "test-user",
                "goals": "Improve commit quality"
            })
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert "user_id" in data
            assert "goals" in data
            assert "progress" in data
    
    @pytest.mark.asyncio
    async def test_get_coaching_session_endpoint(self, client):
        """Test getting coaching session endpoint."""
        # Arrange
        with patch('services.commit_quality_coaching.main.CommitQualityCoachingService') as mock_service:
            mock_instance = AsyncMock()
            mock_instance.get_coaching_session.return_value = {
                "session_id": "test-session",
                "user_id": "test-user",
                "goals": "Improve commit quality",
                "created_at": "2024-01-01T12:00:00",
                "progress": 25,
                "activities": [
                    {"date": "2024-01-01", "description": "First coaching session", "score": 75}
                ]
            }
            mock_service.return_value = mock_instance
            
            # Act
            response = client.get("/coaching-sessions/test-session")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert "user_id" in data
            assert "goals" in data
            assert "progress" in data
            assert "activities" in data


class TestCommitQualityCoachingCLI:
    """Test cases for Commit Quality Coaching CLI tool."""
    
    @pytest.fixture
    def cli(self):
        """Create a CLI instance for testing."""
        from services.commit_quality_coaching.cli import CommitQualityCoachingCLI
        return CommitQualityCoachingCLI()
    
    @pytest.mark.asyncio
    async def test_cli_initialization(self, cli):
        """Test CLI initialization."""
        assert cli is not None
        assert hasattr(cli, 'base_url')
        assert hasattr(cli, 'client')
    
    @pytest.mark.asyncio
    async def test_coach_commit_success(self, cli):
        """Test successful commit coaching via CLI."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "test-session",
            "commit_hash": "abc123",
            "quality_score": 85,
            "suggestions": ["Write better commit messages"]
        }
        
        cli.client.post = AsyncMock(return_value=mock_response)
        
        # Act
        result = await cli.coach_commit("abc123")
        
        # Assert
        assert result["session_id"] == "test-session"
        assert result["commit_hash"] == "abc123"
        assert result["quality_score"] == 85
        cli.client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_coach_commit_api_error(self, cli):
        """Test CLI handling of API errors."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        
        cli.client.post = AsyncMock(return_value=mock_response)
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await cli.coach_commit("abc123")
        
        assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_user_progress_success(self, cli):
        """Test successful user progress retrieval via CLI."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "test-user",
            "overall_progress": 75,
            "average_quality_score": 82,
            "total_commits": 25
        }
        
        cli.client.get = AsyncMock(return_value=mock_response)
        
        # Act
        result = await cli.get_user_progress("test-user")
        
        # Assert
        assert result["user_id"] == "test-user"
        assert result["overall_progress"] == 75
        assert result["average_quality_score"] == 82
        cli.client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_coaching_session_success(self, cli):
        """Test successful coaching session creation via CLI."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "test-session",
            "user_id": "test-user",
            "goals": "Improve commit quality",
            "progress": 0
        }
        
        cli.client.post = AsyncMock(return_value=mock_response)
        
        # Act
        result = await cli.create_coaching_session("test-user", "Improve commit quality")
        
        # Assert
        assert result["session_id"] == "test-session"
        assert result["user_id"] == "test-user"
        assert result["goals"] == "Improve commit quality"
        cli.client.post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
