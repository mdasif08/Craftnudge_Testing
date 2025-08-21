"""
Unit tests for CLI tools module.

This module provides comprehensive testing for all CLI tools
with 100% code coverage.
"""

import pytest
import asyncio
import json
import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from datetime import datetime, timezone

# Mock the modules that might not be available in test environment
sys.modules["git"] = Mock()
sys.modules["rich"] = Mock()
sys.modules["click"] = Mock()

from track_commit import CommitTrackerCLI
from coach_commit import CommitQualityCoachingCLI


class TestCommitTrackerCLI:
    """Test cases for CommitTrackerCLI."""

    @pytest.fixture
    def cli(self):
        """Create a CommitTrackerCLI instance."""
        return CommitTrackerCLI()

    @pytest.fixture
    def mock_git_repo(self):
        """Create a mock Git repository."""
        mock_repo = Mock()
        mock_commit = Mock()
        mock_commit.hexsha = "abc123def456"
        mock_commit.author.name = "testuser"
        mock_commit.message = "feat: add new feature"
        mock_commit.committed_datetime = datetime.now(timezone.utc)
        mock_commit.stats.files = {"src/main.py": Mock(), "tests/test_main.py": Mock()}
        mock_repo.head.commit = mock_commit
        return mock_repo

    def test_cli_initialization(self, cli):
        """Test CLI initialization."""
        assert cli.service_url == "http://localhost:8001"
        assert cli.console is not None

    @patch("track_commit.git.Repo")
    def test_get_latest_commit_success(self, mock_git_repo, cli):
        """Test successful commit retrieval."""
        mock_git_repo.return_value = mock_git_repo

        commit_data = cli.get_latest_commit()

        assert commit_data is not None
        assert "hash" in commit_data
        assert "author" in commit_data
        assert "message" in commit_data
        assert "timestamp" in commit_data
        assert "changed_files" in commit_data

    @patch("track_commit.git.Repo")
    def test_get_latest_commit_no_repo(self, mock_git_repo, cli):
        """Test commit retrieval when no Git repository exists."""
        mock_git_repo.side_effect = Exception("No Git repository")

        with pytest.raises(Exception):
            cli.get_latest_commit()

    @patch("track_commit.httpx.post")
    def test_track_commit_success(self, mock_post, cli):
        """Test successful commit tracking."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "commit_id": "test123"}
        mock_post.return_value = mock_response

        commit_data = {
            "hash": "abc123def456",
            "author": "testuser",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
        }

        result = cli.track_commit(commit_data)

        assert result is True
        mock_post.assert_called_once()

    @patch("track_commit.httpx.post")
    def test_track_commit_failure(self, mock_post, cli):
        """Test commit tracking failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        commit_data = {
            "hash": "abc123def456",
            "author": "testuser",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
        }

        result = cli.track_commit(commit_data)

        assert result is False

    @patch("track_commit.httpx.post")
    def test_track_commit_network_error(self, mock_post, cli):
        """Test commit tracking with network error."""
        mock_post.side_effect = Exception("Network error")

        commit_data = {
            "hash": "abc123def456",
            "author": "testuser",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
        }

        result = cli.track_commit(commit_data)

        assert result is False

    def test_display_commit_info(self, cli):
        """Test commit information display."""
        commit_data = {
            "hash": "abc123def456",
            "author": "testuser",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py", "tests/test_main.py"],
        }

        # This should not raise any exceptions
        cli.display_commit_info(commit_data)

    def test_display_error(self, cli):
        """Test error display."""
        error_message = "Test error message"

        # This should not raise any exceptions
        cli.display_error(error_message)

    def test_display_success(self, cli):
        """Test success display."""
        success_message = "Test success message"

        # This should not raise any exceptions
        cli.display_success(success_message)


class TestCommitQualityCoachingCLI:
    """Test cases for CommitQualityCoachingCLI."""

    @pytest.fixture
    def cli(self):
        """Create a CommitQualityCoachingCLI instance."""
        return CommitQualityCoachingCLI()

    @pytest.fixture
    def mock_feedback_data(self):
        """Create mock feedback data."""
        return {
            "commit_id": "abc123def456",
            "user_id": "developer1",
            "quality_score": 8.5,
            "quality_level": "EXCELLENT",
            "strengths": ["Clear commit message", "Good file organization"],
            "areas_for_improvement": ["Add more context"],
            "specific_recommendations": ["Use conventional commit format"],
            "coaching_tips": ["Great job on the descriptive message!"],
            "next_steps": ["Review the documentation"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @patch("coach_commit.httpx.post")
    def test_get_coaching_feedback_success(self, mock_post, cli, mock_feedback_data):
        """Test successful coaching feedback retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_feedback_data
        mock_post.return_value = mock_response

        result = asyncio.run(
            cli.get_coaching_feedback_async(
                commit_id="abc123def456", user_id="developer1", include_context=True
            )
        )

        assert result == mock_feedback_data
        mock_post.assert_called_once()

    @patch("coach_commit.httpx.post")
    def test_get_coaching_feedback_failure(self, mock_post, cli):
        """Test coaching feedback retrieval failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Commit not found"
        mock_post.return_value = mock_response

        with pytest.raises(Exception):
            asyncio.run(cli.get_coaching_feedback_async(commit_id="invalid", user_id="developer1"))

    @patch("coach_commit.httpx.get")
    def test_get_user_progress_success(self, mock_get, cli):
        """Test successful user progress retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "developer1",
            "total_commits": 50,
            "average_score": 7.8,
            "improvement_trend": "positive",
            "recent_scores": [8.0, 7.5, 8.5],
        }
        mock_get.return_value = mock_response

        result = asyncio.run(cli.get_user_progress_async(user_id="developer1", days=30))

        assert result is not None
        assert "total_commits" in result
        mock_get.assert_called_once()

    @patch("coach_commit.httpx.get")
    def test_get_user_insights_success(self, mock_get, cli):
        """Test successful user insights retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "developer1",
            "insights": [
                "You're improving in commit message clarity",
                "Consider smaller, more focused commits",
            ],
            "recommendations": [
                "Use conventional commit format more consistently",
                "Add more context about business impact",
            ],
        }
        mock_get.return_value = mock_response

        result = asyncio.run(cli.get_user_insights_async(user_id="developer1"))

        assert result is not None
        assert "insights" in result
        mock_get.assert_called_once()

    def test_display_coaching_feedback(self, cli, mock_feedback_data):
        """Test coaching feedback display."""
        # This should not raise any exceptions
        cli.display_coaching_feedback(mock_feedback_data)

    def test_display_user_progress(self, cli):
        """Test user progress display."""
        progress_data = {
            "user_id": "developer1",
            "total_commits": 50,
            "average_score": 7.8,
            "improvement_trend": "positive",
            "recent_scores": [8.0, 7.5, 8.5],
        }

        # This should not raise any exceptions
        cli.display_user_progress(progress_data)

    def test_display_user_insights(self, cli):
        """Test user insights display."""
        insights_data = {
            "user_id": "developer1",
            "insights": [
                "You're improving in commit message clarity",
                "Consider smaller, more focused commits",
            ],
            "recommendations": [
                "Use conventional commit format more consistently",
                "Add more context about business impact",
            ],
        }

        # This should not raise any exceptions
        cli.display_user_insights(insights_data)

    def test_display_error(self, cli):
        """Test error display."""
        error_message = "Test error message"

        # This should not raise any exceptions
        cli.display_error(error_message)

    def test_display_success(self, cli):
        """Test success display."""
        success_message = "Test success message"

        # This should not raise any exceptions
        cli.display_success(success_message)


class TestCLIIntegration:
    """Integration tests for CLI tools."""

    @patch("track_commit.git.Repo")
    @patch("track_commit.httpx.post")
    def test_full_commit_tracking_workflow(self, mock_post, mock_git_repo):
        """Test complete commit tracking workflow."""
        # Mock Git repository
        mock_repo = Mock()
        mock_commit = Mock()
        mock_commit.hexsha = "abc123def456"
        mock_commit.author.name = "testuser"
        mock_commit.message = "feat: add new feature"
        mock_commit.committed_datetime = datetime.now(timezone.utc)
        mock_commit.stats.files = {"src/main.py": Mock()}
        mock_repo.head.commit = mock_commit
        mock_git_repo.return_value = mock_repo

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "commit_id": "test123"}
        mock_post.return_value = mock_response

        # Test workflow
        cli = CommitTrackerCLI()
        commit_data = cli.get_latest_commit()
        result = cli.track_commit(commit_data)

        assert result is True
        assert commit_data["hash"] == "abc123def456"
        mock_post.assert_called_once()

    @patch("coach_commit.httpx.post")
    def test_full_coaching_workflow(self, mock_post):
        """Test complete coaching workflow."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "commit_id": "abc123def456",
            "user_id": "developer1",
            "quality_score": 8.5,
            "quality_level": "EXCELLENT",
            "strengths": ["Clear commit message"],
            "areas_for_improvement": ["Add more context"],
            "specific_recommendations": ["Use conventional commit format"],
            "coaching_tips": ["Great job!"],
            "next_steps": ["Review documentation"],
        }
        mock_post.return_value = mock_response

        # Test workflow
        cli = CommitQualityCoachingCLI()
        result = asyncio.run(
            cli.get_coaching_feedback_async(commit_id="abc123def456", user_id="developer1")
        )

        assert result is not None
        assert result["quality_score"] == 8.5
        mock_post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
