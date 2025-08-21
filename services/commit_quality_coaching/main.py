"""
Commit Quality Coaching Service

This service provides AI-powered coaching and feedback on commit quality,
helping developers improve their coding practices and commit habits.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import get_settings
from shared.database import get_async_db
from shared.events import EventType, EventSource, EventFactory, EventSerializer
from shared.models import (
    Commit,
    Analysis,
    BehaviorPattern,
    UserBehavior,
    CommitQuality,
    AnalysisStatus,
    BehaviorPatternType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Commit Quality Coaching Service",
    description="AI-powered coaching and feedback for commit quality improvement",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client: Optional[redis.Redis] = None


# Pydantic models for coaching
class CoachingFeedback(BaseModel):
    """Model for coaching feedback."""

    commit_id: str = Field(..., description="ID of the commit being coached")
    user_id: str = Field(..., description="ID of the user receiving coaching")
    quality_score: float = Field(..., ge=0, le=10, description="AI-assessed quality score")
    quality_level: CommitQuality = Field(..., description="Quality level assessment")
    strengths: List[str] = Field(default_factory=list, description="Commit strengths")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas to improve")
    specific_recommendations: List[str] = Field(
        default_factory=list, description="Specific recommendations"
    )
    coaching_tips: List[str] = Field(default_factory=list, description="Coaching tips")
    next_steps: List[str] = Field(default_factory=list, description="Next steps for improvement")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoachingSession(BaseModel):
    """Model for a coaching session."""

    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique session ID")
    user_id: str = Field(..., description="ID of the user in the session")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = Field(None, description="Session end time")
    commits_reviewed: List[str] = Field(
        default_factory=list, description="Commits reviewed in session"
    )
    insights_generated: List[str] = Field(default_factory=list, description="Insights generated")
    goals_set: List[str] = Field(default_factory=list, description="Goals set during session")
    progress_metrics: Dict[str, Any] = Field(default_factory=dict, description="Progress metrics")


class UserProgress(BaseModel):
    """Model for user progress tracking."""

    user_id: str = Field(..., description="ID of the user")
    period_start: datetime = Field(..., description="Start of tracking period")
    period_end: datetime = Field(..., description="End of tracking period")
    total_commits: int = Field(0, description="Total commits in period")
    average_quality_score: float = Field(0.0, description="Average quality score")
    quality_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Quality level distribution"
    )
    improvement_areas: List[str] = Field(
        default_factory=list, description="Areas showing improvement"
    )
    regression_areas: List[str] = Field(
        default_factory=list, description="Areas showing regression"
    )
    consistency_score: float = Field(0.0, description="Consistency in commit quality")
    trend_analysis: Dict[str, Any] = Field(default_factory=dict, description="Trend analysis")


class CoachingRequest(BaseModel):
    """Model for coaching request."""

    commit_id: str = Field(..., description="ID of the commit to coach")
    user_id: str = Field(..., description="ID of the user requesting coaching")
    include_historical_context: bool = Field(True, description="Include historical context")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


class CoachingResponse(BaseModel):
    """Model for coaching response."""

    feedback: CoachingFeedback
    session: Optional[CoachingSession] = None
    progress: Optional[UserProgress] = None
    recommendations: List[str] = Field(default_factory=list, description="General recommendations")


class CommitQualityCoachingService:
    """Service for providing AI-powered commit quality coaching."""

    def __init__(self):
        self.redis_client = None
        self.event_factory = EventFactory()
        self.event_serializer = EventSerializer()

    async def initialize(self):
        """Initialize the service."""
        self.redis_client = redis.from_url(settings.redis.url)
        logger.info("Commit Quality Coaching Service initialized")

    async def get_commit_data(self, commit_id: str, db: AsyncSession) -> Optional[Commit]:
        """Get commit data from database."""
        try:
            # This would typically use a repository pattern
            # For now, we'll simulate getting commit data
            # In a real implementation, you'd query the database
            return None
        except Exception as e:
            logger.error(f"Error getting commit data: {e}")
            return None

    async def analyze_commit_quality(self, commit: Commit) -> Dict[str, Any]:
        """Analyze commit quality using AI."""
        try:
            # Simulate AI analysis
            # In a real implementation, this would call Ollama or another AI service

            # Analyze commit message quality
            message_score = self._analyze_message_quality(commit.message)

            # Analyze file changes
            file_score = self._analyze_file_changes(commit.changed_files)

            # Analyze commit size
            size_score = self._analyze_commit_size(commit.changed_files)

            # Calculate overall quality score
            overall_score = (message_score + file_score + size_score) / 3

            # Determine quality level
            if overall_score >= 9.0:
                quality_level = CommitQuality.EXCELLENT
            elif overall_score >= 7.0:
                quality_level = CommitQuality.GOOD
            elif overall_score >= 5.0:
                quality_level = CommitQuality.AVERAGE
            elif overall_score >= 3.0:
                quality_level = CommitQuality.POOR
            else:
                quality_level = CommitQuality.CRITICAL

            return {
                "quality_score": overall_score,
                "quality_level": quality_level,
                "message_score": message_score,
                "file_score": file_score,
                "size_score": size_score,
                "analysis_details": {
                    "message_analysis": self._get_message_analysis(commit.message),
                    "file_analysis": self._get_file_analysis(commit.changed_files),
                    "size_analysis": self._get_size_analysis(commit.changed_files),
                },
            }
        except Exception as e:
            logger.error(f"Error analyzing commit quality: {e}")
            return {"quality_score": 5.0, "quality_level": CommitQuality.AVERAGE, "error": str(e)}

    def _analyze_message_quality(self, message: str) -> float:
        """Analyze commit message quality."""
        if not message:
            return 1.0

        score = 5.0  # Base score

        # Check message length
        if len(message) < 10:
            score -= 2.0
        elif len(message) > 50 and len(message) < 200:
            score += 1.0
        elif len(message) > 200:
            score -= 1.0

        # Check for imperative mood
        if message.lower().startswith(("add", "fix", "update", "remove", "refactor", "improve")):
            score += 1.0

        # Check for descriptive content
        descriptive_words = [
            "because",
            "when",
            "where",
            "why",
            "how",
            "fixes",
            "closes",
            "implements",
        ]
        if any(word in message.lower() for word in descriptive_words):
            score += 1.0

        # Check for issue references
        if any(char in message for char in ["#", "GH-", "JIRA-"]):
            score += 0.5

        return min(10.0, max(1.0, score))

    def _analyze_file_changes(self, changed_files: List[str]) -> float:
        """Analyze file changes quality."""
        if not changed_files:
            return 5.0

        score = 5.0  # Base score

        # Check for too many files (potential for large commits)
        if len(changed_files) > 20:
            score -= 2.0
        elif len(changed_files) <= 5:
            score += 1.0

        # Check for mixed file types (potential for mixed concerns)
        file_extensions = [f.split(".")[-1] if "." in f else "no_ext" for f in changed_files]
        unique_extensions = len(set(file_extensions))

        if unique_extensions > 5:
            score -= 1.0
        elif unique_extensions <= 2:
            score += 0.5

        return min(10.0, max(1.0, score))

    def _analyze_commit_size(self, changed_files: List[str]) -> float:
        """Analyze commit size quality."""
        # This would typically analyze actual file sizes
        # For now, we'll use file count as a proxy
        file_count = len(changed_files)

        if file_count == 0:
            return 5.0
        elif file_count <= 3:
            return 9.0
        elif file_count <= 10:
            return 7.0
        elif file_count <= 20:
            return 5.0
        else:
            return 3.0

    def _get_message_analysis(self, message: str) -> Dict[str, Any]:
        """Get detailed message analysis."""
        return {
            "length": len(message) if message else 0,
            "has_imperative": (
                message.lower().startswith(
                    ("add", "fix", "update", "remove", "refactor", "improve")
                )
                if message
                else False
            ),
            "has_issue_reference": (
                any(char in message for char in ["#", "GH-", "JIRA-"]) if message else False
            ),
            "is_descriptive": (
                any(
                    word in message.lower()
                    for word in [
                        "because",
                        "when",
                        "where",
                        "why",
                        "how",
                        "fixes",
                        "closes",
                        "implements",
                    ]
                )
                if message
                else False
            ),
        }

    def _get_file_analysis(self, changed_files: List[str]) -> Dict[str, Any]:
        """Get detailed file analysis."""
        file_extensions = [f.split(".")[-1] if "." in f else "no_ext" for f in changed_files]
        return {
            "total_files": len(changed_files),
            "unique_extensions": len(set(file_extensions)),
            "extension_distribution": {
                ext: file_extensions.count(ext) for ext in set(file_extensions)
            },
        }

    def _get_size_analysis(self, changed_files: List[str]) -> Dict[str, Any]:
        """Get detailed size analysis."""
        return {
            "file_count": len(changed_files),
            "size_category": (
                "small"
                if len(changed_files) <= 3
                else "medium" if len(changed_files) <= 10 else "large"
            ),
        }

    async def generate_coaching_feedback(
        self, commit: Commit, analysis: Dict[str, Any]
    ) -> CoachingFeedback:
        """Generate coaching feedback based on commit analysis."""
        quality_score = analysis.get("quality_score", 5.0)
        quality_level = analysis.get("quality_level", CommitQuality.AVERAGE)

        # Generate strengths
        strengths = []
        if quality_score >= 7.0:
            strengths.append("Good commit message structure")
        if analysis.get("message_score", 0) >= 7.0:
            strengths.append("Clear and descriptive commit message")
        if analysis.get("file_score", 0) >= 7.0:
            strengths.append("Well-organized file changes")
        if analysis.get("size_score", 0) >= 7.0:
            strengths.append("Appropriate commit size")

        # Generate areas for improvement
        areas_for_improvement = []
        if analysis.get("message_score", 0) < 6.0:
            areas_for_improvement.append("Commit message clarity")
        if analysis.get("file_score", 0) < 6.0:
            areas_for_improvement.append("File organization")
        if analysis.get("size_score", 0) < 6.0:
            areas_for_improvement.append("Commit size management")

        # Generate specific recommendations
        recommendations = []
        if analysis.get("message_score", 0) < 6.0:
            recommendations.extend(
                [
                    "Use imperative mood in commit messages (e.g., 'Add feature' not 'Added feature')",
                    "Include context about why the change was made",
                    "Reference related issues when applicable",
                ]
            )

        if analysis.get("file_score", 0) < 6.0:
            recommendations.extend(
                [
                    "Group related file changes together",
                    "Avoid mixing different types of changes in one commit",
                    "Consider splitting large commits into smaller, focused ones",
                ]
            )

        if analysis.get("size_score", 0) < 6.0:
            recommendations.extend(
                [
                    "Keep commits focused on a single concern",
                    "Aim for commits that can be reviewed in 5-10 minutes",
                    "Use feature branches for larger changes",
                ]
            )

        # Generate coaching tips
        coaching_tips = [
            "Review your commit message before pushing",
            "Think about what someone else would need to understand your change",
            "Consider the 'why' behind your changes, not just the 'what'",
            "Use atomic commits that can be easily reverted if needed",
        ]

        # Generate next steps
        next_steps = [
            "Practice writing clear commit messages for your next 5 commits",
            "Review your recent commit history to identify patterns",
            "Set up commit message templates for consistency",
            "Consider using conventional commit format",
        ]

        return CoachingFeedback(
            commit_id=commit.id,
            user_id=commit.author,  # Assuming author is user_id for now
            quality_score=quality_score,
            quality_level=quality_level,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            specific_recommendations=recommendations,
            coaching_tips=coaching_tips,
            next_steps=next_steps,
        )

    async def get_user_progress(
        self, user_id: str, days: int = 30, db: AsyncSession = None
    ) -> UserProgress:
        """Get user progress over a specified period."""
        try:
            # This would typically query the database for user's commits
            # For now, we'll return a mock progress object
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            return UserProgress(
                user_id=user_id,
                period_start=start_date,
                period_end=end_date,
                total_commits=25,  # Mock data
                average_quality_score=7.2,  # Mock data
                quality_distribution={
                    "EXCELLENT": 5,
                    "GOOD": 12,
                    "AVERAGE": 6,
                    "POOR": 2,
                    "CRITICAL": 0,
                },
                improvement_areas=["Commit message clarity", "File organization"],
                regression_areas=["Commit size consistency"],
                consistency_score=6.8,
                trend_analysis={
                    "trend": "improving",
                    "weekly_average": [6.5, 6.8, 7.0, 7.2],
                    "key_insights": ["Message quality improving", "Size consistency needs work"],
                },
            )
        except Exception as e:
            logger.error(f"Error getting user progress: {e}")
            return None

    async def publish_coaching_event(self, feedback: CoachingFeedback):
        """Publish coaching event to Redis."""
        try:
            event = self.event_factory.create_event(
                event_type=EventType.COACHING_FEEDBACK_GENERATED,
                source=EventSource.COMMIT_QUALITY_COACHING,
                data={
                    "feedback_id": str(uuid4()),
                    "commit_id": feedback.commit_id,
                    "user_id": feedback.user_id,
                    "quality_score": feedback.quality_score,
                    "quality_level": feedback.quality_level.value,
                    "timestamp": feedback.created_at.isoformat(),
                },
            )

            serialized_event = self.event_serializer.serialize(event)
            await self.redis_client.publish("coaching_events", serialized_event)

            logger.info(f"Published coaching event for commit {feedback.commit_id}")
        except Exception as e:
            logger.error(f"Error publishing coaching event: {e}")


# Initialize service
coaching_service = CommitQualityCoachingService()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await coaching_service.initialize()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "commit-quality-coaching",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.post("/coach/commit", response_model=CoachingResponse)
async def coach_commit(
    request: CoachingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db),
):
    """Provide coaching feedback for a specific commit."""
    try:
        # Get commit data
        commit = await coaching_service.get_commit_data(request.commit_id, db)
        if not commit:
            raise HTTPException(status_code=404, detail="Commit not found")

        # Analyze commit quality
        analysis = await coaching_service.analyze_commit_quality(commit)

        # Generate coaching feedback
        feedback = await coaching_service.generate_coaching_feedback(commit, analysis)

        # Get user progress if requested
        progress = None
        if request.include_historical_context:
            progress = await coaching_service.get_user_progress(request.user_id, db=db)

        # Publish event in background
        background_tasks.add_task(coaching_service.publish_coaching_event, feedback)

        return CoachingResponse(
            feedback=feedback,
            progress=progress,
            recommendations=[
                "Focus on writing clear, descriptive commit messages",
                "Keep commits focused on single concerns",
                "Use conventional commit format for consistency",
                "Review your commits before pushing",
            ],
        )
    except Exception as e:
        logger.error(f"Error coaching commit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coach/progress/{user_id}", response_model=UserProgress)
async def get_user_progress(user_id: str, days: int = 30, db: AsyncSession = Depends(get_async_db)):
    """Get user progress over time."""
    try:
        progress = await coaching_service.get_user_progress(user_id, days, db)
        if not progress:
            raise HTTPException(status_code=404, detail="User progress not found")
        return progress
    except Exception as e:
        logger.error(f"Error getting user progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coach/session/start", response_model=CoachingSession)
async def start_coaching_session(user_id: str):
    """Start a new coaching session."""
    try:
        session = CoachingSession(user_id=user_id)
        return session
    except Exception as e:
        logger.error(f"Error starting coaching session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coach/session/{session_id}/end", response_model=CoachingSession)
async def end_coaching_session(session_id: str):
    """End a coaching session."""
    try:
        # This would typically update the session in the database
        # For now, we'll return a mock session
        session = CoachingSession(
            session_id=session_id,
            user_id="user123",
            end_time=datetime.now(timezone.utc),
            commits_reviewed=["commit1", "commit2"],
            insights_generated=["Improved message clarity", "Better file organization"],
            goals_set=["Write clearer commit messages", "Keep commits smaller"],
        )
        return session
    except Exception as e:
        logger.error(f"Error ending coaching session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coach/insights/{user_id}")
async def get_user_insights(user_id: str, days: int = 30):
    """Get AI-generated insights for a user."""
    try:
        # This would typically analyze user's commit history and generate insights
        insights = {
            "user_id": user_id,
            "period_days": days,
            "key_insights": [
                "Your commit messages have improved 15% over the last 30 days",
                "You tend to make larger commits on Fridays",
                "Your best commits are typically made in the morning",
                "Consider breaking down commits with more than 10 files",
            ],
            "recommendations": [
                "Set aside time for commit message review",
                "Use commit templates for consistency",
                "Consider pair programming for complex changes",
                "Review your commits weekly for patterns",
            ],
            "trends": {
                "message_quality": "improving",
                "commit_size": "stable",
                "consistency": "improving",
            },
        }
        return insights
    except Exception as e:
        logger.error(f"Error getting user insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.service_ports.commit_quality_coaching,
        reload=True,
        log_level="info",
    )
