"""
Enterprise-grade Commit Tracker Service for CraftNudge.

This service provides comprehensive Git commit tracking with:
- Real-time commit detection and processing
- Event-driven architecture integration
- AI-powered commit analysis
- Comprehensive monitoring and observability
- RESTful API for commit management
- Local file storage for offline tracking
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from git import Repo, GitCommandError, InvalidGitRepositoryError
from git.diff import Diff
import httpx

from config.settings import settings
from shared.events import (
    EventType, EventSource, EventFactory, CommitEvent, CommitData,
    EventSerializer, EventValidator
)
from shared.models import (
    Commit, CommitCreate, CommitUpdate, CommitQuality, AnalysisStatus,
    ModelConverter, ModelValidator
)
from shared.database import DatabaseService, get_async_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.monitoring.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Commit Tracker Service",
    description="Enterprise Git commit tracking and analysis service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client for event bus
redis_client: Optional[redis.Redis] = None


class CommitTrackerService:
    """Core commit tracking service with business logic."""
    
    def __init__(self):
        self.redis_client = None
        self.db_service = DatabaseService()
        self.event_factory = EventFactory()
    
    async def initialize(self):
        """Initialize the service."""
        # Initialize Redis connection
        self.redis_client = redis.from_url(
            settings.redis.url,
            decode_responses=True,
            socket_connect_timeout=settings.redis.socket_connect_timeout,
            socket_timeout=settings.redis.socket_timeout,
            retry_on_timeout=settings.redis.retry_on_timeout
        )
        
        # Test Redis connection
        await self.redis_client.ping()
        logger.info("Commit tracker service initialized successfully")
    
    async def close(self):
        """Close service connections."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Commit tracker service connections closed")
    
    def extract_commit_data(self, commit, repo_path: str) -> Dict[str, Any]:
        """Extract comprehensive commit data from Git commit object."""
        try:
            # Get commit statistics
            stats = commit.stats
            additions = stats.total.get('insertions', 0)
            deletions = stats.total.get('deletions', 0)
            total_changes = additions + deletions
            
            # Get changed files
            changed_files = []
            if commit.parents:
                diff = commit.diff(commit.parents[0])
                changed_files = [item.a_path for item in diff if item.a_path]
            else:
                # Initial commit
                changed_files = [item.name for item in commit.tree.traverse()]
            
            # Get repository information
            repo = Repo(repo_path)
            repository_name = os.path.basename(repo_path)
            branch_name = repo.active_branch.name if not repo.head.is_detached else "detached"
            
            return {
                "hash": commit.hexsha,
                "author": str(commit.author),
                "message": commit.message.strip(),
                "timestamp": datetime.fromtimestamp(commit.committed_date, tz=timezone.utc),
                "changed_files": changed_files,
                "repository": repository_name,
                "branch": branch_name,
                "additions": additions,
                "deletions": deletions,
                "total_changes": total_changes
            }
        except Exception as e:
            logger.error(f"Error extracting commit data: {e}")
            raise
    
    async def track_commit(self, repo_path: str, commit_hash: Optional[str] = None) -> Commit:
        """Track a specific commit or the latest commit."""
        try:
            # Validate repository path
            if not os.path.exists(repo_path):
                raise ValueError(f"Repository path does not exist: {repo_path}")
            
            # Open Git repository
            repo = Repo(repo_path)
            
            # Get commit
            if commit_hash:
                commit = repo.commit(commit_hash)
            else:
                commit = repo.head.commit
            
            # Extract commit data
            commit_data = self.extract_commit_data(commit, repo_path)
            
            # Check if commit already exists
            existing_commit = await self.db_service.get_commit_by_hash(commit_data['hash'])
            if existing_commit:
                logger.info(f"Commit {commit_data['hash']} already tracked")
                return existing_commit
            
            # Store commit in database
            commit_model = await self.db_service.store_commit(commit_data)
            
            # Save to local file
            await self.save_commit_locally(commit_model)
            
            # Publish commit event
            await self.publish_commit_event(commit_model)
            
            logger.info(f"Successfully tracked commit {commit_data['hash']}")
            return commit_model
            
        except InvalidGitRepositoryError:
            raise ValueError(f"Invalid Git repository: {repo_path}")
        except GitCommandError as e:
            raise ValueError(f"Git command error: {e}")
        except Exception as e:
            logger.error(f"Error tracking commit: {e}")
            raise
    
    async def track_recent_commits(
        self, 
        repo_path: str, 
        days: int = 7, 
        limit: int = 100
    ) -> List[Commit]:
        """Track recent commits within specified time range."""
        try:
            repo = Repo(repo_path)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            tracked_commits = []
            commit_count = 0
            
            for commit in repo.iter_commits('--all'):
                if commit_count >= limit:
                    break
                
                commit_date = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
                if commit_date < cutoff_date:
                    break
                
                try:
                    commit_model = await self.track_commit(repo_path, commit.hexsha)
                    tracked_commits.append(commit_model)
                    commit_count += 1
                except Exception as e:
                    logger.warning(f"Failed to track commit {commit.hexsha}: {e}")
                    continue
            
            logger.info(f"Tracked {len(tracked_commits)} recent commits")
            return tracked_commits
            
        except Exception as e:
            logger.error(f"Error tracking recent commits: {e}")
            raise
    
    async def save_commit_locally(self, commit: Commit):
        """Save commit data to local JSONL file."""
        try:
            # Ensure data directory exists
            data_dir = Path(settings.file.data_dir) / "behaviors"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Append to JSONL file
            commits_file = data_dir / "commits.jsonl"
            with open(commits_file, "a", encoding="utf-8") as f:
                f.write(commit.json() + "\n")
            
            logger.debug(f"Saved commit {commit.hash} to local file")
            
        except Exception as e:
            logger.error(f"Error saving commit locally: {e}")
            # Don't raise - local storage is not critical
    
    async def publish_commit_event(self, commit: Commit):
        """Publish commit event to Redis event bus."""
        try:
            # Create commit event
            commit_data = CommitData(
                hash=commit.hash,
                author=commit.author,
                message=commit.message,
                timestamp=commit.timestamp,
                changed_files=commit.changed_files,
                repository=commit.repository,
                branch=commit.branch,
                additions=commit.additions,
                deletions=commit.deletions,
                total_changes=commit.total_changes
            )
            
            event = self.event_factory.create_commit_event(
                commit_data,
                source=EventSource.COMMIT_TRACKER,
                correlation_id=commit.id
            )
            
            # Validate event
            if not EventValidator.is_valid(event):
                raise ValueError("Invalid commit event")
            
            # Publish to Redis
            await self.redis_client.publish(
                "commit_events",
                EventSerializer.serialize(event)
            )
            
            logger.info(f"Published commit event for {commit.hash}")
            
        except Exception as e:
            logger.error(f"Error publishing commit event: {e}")
            # Don't raise - event publishing is not critical for commit tracking
    
    async def get_commit_statistics(self, repo_path: str) -> Dict[str, Any]:
        """Get comprehensive commit statistics for a repository."""
        try:
            repo = Repo(repo_path)
            
            # Get all commits
            commits = list(repo.iter_commits('--all'))
            
            if not commits:
                return {
                    "total_commits": 0,
                    "total_additions": 0,
                    "total_deletions": 0,
                    "average_commit_size": 0,
                    "commit_frequency": 0,
                    "top_authors": [],
                    "most_changed_files": []
                }
            
            # Calculate statistics
            total_additions = 0
            total_deletions = 0
            authors = {}
            files = {}
            
            for commit in commits:
                stats = commit.stats
                additions = stats.total.get('insertions', 0)
                deletions = stats.total.get('deletions', 0)
                
                total_additions += additions
                total_deletions += deletions
                
                # Count authors
                author = str(commit.author)
                authors[author] = authors.get(author, 0) + 1
                
                # Count file changes
                if commit.parents:
                    diff = commit.diff(commit.parents[0])
                    for item in diff:
                        if item.a_path:
                            files[item.a_path] = files.get(item.a_path, 0) + 1
            
            # Calculate averages
            total_commits = len(commits)
            average_commit_size = (total_additions + total_deletions) / total_commits
            
            # Calculate commit frequency (commits per day)
            if len(commits) > 1:
                first_commit = commits[-1]
                last_commit = commits[0]
                first_date = datetime.fromtimestamp(first_commit.committed_date, tz=timezone.utc)
                last_date = datetime.fromtimestamp(last_commit.committed_date, tz=timezone.utc)
                days_diff = (last_date - first_date).days or 1
                commit_frequency = total_commits / days_diff
            else:
                commit_frequency = 0
            
            # Get top authors and files
            top_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10]
            most_changed_files = sorted(files.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_commits": total_commits,
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "average_commit_size": round(average_commit_size, 2),
                "commit_frequency": round(commit_frequency, 2),
                "top_authors": top_authors,
                "most_changed_files": most_changed_files
            }
            
        except Exception as e:
            logger.error(f"Error getting commit statistics: {e}")
            raise


# Service instance
commit_tracker_service = CommitTrackerService()


# Request/Response models
class TrackCommitRequest(BaseModel):
    """Request model for tracking a commit."""
    repository_path: str = Field(..., description="Path to Git repository")
    commit_hash: Optional[str] = Field(None, description="Specific commit hash to track")
    
    class Config:
        json_schema_extra = {
            "example": {
                "repository_path": "/path/to/repo",
                "commit_hash": "abc123def456"
            }
        }


class TrackRecentCommitsRequest(BaseModel):
    """Request model for tracking recent commits."""
    repository_path: str = Field(..., description="Path to Git repository")
    days: int = Field(default=7, ge=1, le=365, description="Number of days to look back")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of commits to track")
    
    class Config:
        json_schema_extra = {
            "example": {
                "repository_path": "/path/to/repo",
                "days": 7,
                "limit": 100
            }
        }


class CommitStatisticsResponse(BaseModel):
    """Response model for commit statistics."""
    repository_path: str
    statistics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await commit_tracker_service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await commit_tracker_service.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        await redis_client.ping()
        
        # Check database connection
        db_health = await get_database_health()
        
        return {
            "status": "healthy",
            "service": "commit_tracker",
            "timestamp": datetime.now(timezone.utc),
            "database": db_health,
            "redis": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "commit_tracker",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            }
        )


@app.post("/track-commit", response_model=Commit)
async def track_commit(
    request: TrackCommitRequest,
    background_tasks: BackgroundTasks
):
    """Track a specific commit or the latest commit."""
    try:
        commit = await commit_tracker_service.track_commit(
            request.repository_path,
            request.commit_hash
        )
        
        # Trigger AI analysis in background
        background_tasks.add_task(trigger_ai_analysis, commit.id)
        
        return commit
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking commit: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/track-recent-commits", response_model=List[Commit])
async def track_recent_commits(request: TrackRecentCommitsRequest):
    """Track recent commits within specified time range."""
    try:
        commits = await commit_tracker_service.track_recent_commits(
            request.repository_path,
            request.days,
            request.limit
        )
        
        return commits
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking recent commits: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/commits", response_model=List[Commit])
async def get_commits(
    skip: int = Query(0, ge=0, description="Number of commits to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of commits to return"),
    author: Optional[str] = Query(None, description="Filter by author"),
    repository: Optional[str] = Query(None, description="Filter by repository")
):
    """Get commits with optional filtering."""
    try:
        # This would be implemented with database queries
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error getting commits: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/commits/{commit_id}", response_model=Commit)
async def get_commit(commit_id: str):
    """Get a specific commit by ID."""
    try:
        # This would be implemented with database query
        # For now, return 404
        raise HTTPException(status_code=404, detail="Commit not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting commit: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/statistics/{repository_path:path}", response_model=CommitStatisticsResponse)
async def get_commit_statistics(repository_path: str):
    """Get commit statistics for a repository."""
    try:
        statistics = await commit_tracker_service.get_commit_statistics(repository_path)
        
        return CommitStatisticsResponse(
            repository_path=repository_path,
            statistics=statistics
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting commit statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    try:
        # This would include various metrics like:
        # - Commits tracked per hour/day
        # - Average processing time
        # - Error rates
        # - Repository coverage
        
        return {
            "commits_tracked_today": 0,
            "commits_tracked_total": 0,
            "average_processing_time_ms": 0,
            "error_rate_percent": 0,
            "active_repositories": 0,
            "last_commit_tracked": None
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Background tasks
async def trigger_ai_analysis(commit_id: str):
    """Trigger AI analysis for a commit."""
    try:
        # This would send a request to the AI analysis service
        # For now, just log the action
        logger.info(f"Triggering AI analysis for commit {commit_id}")
        
        # Example implementation:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         f"http://localhost:{settings.service.ai_analysis_port}/analyze-commit",
        #         json={"commit_id": commit_id}
        #     )
        
    except Exception as e:
        logger.error(f"Error triggering AI analysis: {e}")


# CLI interface
async def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Commit Tracker Service")
    parser.add_argument("--repo-path", required=True, help="Path to Git repository")
    parser.add_argument("--commit-hash", help="Specific commit hash to track")
    parser.add_argument("--recent", action="store_true", help="Track recent commits")
    parser.add_argument("--days", type=int, default=7, help="Days to look back for recent commits")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of commits to track")
    
    args = parser.parse_args()
    
    try:
        # Initialize service
        await commit_tracker_service.initialize()
        
        if args.recent:
            commits = await commit_tracker_service.track_recent_commits(
                args.repo_path, args.days, args.limit
            )
            print(f"Tracked {len(commits)} recent commits")
        else:
            commit = await commit_tracker_service.track_commit(
                args.repo_path, args.commit_hash
            )
            print(f"Tracked commit: {commit.hash}")
        
        # Cleanup
        await commit_tracker_service.close()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
