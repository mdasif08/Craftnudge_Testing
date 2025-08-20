"""
Enterprise-grade data models for CraftNudge microservices.

This module provides comprehensive data models with:
- Type-safe model definitions
- Comprehensive validation rules
- Business logic encapsulation
- Database relationship mappings
- API serialization support
- Audit and tracking capabilities
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, validator, root_validator, computed_field
from sqlalchemy import Column, String, DateTime, Float, JSON, Text, Integer, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

Base = declarative_base()


class CommitQuality(Enum):
    """Commit quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


class CommitType(Enum):
    """Types of commits."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TEST = "test"
    CHORE = "chore"
    HOTFIX = "hotfix"
    RELEASE = "release"


class AnalysisStatus(Enum):
    """Analysis processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BehaviorPattern(Enum):
    """Behavior pattern types."""
    COMMIT_FREQUENCY = "commit_frequency"
    COMMIT_SIZE = "commit_size"
    COMMIT_TIMING = "commit_timing"
    CODE_QUALITY = "code_quality"
    TESTING_PATTERNS = "testing_patterns"
    DOCUMENTATION_PATTERNS = "documentation_patterns"
    REVIEW_PATTERNS = "review_patterns"
    COLLABORATION_PATTERNS = "collaboration_patterns"


# Pydantic Models for API
class CommitBase(BaseModel):
    """Base commit model with common fields."""
    
    hash: str = Field(..., min_length=7, max_length=40, description="Git commit hash")
    author: str = Field(..., min_length=1, max_length=255, description="Commit author")
    message: str = Field(..., min_length=1, max_length=1000, description="Commit message")
    timestamp: datetime = Field(..., description="Commit timestamp")
    changed_files: List[str] = Field(default_factory=list, description="Changed files")
    repository: str = Field(..., min_length=1, max_length=255, description="Repository name")
    branch: str = Field(..., min_length=1, max_length=255, description="Branch name")
    additions: int = Field(default=0, ge=0, description="Lines added")
    deletions: int = Field(default=0, ge=0, description="Lines deleted")
    total_changes: int = Field(default=0, ge=0, description="Total changes")
    
    @validator('hash')
    def validate_hash(cls, v):
        """Validate Git commit hash format."""
        if not v or len(v) < 7:
            raise ValueError('Commit hash must be at least 7 characters')
        return v
    
    @validator('message')
    def validate_message(cls, v):
        """Validate commit message."""
        if not v or not v.strip():
            raise ValueError('Commit message cannot be empty')
        return v.strip()
    
    @computed_field
    @property
    def change_ratio(self) -> float:
        """Calculate the ratio of additions to deletions."""
        if self.deletions == 0:
            return float('inf') if self.additions > 0 else 0.0
        return self.additions / self.deletions
    
    @computed_field
    @property
    def commit_type(self) -> CommitType:
        """Infer commit type from message."""
        message_lower = self.message.lower()
        
        if any(word in message_lower for word in ['fix', 'bug', 'issue']):
            return CommitType.BUGFIX
        elif any(word in message_lower for word in ['feat', 'feature', 'add']):
            return CommitType.FEATURE
        elif any(word in message_lower for word in ['refactor', 'refactoring']):
            return CommitType.REFACTOR
        elif any(word in message_lower for word in ['doc', 'docs', 'documentation']):
            return CommitType.DOCUMENTATION
        elif any(word in message_lower for word in ['test', 'spec']):
            return CommitType.TEST
        elif any(word in message_lower for word in ['chore', 'maintenance']):
            return CommitType.CHORE
        elif any(word in message_lower for word in ['hotfix', 'urgent']):
            return CommitType.HOTFIX
        elif any(word in message_lower for word in ['release', 'version']):
            return CommitType.RELEASE
        else:
            return CommitType.CHORE


class CommitCreate(CommitBase):
    """Model for creating a new commit."""
    pass


class CommitUpdate(BaseModel):
    """Model for updating a commit."""
    message: Optional[str] = Field(None, min_length=1, max_length=1000)
    changed_files: Optional[List[str]] = None
    additions: Optional[int] = Field(None, ge=0)
    deletions: Optional[int] = Field(None, ge=0)
    total_changes: Optional[int] = Field(None, ge=0)


class Commit(CommitBase):
    """Complete commit model with all fields."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique commit ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    quality_score: Optional[float] = Field(None, ge=0, le=10, description="AI-assessed quality score")
    quality_level: Optional[CommitQuality] = None
    analysis_status: AnalysisStatus = Field(default=AnalysisStatus.PENDING)
    
    class Config:
        from_attributes = True
        use_enum_values = True
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        """Validate quality score and set quality level."""
        if v is not None:
            if v >= 9.0:
                cls.quality_level = CommitQuality.EXCELLENT
            elif v >= 7.0:
                cls.quality_level = CommitQuality.GOOD
            elif v >= 5.0:
                cls.quality_level = CommitQuality.AVERAGE
            elif v >= 3.0:
                cls.quality_level = CommitQuality.POOR
            else:
                cls.quality_level = CommitQuality.CRITICAL
        return v


class AnalysisBase(BaseModel):
    """Base analysis model."""
    
    commit_id: str = Field(..., description="Commit ID being analyzed")
    quality_score: float = Field(..., ge=0, le=10, description="Quality score (0-10)")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    patterns_detected: List[str] = Field(default_factory=list, description="Detected patterns")
    behavioral_insights: Dict[str, Any] = Field(default_factory=dict, description="Behavioral insights")
    model_used: str = Field(default="llama2", description="AI model used")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Confidence score")
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        """Validate quality score."""
        if not 0 <= v <= 10:
            raise ValueError('Quality score must be between 0 and 10')
        return round(v, 2)
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Validate confidence score."""
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return round(v, 3)


class AnalysisCreate(AnalysisBase):
    """Model for creating a new analysis."""
    pass


class AnalysisUpdate(BaseModel):
    """Model for updating an analysis."""
    quality_score: Optional[float] = Field(None, ge=0, le=10)
    suggestions: Optional[List[str]] = None
    patterns_detected: Optional[List[str]] = None
    behavioral_insights: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)


class Analysis(AnalysisBase):
    """Complete analysis model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique analysis ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_duration: float = Field(default=0.0, ge=0, description="Analysis duration in seconds")
    status: AnalysisStatus = Field(default=AnalysisStatus.COMPLETED)
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True
        use_enum_values = True


class BehaviorPatternBase(BaseModel):
    """Base behavior pattern model."""
    
    user_id: str = Field(..., description="User ID")
    pattern_type: BehaviorPattern = Field(..., description="Pattern type")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern data")
    confidence: float = Field(..., ge=0, le=1, description="Pattern confidence")
    impact_score: float = Field(default=0.0, ge=0, le=10, description="Impact score")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return round(v, 3)
    
    @validator('impact_score')
    def validate_impact_score(cls, v):
        """Validate impact score."""
        if not 0 <= v <= 10:
            raise ValueError('Impact score must be between 0 and 10')
        return round(v, 2)


class BehaviorPatternCreate(BehaviorPatternBase):
    """Model for creating a new behavior pattern."""
    pass


class BehaviorPatternUpdate(BaseModel):
    """Model for updating a behavior pattern."""
    pattern_data: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    impact_score: Optional[float] = Field(None, ge=0, le=10)
    recommendations: Optional[List[str]] = None


class BehaviorPattern(BehaviorPatternBase):
    """Complete behavior pattern model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique pattern ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = Field(default=True, description="Whether the pattern is active")
    
    class Config:
        from_attributes = True
        use_enum_values = True


class UserBehavior(BaseModel):
    """User behavior summary model."""
    
    user_id: str = Field(..., description="User ID")
    commit_frequency: float = Field(default=0.0, ge=0, description="Average commits per day")
    average_commit_size: int = Field(default=0, ge=0, description="Average commit size in lines")
    common_patterns: List[str] = Field(default_factory=list, description="Common patterns")
    improvement_areas: List[str] = Field(default_factory=list, description="Areas for improvement")
    quality_trend: List[float] = Field(default_factory=list, description="Quality score trend")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @computed_field
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score from trend."""
        if not self.quality_trend:
            return 0.0
        return round(sum(self.quality_trend) / len(self.quality_trend), 2)
    
    @computed_field
    @property
    def quality_trend_direction(self) -> str:
        """Determine quality trend direction."""
        if len(self.quality_trend) < 2:
            return "stable"
        
        recent_avg = sum(self.quality_trend[-3:]) / min(3, len(self.quality_trend))
        older_avg = sum(self.quality_trend[:-3]) / max(1, len(self.quality_trend) - 3)
        
        if recent_avg > older_avg + 0.5:
            return "improving"
        elif recent_avg < older_avg - 0.5:
            return "declining"
        else:
            return "stable"


# SQLAlchemy Models for Database
class CommitModel(Base):
    """SQLAlchemy model for commits."""
    
    __tablename__ = "commits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hash = Column(String(40), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    changed_files = Column(JSONB, nullable=False, default=list)
    repository = Column(String(255), nullable=False, index=True)
    branch = Column(String(255), nullable=False, index=True)
    additions = Column(Integer, nullable=False, default=0)
    deletions = Column(Integer, nullable=False, default=0)
    total_changes = Column(Integer, nullable=False, default=0)
    quality_score = Column(Float, nullable=True)
    quality_level = Column(String(20), nullable=True)
    analysis_status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Relationships
    analyses: Mapped[List["AnalysisModel"]] = relationship("AnalysisModel", back_populates="commit")
    
    # Indexes
    __table_args__ = (
        Index('idx_commits_author_timestamp', 'author', 'timestamp'),
        Index('idx_commits_repository_branch', 'repository', 'branch'),
        Index('idx_commits_quality_score', 'quality_score'),
        Index('idx_commits_analysis_status', 'analysis_status'),
    )


class AnalysisModel(Base):
    """SQLAlchemy model for analyses."""
    
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    commit_id = Column(UUID(as_uuid=True), ForeignKey("commits.id"), nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    suggestions = Column(JSONB, nullable=False, default=list)
    patterns_detected = Column(JSONB, nullable=False, default=list)
    behavioral_insights = Column(JSONB, nullable=False, default=dict)
    analysis_duration = Column(Float, nullable=False, default=0.0)
    model_used = Column(String(100), nullable=False, default="llama2")
    confidence_score = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False, default="completed")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Relationships
    commit: Mapped["CommitModel"] = relationship("CommitModel", back_populates="analyses")
    
    # Indexes
    __table_args__ = (
        Index('idx_analyses_commit_id', 'commit_id'),
        Index('idx_analyses_quality_score', 'quality_score'),
        Index('idx_analyses_status', 'status'),
        Index('idx_analyses_model_used', 'model_used'),
    )


class BehaviorPatternModel(Base):
    """SQLAlchemy model for behavior patterns."""
    
    __tablename__ = "behavior_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    pattern_data = Column(JSONB, nullable=False)
    confidence = Column(Float, nullable=False)
    impact_score = Column(Float, nullable=False, default=0.0)
    recommendations = Column(JSONB, nullable=False, default=list)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_behavior_patterns_user_id', 'user_id'),
        Index('idx_behavior_patterns_type', 'pattern_type'),
        Index('idx_behavior_patterns_active', 'is_active'),
        Index('idx_behavior_patterns_confidence', 'confidence'),
    )


class UserBehaviorModel(Base):
    """SQLAlchemy model for user behavior summaries."""
    
    __tablename__ = "user_behaviors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    commit_frequency = Column(Float, nullable=False, default=0.0)
    average_commit_size = Column(Integer, nullable=False, default=0)
    common_patterns = Column(JSONB, nullable=False, default=list)
    improvement_areas = Column(JSONB, nullable=False, default=list)
    quality_trend = Column(JSONB, nullable=False, default=list)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_user_behaviors_user_id', 'user_id'),
        Index('idx_user_behaviors_frequency', 'commit_frequency'),
        Index('idx_user_behaviors_commit_size', 'average_commit_size'),
    )


# Model conversion utilities
class ModelConverter:
    """Utility class for converting between Pydantic and SQLAlchemy models."""
    
    @staticmethod
    def commit_to_model(commit: Commit) -> CommitModel:
        """Convert Pydantic Commit to SQLAlchemy CommitModel."""
        return CommitModel(
            id=uuid.UUID(commit.id),
            hash=commit.hash,
            author=commit.author,
            message=commit.message,
            timestamp=commit.timestamp,
            changed_files=commit.changed_files,
            repository=commit.repository,
            branch=commit.branch,
            additions=commit.additions,
            deletions=commit.deletions,
            total_changes=commit.total_changes,
            quality_score=commit.quality_score,
            quality_level=commit.quality_level.value if commit.quality_level else None,
            analysis_status=commit.analysis_status.value,
            created_at=commit.created_at,
            updated_at=commit.updated_at
        )
    
    @staticmethod
    def model_to_commit(model: CommitModel) -> Commit:
        """Convert SQLAlchemy CommitModel to Pydantic Commit."""
        return Commit(
            id=str(model.id),
            hash=model.hash,
            author=model.author,
            message=model.message,
            timestamp=model.timestamp,
            changed_files=model.changed_files,
            repository=model.repository,
            branch=model.branch,
            additions=model.additions,
            deletions=model.deletions,
            total_changes=model.total_changes,
            quality_score=model.quality_score,
            quality_level=CommitQuality(model.quality_level) if model.quality_level else None,
            analysis_status=AnalysisStatus(model.analysis_status),
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    @staticmethod
    def analysis_to_model(analysis: Analysis) -> AnalysisModel:
        """Convert Pydantic Analysis to SQLAlchemy AnalysisModel."""
        return AnalysisModel(
            id=uuid.UUID(analysis.id),
            commit_id=uuid.UUID(analysis.commit_id),
            quality_score=analysis.quality_score,
            suggestions=analysis.suggestions,
            patterns_detected=analysis.patterns_detected,
            behavioral_insights=analysis.behavioral_insights,
            analysis_duration=analysis.analysis_duration,
            model_used=analysis.model_used,
            confidence_score=analysis.confidence_score,
            status=analysis.status.value,
            error_message=analysis.error_message,
            created_at=analysis.created_at,
            updated_at=analysis.updated_at
        )
    
    @staticmethod
    def model_to_analysis(model: AnalysisModel) -> Analysis:
        """Convert SQLAlchemy AnalysisModel to Pydantic Analysis."""
        return Analysis(
            id=str(model.id),
            commit_id=str(model.commit_id),
            quality_score=model.quality_score,
            suggestions=model.suggestions,
            patterns_detected=model.patterns_detected,
            behavioral_insights=model.behavioral_insights,
            analysis_duration=model.analysis_duration,
            model_used=model.model_used,
            confidence_score=model.confidence_score,
            status=AnalysisStatus(model.status),
            error_message=model.error_message,
            created_at=model.created_at,
            updated_at=model.updated_at
        )


# Model validation utilities
class ModelValidator:
    """Utility class for model validation."""
    
    @staticmethod
    def validate_commit_data(data: Dict[str, Any]) -> List[str]:
        """Validate commit data and return list of errors."""
        errors = []
        
        required_fields = ['hash', 'author', 'message', 'timestamp', 'repository', 'branch']
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")
        
        if 'hash' in data and len(data['hash']) < 7:
            errors.append("Commit hash must be at least 7 characters")
        
        if 'message' in data and not data['message'].strip():
            errors.append("Commit message cannot be empty")
        
        if 'quality_score' in data and data['quality_score'] is not None:
            if not 0 <= data['quality_score'] <= 10:
                errors.append("Quality score must be between 0 and 10")
        
        return errors
    
    @staticmethod
    def validate_analysis_data(data: Dict[str, Any]) -> List[str]:
        """Validate analysis data and return list of errors."""
        errors = []
        
        required_fields = ['commit_id', 'quality_score']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if 'quality_score' in data:
            if not 0 <= data['quality_score'] <= 10:
                errors.append("Quality score must be between 0 and 10")
        
        if 'confidence_score' in data and data['confidence_score'] is not None:
            if not 0 <= data['confidence_score'] <= 1:
                errors.append("Confidence score must be between 0 and 1")
        
        return errors


# Export commonly used classes and functions
__all__ = [
    'CommitQuality', 'CommitType', 'AnalysisStatus', 'BehaviorPattern',
    'CommitBase', 'CommitCreate', 'CommitUpdate', 'Commit',
    'AnalysisBase', 'AnalysisCreate', 'AnalysisUpdate', 'Analysis',
    'BehaviorPatternBase', 'BehaviorPatternCreate', 'BehaviorPatternUpdate', 'BehaviorPattern',
    'UserBehavior',
    'CommitModel', 'AnalysisModel', 'BehaviorPatternModel', 'UserBehaviorModel',
    'ModelConverter', 'ModelValidator'
]
