"""
Enterprise-grade event system for CraftNudge microservices.

This module provides a comprehensive event-driven architecture with:
- Type-safe event definitions
- Event validation and serialization
- Event routing and filtering
- Event replay and audit capabilities
- Event versioning and compatibility
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.generics import GenericModel


class EventType(Enum):
    """Event types for the CraftNudge system."""
    
    # Commit-related events
    COMMIT_CREATED = "commit.created"
    COMMIT_UPDATED = "commit.updated"
    COMMIT_DELETED = "commit.deleted"
    COMMIT_ANALYZED = "commit.analyzed"
    
    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    
    # User behavior events
    BEHAVIOR_PATTERN_DETECTED = "behavior.pattern_detected"
    BEHAVIOR_INSIGHT_GENERATED = "behavior.insight_generated"
    BEHAVIOR_RECOMMENDATION_CREATED = "behavior.recommendation_created"
    
    # System events
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"
    SERVICE_HEALTH_CHECK = "service.health_check"
    
    # Error events
    ERROR_OCCURRED = "error.occurred"
    ERROR_RESOLVED = "error.resolved"
    
    # GitHub integration events
    GITHUB_WEBHOOK_RECEIVED = "github.webhook_received"
    GITHUB_COMMIT_PUSHED = "github.commit_pushed"
    GITHUB_PULL_REQUEST_CREATED = "github.pull_request_created"
    
    # Database events
    DATA_STORED = "data.stored"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    
    # Notification events
    NOTIFICATION_SENT = "notification.sent"
    NOTIFICATION_DELIVERED = "notification.delivered"
    NOTIFICATION_FAILED = "notification.failed"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


class EventSource(Enum):
    """Event source systems."""
    COMMIT_TRACKER = "commit_tracker"
    AI_ANALYSIS = "ai_analysis"
    DATABASE = "database"
    FRONTEND = "frontend"
    GITHUB_WEBHOOK = "github_webhook"
    SYSTEM = "system"


@dataclass
class EventMetadata:
    """Event metadata for tracking and auditing."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: EventSource = EventSource.SYSTEM
    version: str = "1.0.0"
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'event_id': self.event_id,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'version': self.version,
            'priority': self.priority.value,
            'status': self.status.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'ttl': self.ttl
        }


class BaseEvent(BaseModel):
    """Base event class with common functionality."""
    
    metadata: EventMetadata = Field(default_factory=EventMetadata)
    event_type: EventType
    data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            EventMetadata: lambda v: v.to_dict()
        }
    
    @validator('data')
    def validate_data(cls, v):
        """Validate event data."""
        if not isinstance(v, dict):
            raise ValueError('Event data must be a dictionary')
        return v
    
    def to_json(self) -> str:
        """Serialize event to JSON."""
        return self.json()
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEvent':
        """Deserialize event from JSON."""
        data = json.loads(json_str)
        return cls(**data)
    
    def get_correlation_id(self) -> str:
        """Get correlation ID for event tracing."""
        return self.metadata.correlation_id or self.metadata.event_id
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for event tracing."""
        self.metadata.correlation_id = correlation_id
    
    def get_causation_id(self) -> Optional[str]:
        """Get causation ID for event tracing."""
        return self.metadata.causation_id
    
    def set_causation_id(self, causation_id: str):
        """Set causation ID for event tracing."""
        self.metadata.causation_id = causation_id


# Generic event type for type safety
T = TypeVar('T', bound=BaseModel)


class TypedEvent(BaseEvent, Generic[T]):
    """Generic typed event for type-safe event handling."""
    
    data: T
    
    @classmethod
    def create(cls, event_type: EventType, data: T, **kwargs) -> 'TypedEvent[T]':
        """Create a typed event."""
        return cls(event_type=event_type, data=data, **kwargs)


# Specific event data models
class CommitData(BaseModel):
    """Commit event data."""
    
    hash: str = Field(..., description="Git commit hash")
    author: str = Field(..., description="Commit author")
    message: str = Field(..., description="Commit message")
    timestamp: datetime = Field(..., description="Commit timestamp")
    changed_files: List[str] = Field(default_factory=list, description="Changed files")
    repository: str = Field(..., description="Repository name")
    branch: str = Field(..., description="Branch name")
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")
    total_changes: int = Field(default=0, description="Total changes")
    
    @validator('hash')
    def validate_hash(cls, v):
        """Validate Git commit hash."""
        if len(v) < 7:
            raise ValueError('Commit hash must be at least 7 characters')
        return v
    
    @validator('message')
    def validate_message(cls, v):
        """Validate commit message."""
        if not v.strip():
            raise ValueError('Commit message cannot be empty')
        return v.strip()


class AnalysisData(BaseModel):
    """Analysis event data."""
    
    commit_id: str = Field(..., description="Commit ID being analyzed")
    quality_score: float = Field(..., ge=0, le=10, description="Quality score (0-10)")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    patterns_detected: List[str] = Field(default_factory=list, description="Detected patterns")
    behavioral_insights: Dict[str, Any] = Field(default_factory=dict, description="Behavioral insights")
    analysis_duration: float = Field(default=0.0, description="Analysis duration in seconds")
    model_used: str = Field(default="llama2", description="AI model used")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Confidence score")
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        """Validate quality score."""
        if not 0 <= v <= 10:
            raise ValueError('Quality score must be between 0 and 10')
        return v


class ErrorData(BaseModel):
    """Error event data."""
    
    error_type: str = Field(..., description="Error type")
    error_message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace")
    service: str = Field(..., description="Service where error occurred")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    severity: str = Field(default="error", description="Error severity")
    
    @validator('severity')
    def validate_severity(cls, v):
        """Validate error severity."""
        valid_severities = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f'Severity must be one of: {valid_severities}')
        return v.lower()


class BehaviorData(BaseModel):
    """Behavior event data."""
    
    user_id: str = Field(..., description="User ID")
    pattern_type: str = Field(..., description="Pattern type")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern data")
    confidence: float = Field(..., ge=0, le=1, description="Pattern confidence")
    impact_score: float = Field(default=0.0, ge=0, le=10, description="Impact score")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    @validator('pattern_type')
    def validate_pattern_type(cls, v):
        """Validate pattern type."""
        valid_patterns = [
            'commit_frequency', 'commit_size', 'commit_timing',
            'code_quality', 'testing_patterns', 'documentation_patterns'
        ]
        if v not in valid_patterns:
            raise ValueError(f'Pattern type must be one of: {valid_patterns}')
        return v


# Specific event classes
class CommitEvent(TypedEvent[CommitData]):
    """Commit-related event."""
    
    event_type: EventType = EventType.COMMIT_CREATED
    
    @classmethod
    def create_commit_created(cls, commit_data: CommitData, **kwargs) -> 'CommitEvent':
        """Create a commit created event."""
        return cls.create(EventType.COMMIT_CREATED, commit_data, **kwargs)
    
    @classmethod
    def create_commit_analyzed(cls, commit_data: CommitData, **kwargs) -> 'CommitEvent':
        """Create a commit analyzed event."""
        return cls.create(EventType.COMMIT_ANALYZED, commit_data, **kwargs)


class AnalysisEvent(TypedEvent[AnalysisData]):
    """Analysis-related event."""
    
    event_type: EventType = EventType.ANALYSIS_COMPLETED
    
    @classmethod
    def create_analysis_started(cls, analysis_data: AnalysisData, **kwargs) -> 'AnalysisEvent':
        """Create an analysis started event."""
        return cls.create(EventType.ANALYSIS_STARTED, analysis_data, **kwargs)
    
    @classmethod
    def create_analysis_completed(cls, analysis_data: AnalysisData, **kwargs) -> 'AnalysisEvent':
        """Create an analysis completed event."""
        return cls.create(EventType.ANALYSIS_COMPLETED, analysis_data, **kwargs)
    
    @classmethod
    def create_analysis_failed(cls, analysis_data: AnalysisData, **kwargs) -> 'AnalysisEvent':
        """Create an analysis failed event."""
        return cls.create(EventType.ANALYSIS_FAILED, analysis_data, **kwargs)


class ErrorEvent(TypedEvent[ErrorData]):
    """Error-related event."""
    
    event_type: EventType = EventType.ERROR_OCCURRED
    
    @classmethod
    def create_error(cls, error_data: ErrorData, **kwargs) -> 'ErrorEvent':
        """Create an error event."""
        return cls.create(EventType.ERROR_OCCURRED, error_data, **kwargs)


class BehaviorEvent(TypedEvent[BehaviorData]):
    """Behavior-related event."""
    
    event_type: EventType = EventType.BEHAVIOR_PATTERN_DETECTED
    
    @classmethod
    def create_pattern_detected(cls, behavior_data: BehaviorData, **kwargs) -> 'BehaviorEvent':
        """Create a pattern detected event."""
        return cls.create(EventType.BEHAVIOR_PATTERN_DETECTED, behavior_data, **kwargs)
    
    @classmethod
    def create_insight_generated(cls, behavior_data: BehaviorData, **kwargs) -> 'BehaviorEvent':
        """Create an insight generated event."""
        return cls.create(EventType.BEHAVIOR_INSIGHT_GENERATED, behavior_data, **kwargs)


# Event factory for creating events
class EventFactory:
    """Factory for creating events with proper metadata."""
    
    @staticmethod
    def create_event(
        event_type: EventType,
        data: Dict[str, Any],
        source: EventSource = EventSource.SYSTEM,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs
    ) -> BaseEvent:
        """Create an event with proper metadata."""
        
        metadata = EventMetadata(
            correlation_id=correlation_id,
            causation_id=causation_id,
            source=source,
            priority=priority,
            **kwargs
        )
        
        return BaseEvent(
            metadata=metadata,
            event_type=event_type,
            data=data
        )
    
    @staticmethod
    def create_commit_event(
        commit_data: CommitData,
        event_type: EventType = EventType.COMMIT_CREATED,
        **kwargs
    ) -> CommitEvent:
        """Create a commit event."""
        return CommitEvent.create(event_type, commit_data, **kwargs)
    
    @staticmethod
    def create_analysis_event(
        analysis_data: AnalysisData,
        event_type: EventType = EventType.ANALYSIS_COMPLETED,
        **kwargs
    ) -> AnalysisEvent:
        """Create an analysis event."""
        return AnalysisEvent.create(event_type, analysis_data, **kwargs)
    
    @staticmethod
    def create_error_event(
        error_data: ErrorData,
        **kwargs
    ) -> ErrorEvent:
        """Create an error event."""
        return ErrorEvent.create(EventType.ERROR_OCCURRED, error_data, **kwargs)


# Event handler interface
class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: BaseEvent) -> bool:
        """Handle an event."""
        pass
    
    @abstractmethod
    def can_handle(self, event: BaseEvent) -> bool:
        """Check if this handler can handle the event."""
        pass


# Event bus interface
class EventBus(ABC):
    """Abstract event bus interface."""
    
    @abstractmethod
    async def publish(self, event: BaseEvent) -> bool:
        """Publish an event to the bus."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Subscribe to events of a specific type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Unsubscribe from events of a specific type."""
        pass


# Event replay and audit
class EventStore(ABC):
    """Abstract event store for persistence and replay."""
    
    @abstractmethod
    async def store(self, event: BaseEvent) -> bool:
        """Store an event."""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[BaseEvent]:
        """Get events with optional filtering."""
        pass
    
    @abstractmethod
    async def get_event_by_id(self, event_id: str) -> Optional[BaseEvent]:
        """Get event by ID."""
        pass


# Event utilities
class EventUtils:
    """Utility functions for event handling."""
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def is_retryable_event(event: BaseEvent) -> bool:
        """Check if an event can be retried."""
        return (
            event.metadata.retry_count < event.metadata.max_retries and
            event.metadata.status != EventStatus.COMPLETED
        )
    
    @staticmethod
    def increment_retry_count(event: BaseEvent) -> BaseEvent:
        """Increment the retry count for an event."""
        event.metadata.retry_count += 1
        event.metadata.status = EventStatus.RETRY
        return event
    
    @staticmethod
    def set_event_status(event: BaseEvent, status: EventStatus) -> BaseEvent:
        """Set the status of an event."""
        event.metadata.status = status
        return event
    
    @staticmethod
    def is_event_expired(event: BaseEvent) -> bool:
        """Check if an event has expired."""
        if event.metadata.ttl is None:
            return False
        
        age = (datetime.now(timezone.utc) - event.metadata.timestamp).total_seconds()
        return age > event.metadata.ttl
    
    @staticmethod
    def filter_events_by_type(events: List[BaseEvent], event_type: EventType) -> List[BaseEvent]:
        """Filter events by type."""
        return [event for event in events if event.event_type == event_type]
    
    @staticmethod
    def filter_events_by_time_range(
        events: List[BaseEvent],
        start_time: datetime,
        end_time: datetime
    ) -> List[BaseEvent]:
        """Filter events by time range."""
        return [
            event for event in events
            if start_time <= event.metadata.timestamp <= end_time
        ]


# Event serialization helpers
class EventSerializer:
    """Helper for event serialization and deserialization."""
    
    @staticmethod
    def serialize(event: BaseEvent) -> str:
        """Serialize an event to JSON string."""
        return event.to_json()
    
    @staticmethod
    def deserialize(json_str: str) -> BaseEvent:
        """Deserialize an event from JSON string."""
        return BaseEvent.from_json(json_str)
    
    @staticmethod
    def serialize_batch(events: List[BaseEvent]) -> str:
        """Serialize a batch of events to JSON string."""
        return json.dumps([event.dict() for event in events])
    
    @staticmethod
    def deserialize_batch(json_str: str) -> List[BaseEvent]:
        """Deserialize a batch of events from JSON string."""
        data = json.loads(json_str)
        return [BaseEvent(**event_data) for event_data in data]


# Event validation
class EventValidator:
    """Validator for events."""
    
    @staticmethod
    def validate_event(event: BaseEvent) -> List[str]:
        """Validate an event and return list of errors."""
        errors = []
        
        # Validate metadata
        if not event.metadata.event_id:
            errors.append("Event ID is required")
        
        if not event.metadata.timestamp:
            errors.append("Event timestamp is required")
        
        # Validate event type
        if not event.event_type:
            errors.append("Event type is required")
        
        # Validate data
        if not isinstance(event.data, dict):
            errors.append("Event data must be a dictionary")
        
        # Validate correlation ID format
        if event.metadata.correlation_id and not EventValidator._is_valid_uuid(
            event.metadata.correlation_id
        ):
            errors.append("Invalid correlation ID format")
        
        return errors
    
    @staticmethod
    def _is_valid_uuid(uuid_str: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            uuid.UUID(uuid_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid(event: BaseEvent) -> bool:
        """Check if an event is valid."""
        return len(EventValidator.validate_event(event)) == 0


# Export commonly used classes and functions
__all__ = [
    'EventType', 'EventPriority', 'EventStatus', 'EventSource',
    'EventMetadata', 'BaseEvent', 'TypedEvent',
    'CommitData', 'AnalysisData', 'ErrorData', 'BehaviorData',
    'CommitEvent', 'AnalysisEvent', 'ErrorEvent', 'BehaviorEvent',
    'EventFactory', 'EventHandler', 'EventBus', 'EventStore',
    'EventUtils', 'EventSerializer', 'EventValidator'
]
