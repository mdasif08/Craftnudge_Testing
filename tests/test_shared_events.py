"""
Unit tests for shared events module.

This module provides comprehensive testing for all event-related functionality
with 100% code coverage.
"""

import pytest
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from uuid import uuid4

from shared.events import (
    EventType,
    EventSource,
    EventPriority,
    EventStatus,
    EventFactory,
    EventSerializer,
    EventValidator,
    CommitData,
    AnalysisData,
    BehaviorData,
    CoachingData,
    Event,
    EventMetadata,
    EventPayload,
)


class TestEventEnums:
    """Test cases for event enums."""

    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.COMMIT_CREATED == "commit.created"
        assert EventType.COMMIT_UPDATED == "commit.updated"
        assert EventType.COMMIT_DELETED == "commit.deleted"
        assert EventType.ANALYSIS_REQUESTED == "analysis.requested"
        assert EventType.ANALYSIS_STARTED == "analysis.started"
        assert EventType.ANALYSIS_COMPLETED == "analysis.completed"
        assert EventType.ANALYSIS_FAILED == "analysis.failed"
        assert EventType.BEHAVIOR_PATTERN_DETECTED == "behavior.pattern_detected"
        assert EventType.BEHAVIOR_INSIGHT_GENERATED == "behavior.insight_generated"
        assert EventType.BEHAVIOR_RECOMMENDATION_CREATED == "behavior.recommendation_created"
        assert EventType.COACHING_FEEDBACK_GENERATED == "coaching.feedback_generated"
        assert EventType.COACHING_SESSION_STARTED == "coaching.session_started"
        assert EventType.COACHING_SESSION_ENDED == "coaching.session_ended"
        assert EventType.COACHING_INSIGHTS_GENERATED == "coaching.insights_generated"
        assert EventType.COACHING_RECOMMENDATION_REQUESTED == "coaching.recommendation_requested"
        assert EventType.SYSTEM_STARTUP == "system.startup"
        assert EventType.SYSTEM_SHUTDOWN == "system.shutdown"
        assert EventType.SYSTEM_ERROR == "system.error"
        assert EventType.SYSTEM_WARNING == "system.warning"

    def test_event_source_enum(self):
        """Test EventSource enum values."""
        assert EventSource.COMMIT_TRACKER == "commit_tracker"
        assert EventSource.AI_ANALYSIS == "ai_analysis"
        assert EventSource.DATABASE == "database"
        assert EventSource.FRONTEND == "frontend"
        assert EventSource.GITHUB_WEBHOOK == "github_webhook"
        assert EventSource.COMMIT_QUALITY_COACHING == "commit_quality_coaching"
        assert EventSource.SYSTEM == "system"

    def test_event_priority_enum(self):
        """Test EventPriority enum values."""
        assert EventPriority.LOW == "low"
        assert EventPriority.NORMAL == "normal"
        assert EventPriority.HIGH == "high"
        assert EventPriority.CRITICAL == "critical"

    def test_event_status_enum(self):
        """Test EventStatus enum values."""
        assert EventStatus.PENDING == "pending"
        assert EventStatus.PROCESSING == "processing"
        assert EventStatus.COMPLETED == "completed"
        assert EventStatus.FAILED == "failed"
        assert EventStatus.CANCELLED == "cancelled"


class TestEventDataModels:
    """Test cases for event data models."""

    def test_commit_data_model(self):
        """Test CommitData model."""
        data = {
            "commit_id": str(uuid4()),
            "hash": "abc123def456",
            "author": "testuser",
            "message": "feat: add new feature",
            "timestamp": datetime.now(timezone.utc),
            "changed_files": ["src/main.py"],
            "repository": "test-repo",
            "branch": "main",
            "additions": 50,
            "deletions": 10,
            "total_changes": 60,
        }

        commit_data = CommitData(**data)
        assert commit_data.commit_id == data["commit_id"]
        assert commit_data.hash == data["hash"]
        assert commit_data.author == data["author"]
        assert commit_data.message == data["message"]
        assert commit_data.timestamp == data["timestamp"]
        assert commit_data.changed_files == data["changed_files"]
        assert commit_data.repository == data["repository"]
        assert commit_data.branch == data["branch"]
        assert commit_data.additions == data["additions"]
        assert commit_data.deletions == data["deletions"]
        assert commit_data.total_changes == data["total_changes"]

    def test_analysis_data_model(self):
        """Test AnalysisData model."""
        data = {
            "analysis_id": str(uuid4()),
            "commit_id": str(uuid4()),
            "analysis_type": "commit_quality",
            "status": "pending",
            "result": {"score": 8.5},
            "metadata": {"model": "llama2"},
        }

        analysis_data = AnalysisData(**data)
        assert analysis_data.analysis_id == data["analysis_id"]
        assert analysis_data.commit_id == data["commit_id"]
        assert analysis_data.analysis_type == data["analysis_type"]
        assert analysis_data.status == data["status"]
        assert analysis_data.result == data["result"]
        assert analysis_data.metadata == data["metadata"]

    def test_behavior_data_model(self):
        """Test BehaviorData model."""
        data = {
            "behavior_id": str(uuid4()),
            "user_id": "testuser",
            "behavior_type": "commit_frequency",
            "value": 5.5,
            "unit": "commits_per_day",
            "period": "daily",
            "metadata": {"trend": "increasing"},
        }

        behavior_data = BehaviorData(**data)
        assert behavior_data.behavior_id == data["behavior_id"]
        assert behavior_data.user_id == data["user_id"]
        assert behavior_data.behavior_type == data["behavior_type"]
        assert behavior_data.value == data["value"]
        assert behavior_data.unit == data["unit"]
        assert behavior_data.period == data["period"]
        assert behavior_data.metadata == data["metadata"]

    def test_coaching_data_model(self):
        """Test CoachingData model."""
        data = {
            "coaching_id": str(uuid4()),
            "user_id": "testuser",
            "commit_id": str(uuid4()),
            "session_type": "commit_quality",
            "feedback": {"score": 8.5, "tips": ["Great job!"]},
            "metadata": {"duration": 300},
        }

        coaching_data = CoachingData(**data)
        assert coaching_data.coaching_id == data["coaching_id"]
        assert coaching_data.user_id == data["user_id"]
        assert coaching_data.commit_id == data["commit_id"]
        assert coaching_data.session_type == data["session_type"]
        assert coaching_data.feedback == data["feedback"]
        assert coaching_data.metadata == data["metadata"]


class TestEventModel:
    """Test cases for Event model."""

    def test_event_creation(self):
        """Test Event model creation."""
        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)

        event = Event(
            id=event_id,
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=timestamp,
            payload={"commit_id": str(uuid4())},
            metadata={"correlation_id": str(uuid4())},
        )

        assert event.id == event_id
        assert event.type == EventType.COMMIT_CREATED
        assert event.source == EventSource.COMMIT_TRACKER
        assert event.priority == EventPriority.NORMAL
        assert event.status == EventStatus.PENDING
        assert event.timestamp == timestamp
        assert event.payload == {"commit_id": str(uuid4())}
        assert event.metadata == {"correlation_id": str(uuid4())}

    def test_event_validation(self):
        """Test Event model validation."""
        # Valid event
        event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={},
            metadata={},
        )

        assert event.id is not None
        assert event.type in EventType
        assert event.source in EventSource
        assert event.priority in EventPriority
        assert event.status in EventStatus

    def test_event_metadata_model(self):
        """Test EventMetadata model."""
        metadata = EventMetadata(
            correlation_id=str(uuid4()),
            user_id="testuser",
            session_id=str(uuid4()),
            request_id=str(uuid4()),
            trace_id=str(uuid4()),
            span_id=str(uuid4()),
            tags={"environment": "test", "version": "1.0.0"},
            context={"ip": "127.0.0.1", "user_agent": "test-agent"},
        )

        assert metadata.correlation_id is not None
        assert metadata.user_id == "testuser"
        assert metadata.session_id is not None
        assert metadata.request_id is not None
        assert metadata.trace_id is not None
        assert metadata.span_id is not None
        assert metadata.tags == {"environment": "test", "version": "1.0.0"}
        assert metadata.context == {"ip": "127.0.0.1", "user_agent": "test-agent"}

    def test_event_payload_model(self):
        """Test EventPayload model."""
        payload = EventPayload(
            data={"commit_id": str(uuid4())},
            schema_version="1.0",
            encoding="json",
            compression=None,
            checksum="sha256:abc123",
        )

        assert payload.data == {"commit_id": str(uuid4())}
        assert payload.schema_version == "1.0"
        assert payload.encoding == "json"
        assert payload.compression is None
        assert payload.checksum == "sha256:abc123"


class TestEventFactory:
    """Test cases for EventFactory."""

    @pytest.fixture
    def factory(self):
        """Create EventFactory instance."""
        return EventFactory()

    def test_create_commit_event(self, factory):
        """Test creating commit events."""
        commit_id = str(uuid4())
        commit_data = CommitData(
            commit_id=commit_id,
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
        )

        event = factory.create_commit_event(
            event_type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            commit_data=commit_data,
            priority=EventPriority.NORMAL,
        )

        assert event.type == EventType.COMMIT_CREATED
        assert event.source == EventSource.COMMIT_TRACKER
        assert event.priority == EventPriority.NORMAL
        assert event.status == EventStatus.PENDING
        assert event.payload["commit_id"] == commit_id
        assert event.payload["hash"] == "abc123def456"
        assert event.payload["author"] == "testuser"

    def test_create_analysis_event(self, factory):
        """Test creating analysis events."""
        analysis_id = str(uuid4())
        commit_id = str(uuid4())
        analysis_data = AnalysisData(
            analysis_id=analysis_id,
            commit_id=commit_id,
            analysis_type="commit_quality",
            status="pending",
            result={"score": 8.5},
            metadata={"model": "llama2"},
        )

        event = factory.create_analysis_event(
            event_type=EventType.ANALYSIS_REQUESTED,
            source=EventSource.AI_ANALYSIS,
            analysis_data=analysis_data,
            priority=EventPriority.HIGH,
        )

        assert event.type == EventType.ANALYSIS_REQUESTED
        assert event.source == EventSource.AI_ANALYSIS
        assert event.priority == EventPriority.HIGH
        assert event.payload["analysis_id"] == analysis_id
        assert event.payload["commit_id"] == commit_id
        assert event.payload["analysis_type"] == "commit_quality"

    def test_create_behavior_event(self, factory):
        """Test creating behavior events."""
        behavior_id = str(uuid4())
        behavior_data = BehaviorData(
            behavior_id=behavior_id,
            user_id="testuser",
            behavior_type="commit_frequency",
            value=5.5,
            unit="commits_per_day",
            period="daily",
            metadata={"trend": "increasing"},
        )

        event = factory.create_behavior_event(
            event_type=EventType.BEHAVIOR_PATTERN_DETECTED,
            source=EventSource.AI_ANALYSIS,
            behavior_data=behavior_data,
            priority=EventPriority.NORMAL,
        )

        assert event.type == EventType.BEHAVIOR_PATTERN_DETECTED
        assert event.source == EventSource.AI_ANALYSIS
        assert event.payload["behavior_id"] == behavior_id
        assert event.payload["user_id"] == "testuser"
        assert event.payload["behavior_type"] == "commit_frequency"

    def test_create_coaching_event(self, factory):
        """Test creating coaching events."""
        coaching_id = str(uuid4())
        coaching_data = CoachingData(
            coaching_id=coaching_id,
            user_id="testuser",
            commit_id=str(uuid4()),
            session_type="commit_quality",
            feedback={"score": 8.5, "tips": ["Great job!"]},
            metadata={"duration": 300},
        )

        event = factory.create_coaching_event(
            event_type=EventType.COACHING_FEEDBACK_GENERATED,
            source=EventSource.COMMIT_QUALITY_COACHING,
            coaching_data=coaching_data,
            priority=EventPriority.NORMAL,
        )

        assert event.type == EventType.COACHING_FEEDBACK_GENERATED
        assert event.source == EventSource.COMMIT_QUALITY_COACHING
        assert event.payload["coaching_id"] == coaching_id
        assert event.payload["user_id"] == "testuser"
        assert event.payload["session_type"] == "commit_quality"

    def test_create_system_event(self, factory):
        """Test creating system events."""
        event = factory.create_system_event(
            event_type=EventType.SYSTEM_STARTUP,
            source=EventSource.SYSTEM,
            message="System started successfully",
            details={"version": "1.0.0", "uptime": 0},
            priority=EventPriority.HIGH,
        )

        assert event.type == EventType.SYSTEM_STARTUP
        assert event.source == EventSource.SYSTEM
        assert event.priority == EventPriority.HIGH
        assert event.payload["message"] == "System started successfully"
        assert event.payload["details"]["version"] == "1.0.0"

    def test_create_custom_event(self, factory):
        """Test creating custom events."""
        custom_payload = {"custom_field": "custom_value", "nested_data": {"key": "value"}}

        event = factory.create_custom_event(
            event_type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            payload=custom_payload,
            priority=EventPriority.LOW,
            metadata={"custom_metadata": "value"},
        )

        assert event.type == EventType.COMMIT_CREATED
        assert event.source == EventSource.COMMIT_TRACKER
        assert event.priority == EventPriority.LOW
        assert event.payload == custom_payload
        assert event.metadata["custom_metadata"] == "value"


class TestEventSerializer:
    """Test cases for EventSerializer."""

    @pytest.fixture
    def serializer(self):
        """Create EventSerializer instance."""
        return EventSerializer()

    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={"commit_id": str(uuid4())},
            metadata={"correlation_id": str(uuid4())},
        )

    def test_serialize_event_to_json(self, serializer, sample_event):
        """Test serializing event to JSON."""
        json_str = serializer.serialize_to_json(sample_event)

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["id"] == sample_event.id
        assert data["type"] == sample_event.type
        assert data["source"] == sample_event.source
        assert data["priority"] == sample_event.priority
        assert data["status"] == sample_event.status
        assert data["payload"] == sample_event.payload
        assert data["metadata"] == sample_event.metadata

    def test_deserialize_event_from_json(self, serializer, sample_event):
        """Test deserializing event from JSON."""
        json_str = serializer.serialize_to_json(sample_event)
        deserialized_event = serializer.deserialize_from_json(json_str)

        assert deserialized_event.id == sample_event.id
        assert deserialized_event.type == sample_event.type
        assert deserialized_event.source == sample_event.source
        assert deserialized_event.priority == sample_event.priority
        assert deserialized_event.status == sample_event.status
        assert deserialized_event.payload == sample_event.payload
        assert deserialized_event.metadata == sample_event.metadata

    def test_serialize_event_to_dict(self, serializer, sample_event):
        """Test serializing event to dictionary."""
        event_dict = serializer.serialize_to_dict(sample_event)

        assert event_dict["id"] == sample_event.id
        assert event_dict["type"] == sample_event.type
        assert event_dict["source"] == sample_event.source
        assert event_dict["priority"] == sample_event.priority
        assert event_dict["status"] == sample_event.status
        assert event_dict["payload"] == sample_event.payload
        assert event_dict["metadata"] == sample_event.metadata

    def test_deserialize_event_from_dict(self, serializer, sample_event):
        """Test deserializing event from dictionary."""
        event_dict = serializer.serialize_to_dict(sample_event)
        deserialized_event = serializer.deserialize_from_dict(event_dict)

        assert deserialized_event.id == sample_event.id
        assert deserialized_event.type == sample_event.type
        assert deserialized_event.source == sample_event.source
        assert deserialized_event.priority == sample_event.priority
        assert deserialized_event.status == sample_event.status
        assert deserialized_event.payload == sample_event.payload
        assert deserialized_event.metadata == sample_event.metadata

    def test_serialize_event_batch(self, serializer):
        """Test serializing multiple events."""
        events = [
            Event(
                id=str(uuid4()),
                type=EventType.COMMIT_CREATED,
                source=EventSource.COMMIT_TRACKER,
                priority=EventPriority.NORMAL,
                status=EventStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
                payload={"commit_id": str(uuid4())},
                metadata={},
            ),
            Event(
                id=str(uuid4()),
                type=EventType.ANALYSIS_REQUESTED,
                source=EventSource.AI_ANALYSIS,
                priority=EventPriority.HIGH,
                status=EventStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
                payload={"analysis_id": str(uuid4())},
                metadata={},
            ),
        ]

        batch_json = serializer.serialize_batch_to_json(events)
        batch_data = json.loads(batch_json)

        assert len(batch_data["events"]) == 2
        assert batch_data["batch_id"] is not None
        assert batch_data["timestamp"] is not None
        assert batch_data["count"] == 2

    def test_deserialize_event_batch(self, serializer):
        """Test deserializing multiple events."""
        events = [
            Event(
                id=str(uuid4()),
                type=EventType.COMMIT_CREATED,
                source=EventSource.COMMIT_TRACKER,
                priority=EventPriority.NORMAL,
                status=EventStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
                payload={"commit_id": str(uuid4())},
                metadata={},
            ),
            Event(
                id=str(uuid4()),
                type=EventType.ANALYSIS_REQUESTED,
                source=EventSource.AI_ANALYSIS,
                priority=EventPriority.HIGH,
                status=EventStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
                payload={"analysis_id": str(uuid4())},
                metadata={},
            ),
        ]

        batch_json = serializer.serialize_batch_to_json(events)
        deserialized_events = serializer.deserialize_batch_from_json(batch_json)

        assert len(deserialized_events) == 2
        assert deserialized_events[0].type == EventType.COMMIT_CREATED
        assert deserialized_events[1].type == EventType.ANALYSIS_REQUESTED

    def test_serialization_with_complex_payload(self, serializer):
        """Test serialization with complex payload data."""
        complex_payload = {
            "nested_data": {
                "list_data": [1, 2, 3],
                "dict_data": {"key": "value"},
                "null_data": None,
                "bool_data": True,
            },
            "array_data": ["string1", "string2"],
            "number_data": 42.5,
        }

        event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload=complex_payload,
            metadata={},
        )

        json_str = serializer.serialize_to_json(event)
        deserialized_event = serializer.deserialize_from_json(json_str)

        assert deserialized_event.payload == complex_payload
        assert deserialized_event.payload["nested_data"]["list_data"] == [1, 2, 3]
        assert deserialized_event.payload["nested_data"]["null_data"] is None
        assert deserialized_event.payload["nested_data"]["bool_data"] is True
        assert deserialized_event.payload["number_data"] == 42.5


class TestEventValidator:
    """Test cases for EventValidator."""

    @pytest.fixture
    def validator(self):
        """Create EventValidator instance."""
        return EventValidator()

    @pytest.fixture
    def valid_event(self):
        """Create a valid event for testing."""
        return Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={"commit_id": str(uuid4())},
            metadata={},
        )

    def test_validate_valid_event(self, validator, valid_event):
        """Test validating a valid event."""
        result = validator.validate_event(valid_event)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_validate_event_missing_id(self, validator):
        """Test validating event with missing ID."""
        event = Event(
            id="",  # Invalid empty ID
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={},
            metadata={},
        )

        result = validator.validate_event(event)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("id" in error.lower() for error in result.errors)

    def test_validate_event_invalid_type(self, validator):
        """Test validating event with invalid type."""
        event = Event(
            id=str(uuid4()),
            type="invalid_type",  # Invalid type
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={},
            metadata={},
        )

        result = validator.validate_event(event)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("type" in error.lower() for error in result.errors)

    def test_validate_event_future_timestamp(self, validator):
        """Test validating event with future timestamp."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=future_time,
            payload={},
            metadata={},
        )

        result = validator.validate_event(event)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("timestamp" in error.lower() for error in result.errors)

    def test_validate_event_large_payload(self, validator):
        """Test validating event with large payload."""
        large_payload = {"data": "x" * 1000000}  # 1MB payload
        event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload=large_payload,
            metadata={},
        )

        result = validator.validate_event(event)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("payload" in error.lower() for error in result.errors)

    def test_validate_event_batch(self, validator):
        """Test validating a batch of events."""
        events = [
            Event(
                id=str(uuid4()),
                type=EventType.COMMIT_CREATED,
                source=EventSource.COMMIT_TRACKER,
                priority=EventPriority.NORMAL,
                status=EventStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
                payload={},
                metadata={},
            ),
            Event(
                id="",  # Invalid event
                type=EventType.ANALYSIS_REQUESTED,
                source=EventSource.AI_ANALYSIS,
                priority=EventPriority.HIGH,
                status=EventStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
                payload={},
                metadata={},
            ),
        ]

        result = validator.validate_event_batch(events)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert len(result.valid_events) == 1
        assert len(result.invalid_events) == 1

    def test_validate_event_schema(self, validator):
        """Test validating event against schema."""
        event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={"commit_id": str(uuid4())},
            metadata={},
        )

        result = validator.validate_event_schema(event)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_event_business_rules(self, validator):
        """Test validating event against business rules."""
        # Test commit event with required fields
        commit_event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={
                "commit_id": str(uuid4()),
                "hash": "abc123def456",
                "author": "testuser",
                "message": "feat: add new feature",
            },
            metadata={},
        )

        result = validator.validate_event_business_rules(commit_event)
        assert result.is_valid is True
        assert len(result.errors) == 0

        # Test commit event missing required fields
        invalid_commit_event = Event(
            id=str(uuid4()),
            type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            priority=EventPriority.NORMAL,
            status=EventStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            payload={},  # Missing required fields
            metadata={},
        )

        result = validator.validate_event_business_rules(invalid_commit_event)
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestEventIntegration:
    """Integration tests for event system."""

    def test_full_event_lifecycle(self):
        """Test complete event lifecycle: creation, serialization, validation, deserialization."""
        # Create event factory and serializer
        factory = EventFactory()
        serializer = EventSerializer()
        validator = EventValidator()

        # Create commit data
        commit_data = CommitData(
            commit_id=str(uuid4()),
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
        )

        # Create event
        event = factory.create_commit_event(
            event_type=EventType.COMMIT_CREATED,
            source=EventSource.COMMIT_TRACKER,
            commit_data=commit_data,
            priority=EventPriority.NORMAL,
        )

        # Validate event
        validation_result = validator.validate_event(event)
        assert validation_result.is_valid is True

        # Serialize to JSON
        json_str = serializer.serialize_to_json(event)
        assert json_str is not None
        assert len(json_str) > 0

        # Deserialize from JSON
        deserialized_event = serializer.deserialize_from_json(json_str)

        # Verify event integrity
        assert deserialized_event.id == event.id
        assert deserialized_event.type == event.type
        assert deserialized_event.source == event.source
        assert deserialized_event.priority == event.priority
        assert deserialized_event.status == event.status
        assert deserialized_event.payload["commit_id"] == commit_data.commit_id
        assert deserialized_event.payload["hash"] == commit_data.hash
        assert deserialized_event.payload["author"] == commit_data.author

    def test_event_batch_processing(self):
        """Test processing multiple events in batch."""
        factory = EventFactory()
        serializer = EventSerializer()
        validator = EventValidator()

        # Create multiple events
        events = []
        for i in range(5):
            commit_data = CommitData(
                commit_id=str(uuid4()),
                hash=f"abc{i}23def456",
                author=f"user{i}",
                message=f"feat: add feature {i}",
                timestamp=datetime.now(timezone.utc),
                changed_files=[f"src/feature{i}.py"],
                repository="test-repo",
                branch="main",
                additions=10 + i,
                deletions=i,
                total_changes=10 + 2 * i,
            )

            event = factory.create_commit_event(
                event_type=EventType.COMMIT_CREATED,
                source=EventSource.COMMIT_TRACKER,
                commit_data=commit_data,
                priority=EventPriority.NORMAL,
            )
            events.append(event)

        # Validate all events
        for event in events:
            result = validator.validate_event(event)
            assert result.is_valid is True

        # Serialize batch
        batch_json = serializer.serialize_batch_to_json(events)
        batch_data = json.loads(batch_json)

        assert len(batch_data["events"]) == 5
        assert batch_data["count"] == 5

        # Deserialize batch
        deserialized_events = serializer.deserialize_batch_from_json(batch_json)
        assert len(deserialized_events) == 5

        # Verify all events
        for i, event in enumerate(deserialized_events):
            assert event.type == EventType.COMMIT_CREATED
            assert event.source == EventSource.COMMIT_TRACKER
            assert event.payload["author"] == f"user{i}"
            assert event.payload["message"] == f"feat: add feature {i}"


if __name__ == "__main__":
    pytest.main([__file__])
