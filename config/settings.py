"""
Enterprise-grade configuration management for CraftNudge microservices.

This module provides centralized configuration management with:
- Environment-specific settings
- Type validation and defaults
- Security best practices
- Monitoring and observability
- Database and service configurations
"""

import os
import secrets
from typing import Optional, List, Dict, Any
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    url: str = Field(
        default="postgresql://postgres:password@localhost:5432/craftnudge",
        description="Database connection URL",
    )
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=30, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Connection pool timeout")
    pool_recycle: int = Field(default=3600, description="Connection pool recycle time")
    echo: bool = Field(default=False, description="Enable SQL logging")

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Database URL must be PostgreSQL")
        return v


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[SecretStr] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Enable SSL connection")
    ssl_cert_reqs: str = Field(default="required", description="SSL certificate requirements")
    decode_responses: bool = Field(default=True, description="Decode responses as strings")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout")
    socket_timeout: int = Field(default=5, description="Socket timeout")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    health_check_interval: int = Field(default=30, description="Health check interval")


class OllamaSettings(BaseSettings):
    """Ollama AI configuration settings."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    model: str = Field(default="llama2", description="Default AI model")
    timeout: int = Field(default=30, description="Request timeout")
    max_tokens: int = Field(default=2048, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, description="AI response temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    retry_attempts: int = Field(default=3, description="Retry attempts on failure")

    @field_validator("base_url")
    @classmethod
    def validate_ollama_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must be HTTP/HTTPS")
        return v.rstrip("/")


class GitHubSettings(BaseSettings):
    """GitHub integration configuration settings."""

    webhook_secret: Optional[SecretStr] = Field(
        default=None, description="GitHub webhook secret for verification"
    )
    access_token: Optional[SecretStr] = Field(
        default=None, description="GitHub access token for API calls"
    )
    app_id: Optional[str] = Field(default=None, description="GitHub App ID")
    private_key_path: Optional[str] = Field(default=None, description="GitHub App private key path")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for GitHub")

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must be HTTP/HTTPS")
        return v


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32), description="Application secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    bcrypt_rounds: int = Field(default=12, description="BCrypt rounds")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration settings."""

    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    jaeger_endpoint: Optional[str] = Field(default=None, description="Jaeger tracing endpoint")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry error tracking DSN")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class ServiceSettings(BaseSettings):
    """Service-specific configuration settings."""

    # Service ports
    commit_tracker_port: int = Field(default=8001, description="Commit tracker service port")
    ai_analysis_port: int = Field(default=8002, description="AI analysis service port")
    database_port: int = Field(default=8003, description="Database service port")
    frontend_port: int = Field(default=8000, description="Frontend service port")
    github_webhook_port: int = Field(default=8004, description="GitHub webhook service port")
    commit_quality_coaching_port: int = Field(
        default=8005, description="Commit quality coaching service port"
    )

    # Service timeouts
    request_timeout: int = Field(default=30, description="HTTP request timeout")
    health_check_timeout: int = Field(default=5, description="Health check timeout")
    graceful_shutdown_timeout: int = Field(default=30, description="Graceful shutdown timeout")

    # Service limits
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Max request size (10MB)")
    max_concurrent_requests: int = Field(default=100, description="Max concurrent requests")
    worker_processes: int = Field(default=1, description="Number of worker processes")


class FileSettings(BaseSettings):
    """File and storage configuration settings."""

    data_dir: str = Field(default="data", description="Data directory path")
    commits_file: str = Field(
        default="data/behaviors/commits.jsonl", description="Commits storage file"
    )
    temp_dir: str = Field(default="/tmp/craftnudge", description="Temporary directory")
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max file size (50MB)")
    allowed_extensions: List[str] = Field(
        default=[".json", ".jsonl", ".csv", ".txt"], description="Allowed file extensions"
    )

    @field_validator("data_dir", "temp_dir")
    @classmethod
    def validate_directory_path(cls, v):
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class Settings(PydanticBaseSettings):
    """
    Main application settings with environment-specific configuration.

    Supports multiple environments:
    - development: Local development settings
    - testing: Test environment settings
    - staging: Staging environment settings
    - production: Production environment settings
    """

    # Core application settings
    app_name: str = Field(default="CraftNudge", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Service configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    file: FileSettings = Field(default_factory=FileSettings)

    # Environment-specific overrides
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        valid_environments = ["development", "testing", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v

    @field_validator("debug")
    @classmethod
    def validate_debug_mode(cls, v, info):
        if info.data.get("environment") == "production" and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "__",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application configuration object

    Example:
        >>> settings = get_settings()
        >>> print(settings.database.url)
        >>> print(settings.ollama.model)
    """
    return Settings()


# Global settings instance
settings = get_settings()


def get_database_url() -> str:
    """Get database URL with environment-specific configuration."""
    if settings.environment == "testing":
        return "postgresql://test:test@localhost:5432/craftnudge_test"
    return settings.database.url


def get_redis_url() -> str:
    """Get Redis URL with environment-specific configuration."""
    if settings.environment == "testing":
        return "redis://localhost:6379/1"
    return settings.redis.url


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.environment == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return settings.environment == "development"


def is_testing() -> bool:
    """Check if running in testing environment."""
    return settings.environment == "testing"


# Configuration validation
def validate_configuration() -> Dict[str, Any]:
    """
    Validate all configuration settings and return validation results.

    Returns:
        Dict[str, Any]: Validation results with status and errors

    Example:
        >>> validation = validate_configuration()
        >>> if not validation['valid']:
        >>>     print("Configuration errors:", validation['errors'])
    """
    errors = []
    warnings = []

    # Validate required settings for production
    if is_production():
        if not settings.github.webhook_secret:
            errors.append("GitHub webhook secret is required in production")

        if not settings.security.secret_key or len(settings.security.secret_key) < 32:
            errors.append("Strong secret key is required in production")

        if settings.debug:
            errors.append("Debug mode cannot be enabled in production")

    # Validate database connection
    try:
        import psycopg2
        from urllib.parse import urlparse

        db_url = urlparse(get_database_url())
        if not all([db_url.scheme, db_url.hostname, db_url.port, db_url.username]):
            errors.append("Invalid database URL format")
    except Exception as e:
        errors.append(f"Database configuration error: {e}")

    # Validate Redis connection
    try:
        import redis

        redis_client = redis.from_url(get_redis_url())
        redis_client.ping()
    except Exception as e:
        warnings.append(f"Redis connection warning: {e}")

    # Validate Ollama connection
    try:
        import httpx

        response = httpx.get(f"{settings.ollama.base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            warnings.append("Ollama service may not be available")
    except Exception as e:
        warnings.append(f"Ollama connection warning: {e}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "environment": settings.environment,
        "services": {
            "database": "configured",
            "redis": "configured",
            "ollama": "configured",
            "github": "configured" if settings.github.webhook_secret else "optional",
        },
    }


# Configuration export for external tools
def export_config() -> Dict[str, Any]:
    """
    Export configuration for external tools and monitoring.

    Returns:
        Dict[str, Any]: Configuration export (without sensitive data)
    """
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "debug": settings.debug,
        "database": {
            "pool_size": settings.database.pool_size,
            "pool_timeout": settings.database.pool_timeout,
            "echo": settings.database.echo,
        },
        "redis": {
            "db": settings.redis.db,
            "ssl": settings.redis.ssl,
            "decode_responses": settings.redis.decode_responses,
        },
        "ollama": {
            "base_url": settings.ollama.base_url,
            "model": settings.ollama.model,
            "timeout": settings.ollama.timeout,
        },
        "monitoring": {
            "log_level": settings.monitoring.log_level,
            "enable_metrics": settings.monitoring.enable_metrics,
            "enable_tracing": settings.monitoring.enable_tracing,
        },
        "service": {
            "commit_tracker_port": settings.service.commit_tracker_port,
            "ai_analysis_port": settings.service.ai_analysis_port,
            "database_port": settings.service.database_port,
            "frontend_port": settings.service.frontend_port,
            "github_webhook_port": settings.service.github_webhook_port,
        },
    }


if __name__ == "__main__":
    """Configuration validation script."""
    import json

    validation = validate_configuration()
    config_export = export_config()

    print("Configuration Validation:")
    print(json.dumps(validation, indent=2))

    print("\nConfiguration Export:")
    print(json.dumps(config_export, indent=2))

    if not validation["valid"]:
        exit(1)
