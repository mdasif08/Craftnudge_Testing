# Multi-stage Dockerfile for CraftNudge microservices
# Supports both development and production builds

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy shared requirements first for better caching
COPY requirements.txt .

# Install shared Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports for all services
EXPOSE 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command (commit-tracker)
CMD ["python", "services/commit_tracker/run.py"]

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard]

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Production command
CMD ["gunicorn", "services.commit_tracker.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8001"]

# Commit Tracker Service
FROM base as commit-tracker

# Copy service-specific requirements
COPY services/commit_tracker/requirements.txt ./service-requirements.txt

# Install service-specific dependencies
RUN pip install --no-cache-dir -r service-requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Service command
CMD ["python", "services/commit_tracker/run.py"]

# Commit Quality Coaching Service
FROM base as commit-quality-coaching

# Copy service-specific requirements
COPY services/commit_quality_coaching/requirements.txt ./service-requirements.txt

# Install service-specific dependencies
RUN pip install --no-cache-dir -r service-requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Service command
CMD ["python", "services/commit_quality_coaching/run.py"]

# CLI Tools Stage
FROM base as cli-tools

# Install CLI-specific dependencies
RUN pip install --no-cache-dir \
    click \
    rich \
    httpx

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Default command (commit tracker CLI)
CMD ["python", "services/commit_tracker/cli.py", "--help"]
