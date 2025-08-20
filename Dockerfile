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

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
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

# Expose port
EXPOSE 8000 8001 8002 8003 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "services.frontend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

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
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "services.frontend.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Service-specific stages
FROM base as commit-tracker

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

# Commit tracker command
CMD ["python", "-m", "uvicorn", "services.commit_tracker.main:app", "--host", "0.0.0.0", "--port", "8001"]

FROM base as ai-analysis

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

# AI analysis command
CMD ["python", "-m", "uvicorn", "services.ai_analysis.main:app", "--host", "0.0.0.0", "--port", "8002"]

FROM base as database-service

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Database service command
CMD ["python", "-m", "uvicorn", "services.database.main:app", "--host", "0.0.0.0", "--port", "8003"]

FROM base as frontend

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Frontend command
CMD ["python", "-m", "uvicorn", "services.frontend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base as github-webhook

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/behaviors && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# GitHub webhook command
CMD ["python", "-m", "uvicorn", "services.github_webhook.main:app", "--host", "0.0.0.0", "--port", "8004"]
