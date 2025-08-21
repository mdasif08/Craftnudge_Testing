#!/usr/bin/env python3
"""
Commit Quality Coaching Service Entry Point

This script starts the Commit Quality Coaching microservice.
"""

import uvicorn
from config.settings import Settings

def main():
    """Start the Commit Quality Coaching service."""
    settings = Settings()
    
    uvicorn.run(
        "services.commit_quality_coaching.main:app",
        host="0.0.0.0",
        port=settings.services.commit_quality_coaching.port,
        reload=settings.app.debug,
        log_level="info" if settings.app.debug else "warning"
    )

if __name__ == "__main__":
    main()
