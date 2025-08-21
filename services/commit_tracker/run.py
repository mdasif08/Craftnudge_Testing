#!/usr/bin/env python3
"""
Commit Tracker Service Entry Point

This script starts the Commit Tracker microservice.
"""

import uvicorn
from config.settings import Settings

def main():
    """Start the Commit Tracker service."""
    settings = Settings()
    
    uvicorn.run(
        "services.commit_tracker.main:app",
        host="0.0.0.0",
        port=settings.services.commit_tracker.port,
        reload=settings.app.debug,
        log_level="info" if settings.app.debug else "warning"
    )

if __name__ == "__main__":
    main()
