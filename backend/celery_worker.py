#!/usr/bin/env python3
"""
Celery Worker Runner for AI Video Tutor Backend

This script starts a Celery worker with proper configuration for the AI Video Tutor platform.
"""

import os
import sys
from pathlib import Path

def setup_python_path():
    """Add necessary directories to Python path for imports"""
    backend_dir = Path(__file__).parent.absolute()
    project_root = backend_dir.parent
    
    paths_to_add = [
        str(backend_dir),
        str(project_root),
        str(backend_dir / "app"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

def main():
    """Main function to start the Celery worker"""
    setup_python_path()
    
    # Import Celery app after setting up the path
    from app.core.celery_app import celery_app
    
    # Set default log level
    log_level = os.getenv('CELERY_LOG_LEVEL', 'info')
    
    # Start worker with configuration
    celery_app.worker_main([
        'worker',
        '--loglevel=' + log_level,
        '--concurrency=2',  # Adjust based on your machine
        '--max-tasks-per-child=1000',
        '--queues=default,render,high',
        '--hostname=worker@%h',
    ])

if __name__ == "__main__":
    main()
