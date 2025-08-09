#!/usr/bin/env python3
"""
Development server runner for AI Video Tutor Backend

This script helps run the FastAPI application with proper Python path setup
to handle imports from different directories.
"""

import os
import sys
import uvicorn
from pathlib import Path

def setup_python_path():
    """Add necessary directories to Python path for imports"""
    # Get the current directory (backend/)
    backend_dir = Path(__file__).parent.absolute()
    
    # Get the project root directory (SF_full/)
    project_root = backend_dir.parent
    
    # Add directories to Python path
    paths_to_add = [
        str(backend_dir),           # For app.* imports
        str(project_root),          # For agents.* imports
        str(backend_dir / "app"),   # For direct module imports
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"Python path setup completed:")
    for path in paths_to_add:
        print(f"  - {path}")

def main():
    """Main function to run the development server"""
    setup_python_path()
    
    # Import the app after setting up the path
    from app.main import app
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=["app/"],  # Only reload on changes in app directory
    )

if __name__ == "__main__":
    main()
