# Package Import Guide for SF_full Project

This guide explains how to properly import packages and modules across different directories in the SF_full project.

## Project Structure Overview

```
SF_full/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── settings.py
│   │   └── ...
│   ├── run_dev.py
│   └── env.example
├── agents/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── core/
│   │   └── rag/
│   └── ...
├── convex/
├── frontend/
└── pyproject.toml
```

## Import Strategies

### 1. **Running from Backend Directory**

When running FastAPI from the `backend/` directory:

```python
# backend/app/main.py
from app.core.settings import settings          # ✅ Correct
from app.services.video_service import VideoService  # ✅ Correct

# Alternative if running directly from backend/
from core.settings import settings              # ⚠️ Works but not recommended
```

### 2. **Cross-Service Imports (Backend ↔ Agents)**

To import agents from backend or vice versa:

```python
# Method 1: Using sys.path (recommended for development)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from agents.src.core.code_generator import CodeGenerator
from agents.src.rag.vector_store import VectorStore

# Method 2: Using relative imports (if properly structured)
from ...agents.src.core.code_generator import CodeGenerator
```

### 3. **Environment-Aware Import Handling**

For robust imports that work across different execution contexts:

```python
# backend/app/main.py
import os
import sys

def setup_imports():
    """Setup imports to work from different execution contexts"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(backend_dir)
    
    paths_to_add = [
        backend_dir,    # For app.* imports
        project_root,   # For agents.* imports
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# Call setup before other imports
setup_imports()

try:
    from app.core.settings import settings
except ImportError:
    from core.settings import settings
```

## Running the Application

### Method 1: Using the Development Runner (Recommended)

```bash
# From the backend directory
cd backend/
python run_dev.py
```

### Method 2: Using uvicorn directly

```bash
# From the backend directory
cd backend/
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# From the project root
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Using python -m

```bash
# From the project root
python -m backend.app.main

# From the backend directory
python -m app.main
```

## Environment Setup

1. **Copy the environment template:**
```bash
cp backend/env.example backend/.env
```

2. **Edit the `.env` file with your actual values:**
```bash
# Required for basic functionality
REDIS_URL=redis://localhost:6379
CONVEX_WEBHOOK_SECRET=your-secret-here
CONVEX_ACTION_BASE_URL=https://your-deployment.convex.cloud
```

## Common Import Patterns

### For Backend Services

```python
# backend/app/services/video_service.py
from app.core.settings import settings
from app.adapters.s3_adapter import S3Adapter
from agents.src.core.video_renderer import VideoRenderer  # Cross-service import
```

### For Agent Services

```python
# agents/src/core/code_generator.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from backend.app.core.settings import settings  # Cross-service import
from agents.src.rag.vector_store import VectorStore  # Same service import
```

### For Tests

```python
# tests/test_main.py
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.app.main import app
from agents.src.core.code_generator import CodeGenerator
```

## Best Practices

1. **Use absolute imports when possible**
2. **Setup Python path at the application entry point**
3. **Use try/except for fallback import paths**
4. **Keep cross-service imports minimal**
5. **Document import dependencies clearly**

## Troubleshooting

### ModuleNotFoundError

```python
# Add debug information to find missing paths
import sys
print("Current Python path:")
for path in sys.path:
    print(f"  {path}")
```

### Circular Imports

- Avoid importing at module level if possible
- Use local imports inside functions
- Restructure code to remove circular dependencies

### Import from Different Working Directories

Use the development runner script (`run_dev.py`) which handles path setup automatically.

## Testing Your Setup

Run the test script to verify imports work correctly:

```bash
cd backend/
python -c "
from app.main import app
from app.core.settings import settings
print('✅ Backend imports working!')
print(f'App title: {app.title}')
print(f'Environment: {settings.ENV}')
"
```
