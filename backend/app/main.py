from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import time
from typing import Optional, Dict, Any
from celery.result import AsyncResult

# Import settings with proper path handling
try:
    from app.core.settings import settings
    from app.core.redis_client import redis_client
    from app.core.celery_app import celery_app
    from app.tasks.hello_tasks import hello_task, add_numbers, long_running_task
    # Video tasks are now organized by purpose:
    # - video_tasks.py: Low-level agent operations  
    # - video_generation_tasks.py: High-level orchestrated workflows
    from app.api.video import router as video_router
    from app.api.video_generation import router as video_generation_router
except ImportError:
    # Fallback for when running from different directories
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.core.settings import Settings
    # Create settings with defaults for basic startup
    settings = Settings(
        REDIS_URL="redis://localhost:6379",
        CONVEX_WEBHOOK_SECRET="dev-secret",
        CONVEX_ACTION_BASE_URL="http://localhost:3000"
    )
    # Import other modules after path setup
    from app.core.redis_client import redis_client
    from app.core.celery_app import celery_app
    from app.tasks.hello_tasks import hello_task, add_numbers, long_running_task
    # Video tasks are now organized by purpose (see above)
    from app.api.video import router as video_router

# Initialize FastAPI app
app = FastAPI(
    title="AI Video Tutor API",
    description="Backend API for AI Video Tutor Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class HealthResponse(BaseModel):
    status: str
    environment: str
    message: str
    redis_connected: bool
    celery_broker_connected: bool
    timestamp: float

class HelloRequest(BaseModel):
    name: str = "World"

class HelloResponse(BaseModel):
    message: str
    name: str

# Task models
class TaskRequest(BaseModel):
    name: str = Field(..., description="Name to greet")

class AddNumbersRequest(BaseModel):
    x: int = Field(..., description="First number")
    y: int = Field(..., description="Second number")

class LongTaskRequest(BaseModel):
    duration: int = Field(default=10, ge=1, le=300, description="Duration in seconds")

# Video generation models are now in app.schemas.video_generation

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Mount API routers
app.include_router(video_router, prefix=settings.API_PREFIX)
app.include_router(video_generation_router, prefix=settings.API_PREFIX)

# Basic routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "AI Video Tutor API is running!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with Redis and Celery status"""
    redis_connected = await redis_client.ping()
    
    # Check Celery broker connection
    celery_broker_connected = False
    try:
        # Try to get Celery worker stats
        stats = celery_app.control.inspect().stats()
        celery_broker_connected = stats is not None
    except Exception:
        celery_broker_connected = False
    
    status = "healthy" if redis_connected and celery_broker_connected else "degraded"
    
    return HealthResponse(
        status=status,
        environment=settings.ENV,
        message="AI Video Tutor API is running successfully",
        redis_connected=redis_connected,
        celery_broker_connected=celery_broker_connected,
        timestamp=time.time()
    )

@app.post("/hello", response_model=HelloResponse)
async def hello_world(request: HelloRequest):
    """Simple hello world endpoint"""
    return HelloResponse(
        message=f"Hello, {request.name}!",
        name=request.name
    )

@app.get("/hello/{name}", response_model=HelloResponse)
async def hello_path(name: str):
    """Hello endpoint with path parameter"""
    return HelloResponse(
        message=f"Hello, {name}!",
        name=name
    )

# Additional example endpoints
@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    return {
        "api_version": "v1",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "hello": "/hello",
            "docs": "/docs"
        }
    }

# ==================== TASK MANAGEMENT ENDPOINTS ====================

@app.post("/tasks/hello", response_model=TaskResponse)
async def create_hello_task(request: TaskRequest):
    """Create a hello world task"""
    task = hello_task.delay(request.name)
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Hello task created for {request.name}"
    )

@app.post("/tasks/add", response_model=TaskResponse)
async def create_add_task(request: AddNumbersRequest):
    """Create an addition task"""
    task = add_numbers.delay(request.x, request.y)
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Addition task created: {request.x} + {request.y}"
    )

@app.post("/tasks/long", response_model=TaskResponse)
async def create_long_task(request: LongTaskRequest):
    """Create a long running task with progress updates"""
    task = long_running_task.delay(request.duration)
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Long running task created ({request.duration}s)"
    )

# Video generation tasks are now handled by the dedicated routers:
# - /api/v1/video-generation/* for high-level workflows
# - /api/v1/video/agents/* for low-level agent operations

@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a specific task"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status,
            meta=result.info if result.info else None
        )
        
        if result.successful():
            response.result = result.result
        elif result.failed():
            response.error = str(result.info)
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {str(e)}")

@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """Get the result of a completed task"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if not result.ready():
            raise HTTPException(status_code=202, detail="Task not yet completed")
        
        if result.failed():
            raise HTTPException(status_code=500, detail=f"Task failed: {result.info}")
            
        return {"task_id": task_id, "result": result.result}
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {str(e)}")

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return {"message": f"Task {task_id} cancellation requested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@app.get("/workers/status")
async def get_worker_status():
    """Get status of Celery workers"""
    try:
        inspect = celery_app.control.inspect()
        
        # Get active tasks, registered tasks, and worker stats
        active = inspect.active() or {}
        registered = inspect.registered() or {}
        stats = inspect.stats() or {}
        
        return {
            "workers": {
                "active_tasks": active,
                "registered_tasks": registered,
                "stats": stats
            },
            "queues": ["default", "render", "high"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get worker status: {str(e)}")

# ==================== REDIS ENDPOINTS ====================

@app.get("/redis/info")
async def get_redis_info():
    """Get Redis server information"""
    try:
        info = redis_client.get_info()
        return {"redis_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Redis info: {str(e)}")

@app.get("/redis/test")
async def test_redis():
    """Test Redis read/write operations"""
    try:
        # Test write
        test_key = f"test_key_{int(time.time())}"
        test_value = {"message": "Redis test", "timestamp": time.time()}
        
        write_success = redis_client.set_json(test_key, test_value, ex=60)
        
        # Test read
        read_value = redis_client.get_json(test_key)
        
        # Cleanup
        redis_client.delete(test_key)
        
        return {
            "write_success": write_success,
            "read_success": read_value is not None,
            "data_matches": read_value == test_value if read_value else False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
