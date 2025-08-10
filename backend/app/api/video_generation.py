"""
High-level video generation and editing workflows API.

This module implements complete video generation and editing workflows with:
- Full video generation pipeline (planning → implementation → rendering → S3 upload)
- Scene editing workflow (retrieve → modify → re-render → update)
- S3 asset management and Convex database synchronization
- Advanced progress tracking and error handling

This is the main API for end-users who want complete video generation workflows,
as opposed to the low-level agent operations in /video/agents/.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from celery.result import AsyncResult

from app.schemas.video_generation import (
    VideoGenerationRequest,
    SceneEditRequest,
    VideoGenerationResponse,
    SceneEditResponse,
    TaskStatusResponse
)
from app.tasks.video_generation_tasks import (
    generate_video_orchestration_task,
    edit_scene_orchestration_task
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-generation", tags=["video-generation"])


@router.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks
) -> VideoGenerationResponse:
    """
    Generate a new video with multiple scenes.
    
    This endpoint accepts parameters for video generation and orchestrates
    the entire pipeline including:
    - Core video generation
    - Video chunking into individual scenes
    - S3 upload for video chunks and source code
    - Convex database updates
    
    Args:
        request: Video generation request parameters
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        VideoGenerationResponse with task information
        
    Raises:
        HTTPException: If request validation fails
    """
    try:
        logger.info(f"Starting video generation for project {request.project_id}")
        
        # Prepare task data
        task_data = {
            "project_id": request.project_id,
            "content_prompt": request.content_prompt,
            "scene_count": request.scene_count,
            "session_id": request.session_id,
            "model_config": request.model_config.dict() if request.model_config else None,
            "render_config": request.render_config.dict() if request.render_config else None,
            "s3_config": request.s3_config.dict() if request.s3_config else None
        }
        
        # Start background task
        task = generate_video_orchestration_task.delay(task_data)
        
        logger.info(f"Video generation task {task.id} queued for project {request.project_id}")
        
        return VideoGenerationResponse(
            task_id=task.id,
            project_id=request.project_id,
            session_id=request.session_id,
            status="queued",
            message=f"Video generation started for {request.scene_count} scenes",
            estimated_duration_minutes=request.scene_count * 2  # Rough estimate
        )
        
    except Exception as e:
        logger.error(f"Failed to start video generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start video generation: {str(e)}"
        )


@router.post("/edit-scene", response_model=SceneEditResponse)
async def edit_scene(
    request: SceneEditRequest,
    background_tasks: BackgroundTasks
) -> SceneEditResponse:
    """
    Edit a specific scene based on user prompt.
    
    This endpoint handles scene editing by:
    - Retrieving original Manim source code from S3
    - Calling Code Generation Agent with edit prompt
    - Re-rendering the modified scene
    - Uploading new version to S3
    - Updating Convex with new URLs and status
    
    Args:
        request: Scene edit request parameters
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        SceneEditResponse with task information
        
    Raises:
        HTTPException: If request validation fails or scene not found
    """
    try:
        logger.info(f"Starting scene edit for scene {request.scene_id}")
        
        # Prepare task data
        task_data = {
            "scene_id": request.scene_id,
            "project_id": request.project_id,
            "edit_prompt": request.edit_prompt,
            "session_id": request.session_id,
            "model_config": request.model_config.dict() if request.model_config else None,
            "render_config": request.render_config.dict() if request.render_config else None,
            "preserve_timing": request.preserve_timing,
            "version_increment": request.version_increment
        }
        
        # Start background task
        task = edit_scene_orchestration_task.delay(task_data)
        
        logger.info(f"Scene edit task {task.id} queued for scene {request.scene_id}")
        
        return SceneEditResponse(
            task_id=task.id,
            scene_id=request.scene_id,
            project_id=request.project_id,
            status="queued",
            message=f"Scene editing started with prompt: {request.edit_prompt[:100]}...",
            estimated_duration_minutes=3  # Scene edits are typically faster
        )
        
    except Exception as e:
        logger.error(f"Failed to start scene edit: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start scene edit: {str(e)}"
        )


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the status of a video generation or scene editing task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        TaskStatusResponse with current task status and progress
        
    Raises:
        HTTPException: If task not found
    """
    try:
        result = AsyncResult(task_id)
        
        # Build response based on task state
        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status.lower(),
            progress=0,
            message="Task queued"
        )
        
        if result.info:
            # Extract progress information from task metadata
            if isinstance(result.info, dict):
                response.progress = result.info.get("progress", 0)
                response.message = result.info.get("message", response.message)
                response.current_step = result.info.get("current_step")
                response.total_steps = result.info.get("total_steps")
                response.estimated_completion = result.info.get("estimated_completion")
                
                # Include any additional metadata
                if "project_id" in result.info:
                    response.project_id = result.info["project_id"]
                if "session_id" in result.info:
                    response.session_id = result.info["session_id"]
                if "scene_id" in result.info:
                    response.scene_id = result.info["scene_id"]
        
        # Handle completed tasks
        if result.successful():
            response.status = "completed"
            response.progress = 100
            response.message = "Task completed successfully"
            response.result = result.result
            
        elif result.failed():
            response.status = "failed"
            response.message = f"Task failed: {str(result.info)}"
            response.error = str(result.info)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {str(e)}"
        )


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Get the detailed result of a completed task.
    
    Args:
        task_id: ID of the completed task
        
    Returns:
        Dictionary containing detailed task results
        
    Raises:
        HTTPException: If task not found or not completed
    """
    try:
        result = AsyncResult(task_id)
        
        if not result.ready():
            raise HTTPException(
                status_code=202,
                detail="Task not yet completed"
            )
        
        if result.failed():
            raise HTTPException(
                status_code=500,
                detail=f"Task failed: {result.info}"
            )
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result.result,
            "completed_at": result.date_done.isoformat() if result.date_done else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task result for {task_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {str(e)}"
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, str]:
    """
    Cancel a running video generation or scene editing task.
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        Dictionary with cancellation confirmation
        
    Raises:
        HTTPException: If cancellation fails
    """
    try:
        from app.core.celery_app import celery_app
        
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        
        logger.info(f"Task {task_id} cancellation requested")
        
        return {
            "task_id": task_id,
            "message": "Task cancellation requested",
            "status": "cancelled"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for video generation service.
    
    Returns:
        Dictionary with service health status
    """
    try:
        # Basic health check - could be expanded to check:
        # - Celery worker availability
        # - S3 connectivity
        # - Convex connectivity
        # - Agent service availability
        
        return {
            "status": "healthy",
            "service": "video-generation",
            "message": "Video generation service is operational"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "video-generation",
            "message": f"Service health check failed: {str(e)}"
        }