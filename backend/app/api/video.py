"""
Low-level video agent operations API.

This module provides direct access to individual video generation components:
- Scene planning
- Implementation generation  
- Code generation
- Individual scene rendering
- Video combination

These are building blocks used by the higher-level video generation workflows
but can also be used independently for fine-grained control.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from app.schemas.video import (
    PlanRequest,
    ImplementRequest,
    CodeGenRequest,
    RenderSceneRequest,
    CombineRequest,
    JobStatusResponse,
    CodeResponse,
    ImplementResponse,
)
from app.tasks.video_tasks import (
    plan_scenes_task,
    implement_scenes_task,
    codegen_task,
    render_scene_task,
    combine_videos_task,
)

router = APIRouter(prefix="/video/agents", tags=["video-agents"])


@router.post("/plan", response_model=JobStatusResponse)
def plan_scenes(req: PlanRequest):
    """
    Generate scene outline using the planning agent.
    
    This is a low-level operation that creates a structured plan
    for video scenes based on topic and description.
    """
    task = plan_scenes_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/implement", response_model=JobStatusResponse)
def implement_scenes(req: ImplementRequest):
    """
    Generate detailed implementation plans for scenes.
    
    Takes a scene outline and creates detailed implementation
    plans for each individual scene.
    """
    task = implement_scenes_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/code", response_model=JobStatusResponse)
def generate_code(req: CodeGenRequest):
    """
    Generate Manim code for a specific scene.
    
    Creates executable Manim code based on scene outline
    and implementation plan.
    """
    task = codegen_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/render/scene", response_model=JobStatusResponse)
def render_scene(req: RenderSceneRequest):
    """
    Render a single scene from Manim code.
    
    Takes Manim code and renders it to a video file,
    with optional quality and version parameters.
    """
    task = render_scene_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/render/combine", response_model=JobStatusResponse)
def combine_videos(req: CombineRequest):
    """
    Combine multiple scene videos into a single video.
    
    Takes rendered scene videos and combines them into
    a final cohesive video with optional hardware acceleration.
    """
    task = combine_videos_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Get the status of any agent task.
    
    Returns current status, progress information, and results
    for any video agent operation.
    """
    result = AsyncResult(job_id)
    return JobStatusResponse(
        job_id=job_id, 
        state=result.status, 
        meta=result.info if result.info else None
    )


