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

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/plan", response_model=JobStatusResponse)
def plan(req: PlanRequest):
    task = plan_scenes_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    result = AsyncResult(job_id)
    return JobStatusResponse(job_id=job_id, state=result.status, meta=result.info if result.info else None)


@router.post("/implement", response_model=JobStatusResponse)
def implement(req: ImplementRequest):
    task = implement_scenes_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/code", response_model=JobStatusResponse)
def codegen(req: CodeGenRequest):
    task = codegen_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/render/scene", response_model=JobStatusResponse)
def render_scene(req: RenderSceneRequest):
    task = render_scene_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


@router.post("/render/combine", response_model=JobStatusResponse)
def combine(req: CombineRequest):
    task = combine_videos_task.delay(req.model_dump())
    return JobStatusResponse(job_id=task.id, state="PENDING")


