from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import json
from .schema import SessionStatus, SceneStatus
from .functions import (
    create_video_session,
    update_video_session_status,
    get_video_session,
    list_video_sessions,
    create_scene,
    update_scene_status,
    get_scene,
    list_scenes,
    get_next_queued_scene,
)

router = APIRouter(prefix="/api/v1", tags=["convex"])

# Video Session Endpoints
@router.post("/sessions/")
async def create_new_session(
    project_id: str,
    target_fps: int,
    width: int,
    height: int,
    codec: str = "h264",
    audio_hz: int = 44100,
):
    """Create a new video session."""
    try:
        session = create_video_session(
            project_id=project_id,
            target_fps=target_fps,
            width=width,
            height=height,
            codec=codec,
            audio_hz=audio_hz,
        )
        return {"status": "success", "data": session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a video session by ID."""
    try:
        session = get_video_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "success", "data": session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/sessions")
async def list_project_sessions(project_id: str):
    """List all video sessions for a project."""
    try:
        sessions = list_video_sessions(project_id)
        return {"status": "success", "data": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/status")
async def update_session_status(
    session_id: str,
    status: SessionStatus,
    error_message: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """Update a video session's status."""
    try:
        updated = update_video_session_status(
            session_id=session_id,
            status=status,
            error_message=error_message,
            job_id=job_id,
        )
        return {"status": "success", "data": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Scene Endpoints
@router.post("/sessions/{session_id}/scenes")
async def add_scene(
    session_id: str,
    project_id: str,
    index: int,
    start_frame: int,
    end_frame: int,
    title: Optional[str] = None,
):
    """Add a new scene to a video session."""
    try:
        scene = create_scene(
            project_id=project_id,
            session_id=session_id,
            index=index,
            start_frame=start_frame,
            end_frame=end_frame,
            title=title,
        )
        return {"status": "success", "data": scene}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenes/{scene_id}")
async def get_scene_by_id(scene_id: str):
    """Get a scene by ID."""
    try:
        scene = get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        return {"status": "success", "data": scene}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/scenes")
async def list_session_scenes(session_id: str):
    """List all scenes for a video session."""
    try:
        scenes = list_scenes(session_id)
        return {"status": "success", "data": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenes/{scene_id}/status")
async def update_scene_status_endpoint(
    scene_id: str,
    status: SceneStatus,
    error_message: Optional[str] = None,
    job_id: Optional[str] = None,
    s3_chunk_key: Optional[str] = None,
    s3_chunk_url: Optional[str] = None,
    s3_source_key: Optional[str] = None,
    checksum: Optional[str] = None,
):
    """Update a scene's status and metadata."""
    try:
        updated = update_scene_status(
            scene_id=scene_id,
            status=status,
            error_message=error_message,
            job_id=job_id,
            s3_chunk_key=s3_chunk_key,
            s3_chunk_url=s3_chunk_url,
            s3_source_key=s3_source_key,
            checksum=checksum,
        )
        return {"status": "success", "data": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenes/next-queued")
async def get_next_queued_scene_endpoint():
    """Get the next scene that's queued for processing."""
    try:
        scene = get_next_queued_scene()
        if not scene:
            return {"status": "success", "data": None, "message": "No queued scenes found"}
        return {"status": "success", "data": scene}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
