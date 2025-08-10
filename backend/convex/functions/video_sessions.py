from typing import Dict, Optional
from convex import ConvexClient
from ..schema import SessionStatus
from .utils import create_document, update_document, get_current_timestamp

# Initialize Convex client
client = ConvexClient("your-convex-deployment-url")  # Replace with your actual Convex deployment URL

def create_video_session(
    project_id: str,
    target_fps: int,
    width: int,
    height: int,
    codec: str = "h264",
    audio_hz: int = 44100,
) -> Dict:
    """
    Create a new video session.
    
    Args:
        project_id: ID of the project this session belongs to
        target_fps: Target frames per second for the video
        width: Video width in pixels
        height: Video height in pixels
        codec: Video codec to use (default: h264)
        audio_hz: Audio sample rate in Hz (default: 44100)
        
    Returns:
        The created video session document
    """
    session_data = {
        "projectId": project_id,
        "status": "idle",
        "targetFps": target_fps,
        "width": width,
        "height": height,
        "codec": codec,
        "audioHz": audio_hz,
    }
    
    return create_document("videoSessions", session_data)

def update_video_session_status(
    session_id: str,
    status: SessionStatus,
    error_message: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict:
    """
    Update a video session's status.
    
    Args:
        session_id: ID of the video session to update
        status: New status for the session
        error_message: Optional error message if status is "error"
        job_id: Optional job ID for tracking the processing job
        
    Returns:
        The updated video session document
    """
    updates = {"status": status}
    
    if error_message:
        updates["errorMessage"] = error_message
    if job_id:
        updates["jobId"] = job_id
    
    return update_document("videoSessions", session_id, updates)

def get_video_session(session_id: str) -> Dict:
    """
    Get a video session by ID.
    
    Args:
        session_id: ID of the video session to retrieve
        
    Returns:
        The video session document
    """
    return client.query("getDocument", {"collection": "videoSessions", "id": session_id})

def list_video_sessions(project_id: str) -> list[Dict]:
    """
    List all video sessions for a project.
    
    Args:
        project_id: ID of the project to list sessions for
        
    Returns:
        List of video session documents
    """
    return client.query("listDocuments", {
        "collection": "videoSessions",
        "filter": {"projectId": project_id},
        "orderBy": [{"field": "createdAt", "direction": "desc"}]
    })
