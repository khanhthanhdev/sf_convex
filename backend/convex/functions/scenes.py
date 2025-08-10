from typing import Dict, List, Optional
from convex import ConvexClient
from ..schema import SceneStatus
from .utils import create_document, update_document, get_current_timestamp
import os

CONVEX_URL = os.getenv("CONVEX_URL")
# Initialize Convex client
client = ConvexClient(CONVEX_URL)  # Replace with your actual Convex deployment URL

def create_scene(
    project_id: str,
    session_id: str,
    index: int,
    start_frame: int,
    end_frame: int,
    title: Optional[str] = None,
) -> Dict:
    """
    Create a new scene for a video session.
    
    Args:
        project_id: ID of the project
        session_id: ID of the video session this scene belongs to
        index: 0-based ordering of the scene in the session
        start_frame: Starting frame number
        end_frame: Ending frame number (inclusive)
        title: Optional title for the scene
        
    Returns:
        The created scene document
    """
    scene_data = {
        "projectId": project_id,
        "sessionId": session_id,
        "index": index,
        "startFrame": start_frame,
        "endFrame": end_frame,
        "durationInFrames": end_frame - start_frame + 1,
        "status": "queued",
    }
    
    if title:
        scene_data["title"] = title
    
    return create_document("scenes", scene_data)

def update_scene_status(
    scene_id: str,
    status: SceneStatus,
    error_message: Optional[str] = None,
    job_id: Optional[str] = None,
    s3_chunk_key: Optional[str] = None,
    s3_chunk_url: Optional[str] = None,
    s3_source_key: Optional[str] = None,
    checksum: Optional[str] = None,
) -> Dict:
    """
    Update a scene's status and related metadata.
    
    Args:
        scene_id: ID of the scene to update
        status: New status for the scene
        error_message: Optional error message if status is "error"
        job_id: Optional job ID for tracking the processing job
        s3_chunk_key: S3 key for the rendered video chunk
        s3_chunk_url: Public URL for the rendered video chunk
        s3_source_key: S3 key for the source code
        checksum: Checksum of the chunk for cache busting
        
    Returns:
        The updated scene document
    """
    updates = {"status": status}
    
    if error_message:
        updates["errorMessage"] = error_message
    if job_id:
        updates["jobId"] = job_id
    if s3_chunk_key:
        updates["s3ChunkKey"] = s3_chunk_key
    if s3_chunk_url:
        updates["s3ChunkUrl"] = s3_chunk_url
    if s3_source_key:
        updates["s3SourceKey"] = s3_source_key
    if checksum:
        updates["checksum"] = checksum
    
    return update_document("scenes", scene_id, updates)

def get_scene(scene_id: str) -> Dict:
    """
    Get a scene by ID.
    
    Args:
        scene_id: ID of the scene to retrieve
        
    Returns:
        The scene document
    """
    return client.query("getDocument", {"collection": "scenes", "id": scene_id})

def list_scenes(session_id: str) -> List[Dict]:
    """
    List all scenes for a video session.
    
    Args:
        session_id: ID of the video session
        
    Returns:
        List of scene documents, ordered by index
    """
    return client.query("listDocuments", {
        "collection": "scenes",
        "filter": {"sessionId": session_id},
        "orderBy": [{"field": "index", "direction": "asc"}]
    })

def get_next_queued_scene() -> Optional[Dict]:
    """
    Get the next scene that's queued for processing.
    
    Returns:
        The next queued scene document, or None if none are queued
    """
    # Atomically claim the next scene to avoid duplicate processing by multiple workers.
    # Implement server-side mutation 'scenes/claimNextQueued' to find-and-update in one transaction.
    return client.mutation("scenes/claimNextQueued", {})
