"""
Pydantic schemas for video generation and scene editing endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for AI models used in video generation."""
    planner_model: str = Field(..., description="Model for scene planning")
    scene_model: Optional[str] = Field(None, description="Model for scene generation (defaults to planner_model)")
    helper_model: Optional[str] = Field(None, description="Helper model for additional tasks")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens for generation")
    use_rag: bool = Field(False, description="Whether to use RAG for context")
    use_visual_fix: bool = Field(False, description="Whether to use visual feedback for code fixing")


class RenderConfig(BaseModel):
    """Configuration for video rendering."""
    quality: str = Field("medium", description="Render quality: preview/low/medium/high/production")
    resolution: str = Field("1080p", description="Video resolution: 720p/1080p/1440p/4k")
    fps: int = Field(30, ge=15, le=60, description="Frames per second")
    max_scene_duration: int = Field(30, ge=5, le=300, description="Maximum scene duration in seconds")
    use_hardware_acceleration: bool = Field(False, description="Use hardware acceleration for rendering")
    max_retries: int = Field(3, ge=1, le=10, description="Maximum retry attempts for failed renders")


class S3Config(BaseModel):
    """Configuration for S3 storage."""
    bucket_name: Optional[str] = Field(None, description="S3 bucket name (uses default if not provided)")
    base_prefix: str = Field("", description="Base prefix for S3 keys")
    public_base_url: Optional[str] = Field(None, description="Public base URL for assets (e.g., CloudFront)")
    enable_versioning: bool = Field(True, description="Enable S3 object versioning")
    storage_class: str = Field("STANDARD", description="S3 storage class")


class VideoGenerationRequest(BaseModel):
    """Request schema for video generation."""
    project_id: str = Field(..., description="Project identifier")
    content_prompt: str = Field(..., min_length=10, description="Content prompt for video generation")
    scene_count: int = Field(..., ge=1, le=20, description="Number of scenes to generate")
    session_id: Optional[str] = Field(None, description="Video session identifier (auto-generated if not provided)")
    model_config: Optional[ModelConfig] = Field(None, description="AI model configuration")
    render_config: Optional[RenderConfig] = Field(None, description="Video rendering configuration")
    s3_config: Optional[S3Config] = Field(None, description="S3 storage configuration")
    
    @validator('content_prompt')
    def validate_content_prompt(cls, v):
        if not v.strip():
            raise ValueError('Content prompt cannot be empty')
        return v.strip()


class SceneEditRequest(BaseModel):
    """Request schema for scene editing."""
    scene_id: str = Field(..., description="Scene identifier to edit")
    project_id: str = Field(..., description="Project identifier")
    edit_prompt: str = Field(..., min_length=5, description="Description of changes to make")
    session_id: Optional[str] = Field(None, description="Video session identifier")
    model_config: Optional[ModelConfig] = Field(None, description="AI model configuration")
    render_config: Optional[RenderConfig] = Field(None, description="Video rendering configuration")
    preserve_timing: bool = Field(True, description="Whether to preserve original scene timing")
    version_increment: bool = Field(True, description="Whether to increment version number")
    
    @validator('edit_prompt')
    def validate_edit_prompt(cls, v):
        if not v.strip():
            raise ValueError('Edit prompt cannot be empty')
        return v.strip()


class VideoGenerationResponse(BaseModel):
    """Response schema for video generation."""
    task_id: str = Field(..., description="Background task identifier")
    project_id: str = Field(..., description="Project identifier")
    session_id: Optional[str] = Field(None, description="Video session identifier")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Human-readable status message")
    estimated_duration_minutes: int = Field(..., description="Estimated completion time in minutes")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")


class SceneEditResponse(BaseModel):
    """Response schema for scene editing."""
    task_id: str = Field(..., description="Background task identifier")
    scene_id: str = Field(..., description="Scene identifier being edited")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Human-readable status message")
    estimated_duration_minutes: int = Field(..., description="Estimated completion time in minutes")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")


class TaskStatusResponse(BaseModel):
    """Response schema for task status queries."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current task status")
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Current status message")
    current_step: Optional[str] = Field(None, description="Current processing step")
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    project_id: Optional[str] = Field(None, description="Project identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    scene_id: Optional[str] = Field(None, description="Scene identifier (for edit tasks)")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result (when completed)")
    error: Optional[str] = Field(None, description="Error message (when failed)")
    created_at: Optional[datetime] = Field(None, description="Task creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")


class SceneResult(BaseModel):
    """Result schema for individual scene generation."""
    scene_id: str = Field(..., description="Scene identifier")
    scene_number: int = Field(..., description="Scene number in sequence")
    title: str = Field(..., description="Scene title")
    duration: float = Field(..., description="Scene duration in seconds")
    status: str = Field(..., description="Scene generation status")
    s3_video_url: Optional[str] = Field(None, description="S3 URL for video chunk")
    s3_source_url: Optional[str] = Field(None, description="S3 URL for source code")
    s3_thumbnail_url: Optional[str] = Field(None, description="S3 URL for thumbnail")
    s3_subtitle_url: Optional[str] = Field(None, description="S3 URL for subtitle file")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    render_attempts: int = Field(1, description="Number of render attempts")
    version: int = Field(1, description="Scene version number")


class VideoGenerationResult(BaseModel):
    """Complete result schema for video generation."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    session_id: str = Field(..., description="Video session identifier")
    status: str = Field(..., description="Overall generation status")
    scene_count: int = Field(..., description="Total number of scenes")
    scenes: List[SceneResult] = Field(..., description="Individual scene results")
    total_duration: float = Field(..., description="Total video duration in seconds")
    s3_combined_video_url: Optional[str] = Field(None, description="S3 URL for combined video")
    s3_manifest_url: Optional[str] = Field(None, description="S3 URL for manifest file")
    convex_session_id: Optional[str] = Field(None, description="Convex video session document ID")
    generation_time_seconds: float = Field(..., description="Total generation time")
    completed_at: datetime = Field(..., description="Completion timestamp")


class SceneEditResult(BaseModel):
    """Result schema for scene editing."""
    task_id: str = Field(..., description="Task identifier")
    scene_id: str = Field(..., description="Scene identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Edit status")
    edit_prompt: str = Field(..., description="Edit prompt used")
    new_version: int = Field(..., description="New scene version number")
    s3_video_url: Optional[str] = Field(None, description="S3 URL for new video version")
    s3_source_url: Optional[str] = Field(None, description="S3 URL for updated source code")
    s3_thumbnail_url: Optional[str] = Field(None, description="S3 URL for new thumbnail")
    duration: Optional[float] = Field(None, description="New scene duration in seconds")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made")
    render_attempts: int = Field(1, description="Number of render attempts")
    edit_time_seconds: float = Field(..., description="Time taken for edit")
    completed_at: datetime = Field(..., description="Completion timestamp")


class S3UtilityResponse(BaseModel):
    """Response schema for S3 utility operations."""
    success: bool = Field(..., description="Whether operation succeeded")
    s3_key: Optional[str] = Field(None, description="S3 key for uploaded file")
    s3_url: Optional[str] = Field(None, description="S3 URL for uploaded file")
    public_url: Optional[str] = Field(None, description="Public URL for accessing file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="File content type")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")