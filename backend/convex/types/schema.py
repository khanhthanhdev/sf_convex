from convex import ConvexClient
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

# Types for status enums
SessionStatus = Literal["idle", "queued", "generating", "rendering", "ready", "error"]
SceneStatus = Literal["queued", "generating_code", "rendering", "uploading", "ready", "error"]
AssetStatus = Literal["pending", "uploading", "ready", "error"]

# Asset type enumeration
class AssetType(str, Enum):
    VIDEO_CHUNK = "video_chunk"
    SOURCE_CODE = "source_code"
    THUMBNAIL = "thumbnail"
    SUBTITLE = "subtitle"
    COMBINED_VIDEO = "combined_video"
    MANIFEST = "manifest"

# URL type enumeration
class UrlType(str, Enum):
    PUBLIC = "public"
    PRESIGNED = "presigned"

# S3 Asset models
class S3Asset(BaseModel):
    s3Key: str
    s3Url: str
    contentType: str
    size: int
    checksum: str
    uploadedAt: float

class AssetReference(BaseModel):
    id: str
    entityId: str  # scene_id or session_id
    entityType: Literal["scene", "session"]
    assetType: AssetType
    s3Key: str
    s3Url: str
    contentType: str
    size: int
    checksum: str
    version: int
    createdAt: float
    metadata: Optional[Dict[str, Any]] = None

# Define models using Pydantic for type safety
class User(BaseModel):
    externalId: str
    name: Optional[str] = None
    email: Optional[str] = None
    createdAt: float

class Project(BaseModel):
    ownerId: str  # Reference to users._id
    title: str
    description: Optional[str] = None
    createdAt: float
    updatedAt: float

class VideoSession(BaseModel):
    projectId: str  # Reference to projects._id
    status: SessionStatus
    targetFps: int
    width: int
    height: int
    codec: str
    audioHz: int
    durationInFrames: Optional[int] = None
    errorMessage: Optional[str] = None
    
    # Combined Video Assets
    combinedVideoAsset: Optional[S3Asset] = None
    combinedSubtitleAsset: Optional[S3Asset] = None
    
    # Manifest Reference
    manifestAsset: Optional[S3Asset] = None
    
    # Asset Summary
    totalAssetSize: int = 0
    assetCount: int = 0
    
    # Asset Status Tracking
    assetsStatus: AssetStatus = "pending"
    assetsErrorMessage: Optional[str] = None
    
    jobId: Optional[str] = None
    version: int = 1
    createdAt: float
    updatedAt: float

class Scene(BaseModel):
    projectId: str  # Reference to projects._id
    sessionId: str  # Reference to videoSessions._id
    index: int  # 0-based ordering
    startFrame: int
    endFrame: int  # inclusive end
    durationInFrames: int
    title: Optional[str] = None
    status: SceneStatus
    errorMessage: Optional[str] = None
    
    # Legacy fields (kept for backward compatibility)
    s3ChunkKey: Optional[str] = None
    s3ChunkUrl: Optional[str] = None
    s3SourceKey: Optional[str] = None
    checksum: Optional[str] = None
    
    # Enhanced S3 Asset References
    videoAsset: Optional[S3Asset] = None
    sourceCodeAsset: Optional[S3Asset] = None
    thumbnailAsset: Optional[S3Asset] = None
    subtitleAsset: Optional[S3Asset] = None
    
    # Asset Status Tracking
    assetsStatus: AssetStatus = "pending"
    assetsErrorMessage: Optional[str] = None
    
    # Version Management
    assetVersion: int = 1
    previousAssetVersions: List[S3Asset] = []
    
    jobId: Optional[str] = None
    version: int = 1
    updatedAt: float

# Initialize Convex client
client = ConvexClient("your-convex-deployment-url")  # Replace with your actual Convex deployment URL

# Define database schema
schema = {
    "users": {
        "schema": User.schema(),
        "indexes": [
            {"name": "by_externalId", "fields": ["externalId"]}
        ]
    },
    "projects": {
        "schema": Project.schema(),
        "indexes": [
            {"name": "by_owner", "fields": ["ownerId"]}
        ]
    },
    "videoSessions": {
        "schema": VideoSession.schema(),
        "indexes": [
            {"name": "by_project", "fields": ["projectId"]},
            {"name": "by_assets_status", "fields": ["assetsStatus"]},
            {"name": "by_project_and_status", "fields": ["projectId", "status"]}
        ]
    },
    "scenes": {
        "schema": Scene.schema(),
        "indexes": [
            {"name": "by_session", "fields": ["sessionId"]},
            {"name": "by_project", "fields": ["projectId"]},
            {"name": "by_assets_status", "fields": ["assetsStatus"]},
            {"name": "by_session_and_index", "fields": ["sessionId", "index"]},
            {"name": "by_project_and_assets_status", "fields": ["projectId", "assetsStatus"]}
        ]
    },
    "assetReferences": {
        "schema": AssetReference.schema(),
        "indexes": [
            {"name": "by_entity", "fields": ["entityId", "entityType"]},
            {"name": "by_entity_and_type", "fields": ["entityId", "entityType", "assetType"]},
            {"name": "by_s3_key", "fields": ["s3Key"]},
            {"name": "by_asset_type", "fields": ["assetType"]},
            {"name": "by_created_at", "fields": ["createdAt"]},
            {"name": "by_version", "fields": ["entityId", "version"]}
        ]
    }
}