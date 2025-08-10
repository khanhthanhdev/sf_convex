from convex import ConvexClient
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

# Types for status enums
SessionStatus = Literal["idle", "queued", "generating", "rendering", "ready", "error"]
SceneStatus = Literal["queued", "generating_code", "rendering", "uploading", "ready", "error"]

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
    s3ChunkKey: Optional[str] = None
    s3ChunkUrl: Optional[str] = None
    s3SourceKey: Optional[str] = None
    checksum: Optional[str] = None
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
            {"name": "by_project", "fields": ["projectId"]}
        ]
    },
    "scenes": {
        "schema": Scene.schema(),
        "indexes": [
            {"name": "by_session", "fields": ["sessionId"]},
            {"name": "by_project", "fields": ["projectId"]}
        ]
    }
}