from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    topic: str = Field(..., description="Video topic")
    description: str = Field(..., description="High-level description")
    session_id: str = Field(..., description="Session identifier")
    model: str = Field(..., description="Planner/scene model name")
    helper_model: Optional[str] = Field(None, description="Helper model name")


class ImplementRequest(BaseModel):
    topic: str
    description: str
    plan_xml: str = Field(..., description="<SCENE_OUTLINE>...</SCENE_OUTLINE>")
    session_id: str
    model: str
    helper_model: Optional[str] = None


class CodeGenRequest(BaseModel):
    topic: str
    description: str
    scene_outline: str
    implementation: str
    scene_number: int = Field(..., ge=1)
    session_id: Optional[str] = None
    model: str
    helper_model: Optional[str] = None


class RenderSceneRequest(BaseModel):
    topic: str
    code: str
    scene_number: int = Field(..., ge=1)
    version: int = Field(0, ge=0)
    session_id: Optional[str] = None
    quality: Optional[str] = Field(None, description="preview/low/medium/high/production")


class CombineRequest(BaseModel):
    topic: str
    use_hardware_acceleration: Optional[bool] = False


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    meta: Optional[Dict[str, Any]] = None


class CodeResponse(BaseModel):
    code: str
    raw_response: Optional[str] = None


class ImplementResponse(BaseModel):
    scene_plans: List[str]


