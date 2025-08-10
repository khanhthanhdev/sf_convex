"""
Low-level video agent operation tasks.

This module contains Celery tasks that provide direct access to individual
video generation components. These are building blocks used by the 
/api/v1/video/agents/* endpoints for fine-grained control over the video
generation process.

For high-level orchestrated workflows, see video_generation_tasks.py.
"""
import asyncio
import os
from typing import Dict, Any, Optional
from app.core.celery_app import celery_app

# Ensure project root on path for importing 'agents'
try:
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except Exception:
    pass

# Import adapters and model wrapper
from app.adapters.video_adapters import (
    VideoPlannerAdapter,
    CodeGenAdapter,
    VideoRenderAdapter,
)
from backend.convex.functions import update_scene_status as convex_update_scene_status  # type: ignore

def _load_model(model_name: str):
    """Instantiate a text/VLM model by name using LiteLLM wrapper from agents.
    Fallback to a simple stub that echoes requests if import fails.
    """
    try:
        from agents.mllm_tools.litellm import LiteLLMWrapper  # type: ignore
    except Exception:
        # Some repos expose it at top-level mllm_tools
        from mllm_tools.litellm import LiteLLMWrapper  # type: ignore
    return LiteLLMWrapper(model_name=model_name, temperature=0.7, print_cost=True, verbose=False, use_langfuse=False)

# ===================== Low-level agent operation tasks =====================
# These tasks provide direct access to individual video generation components
# and are used by the /api/v1/video/agents/* endpoints for fine-grained control.

@celery_app.task(bind=True, name="plan_scenes_task", queue="render")
def plan_scenes_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate scene outline using EnhancedVideoPlanner via adapter."""
    topic: str = payload["topic"]
    description: str = payload["description"]
    session_id: str = payload["session_id"]
    model_name: str = payload["model"]
    helper_model_name: Optional[str] = payload.get("helper_model")

    try:
        planner_model = _load_model(model_name)
        helper_model = _load_model(helper_model_name) if helper_model_name else None
        adapter = VideoPlannerAdapter(planner_model=planner_model, helper_model=helper_model)
        plan_text = asyncio.run(adapter.generate_scene_outline(topic, description, session_id))
        return {"topic": topic, "session_id": session_id, "plan": plan_text}
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)


@celery_app.task(bind=True, name="implement_scenes_task", queue="render")
def implement_scenes_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate scene implementation plans concurrently using adapter."""
    topic: str = payload["topic"]
    description: str = payload["description"]
    plan_xml: str = payload["plan_xml"]
    session_id: str = payload["session_id"]
    model_name: str = payload["model"]
    helper_model_name: Optional[str] = payload.get("helper_model")

    try:
        planner_model = _load_model(model_name)
        helper_model = _load_model(helper_model_name) if helper_model_name else None
        adapter = VideoPlannerAdapter(planner_model=planner_model, helper_model=helper_model)
        scene_plans = asyncio.run(adapter.generate_implementation(topic, description, plan_xml, session_id))
        return {"topic": topic, "session_id": session_id, "scene_plans": scene_plans, "count": len(scene_plans)}
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)


@celery_app.task(bind=True, name="codegen_task", queue="render")
def codegen_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Manim code from implementation using CodeGenAdapter."""
    topic: str = payload["topic"]
    description: str = payload["description"]
    scene_outline: str = payload["scene_outline"]
    implementation: str = payload["implementation"]
    scene_number: int = int(payload["scene_number"])
    session_id: Optional[str] = payload.get("session_id")
    model_name: str = payload["model"]
    helper_model_name: Optional[str] = payload.get("helper_model")

    try:
        scene_model = _load_model(model_name)
        helper_model = _load_model(helper_model_name) if helper_model_name else None
        adapter = CodeGenAdapter(scene_model=scene_model, helper_model=helper_model)
        code, raw = adapter.generate_code(
            topic=topic,
            description=description,
            scene_outline=scene_outline,
            implementation=implementation,
            scene_number=scene_number,
            session_id=session_id,
        )
        return {"topic": topic, "scene_number": scene_number, "code": code, "raw_response": raw}
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)


@celery_app.task(bind=True, name="render_scene_task", queue="render")
def render_scene_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Render a single scene using VideoRenderAdapter."""
    topic: str = payload["topic"]
    code: str = payload["code"]
    scene_number: int = int(payload["scene_number"])
    version: int = int(payload.get("version", 0))
    session_id: Optional[str] = payload.get("session_id")
    quality: Optional[str] = payload.get("quality")

    try:
        adapter = VideoRenderAdapter()
        final_code, error = asyncio.run(
            adapter.render_scene(
                code=code,
                topic=topic,
                scene_number=scene_number,
                version=version,
                session_id=session_id,
                quality=quality,
            )
        )
        # Include S3 artifact URLs when available
        artifacts: Dict[str, Optional[str]] = {}
        try:
            artifacts = adapter.get_scene_artifacts(topic=topic, scene_number=scene_number, version=version)
        except Exception as e:
            # Non-fatal; keep compatibility
            artifacts = {"s3_video_url": None, "s3_srt_url": None, "s3_code_url": None}
            if os.getenv("DEBUG", "1") in {"1", "true", "yes"}:
                print(f"render_scene_task: get_scene_artifacts failed: {e}")

        # Persist S3 URL to Convex when possible
        try:
            scene_id: Optional[str] = payload.get("scene_id")
            s3_video_url = artifacts.get("s3_video_url")
            if scene_id and s3_video_url:
                # Update the scene with the S3 video URL; mark as ready if no error
                convex_update_scene_status(
                    scene_id=scene_id,
                    status=("ready" if not error else "error"),  # type: ignore[arg-type]
                    error_message=(error or None),
                    s3_chunk_url=s3_video_url,
                )
        except Exception as e:
            if os.getenv("DEBUG", "1") in {"1", "true", "yes"}:
                print(f"render_scene_task: failed to update Convex with S3 URL: {e}")

        return {
            "topic": topic,
            "scene_number": scene_number,
            "version": version,
            "error": error,
            "code": final_code,
            **artifacts,
        }
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)


@celery_app.task(bind=True, name="combine_videos_task", queue="render")
def combine_videos_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Combine rendered scene videos into a single mp4."""
    topic: str = payload["topic"]
    use_hw: bool = bool(payload.get("use_hardware_acceleration", False))

    try:
        adapter = VideoRenderAdapter()
        output_url_or_path = asyncio.run(adapter.combine_videos(topic=topic, use_hardware_acceleration=use_hw))
        # Maintain backward compatibility with 'output' while adding 'output_url'
        return {"topic": topic, "output": output_url_or_path, "output_url": output_url_or_path}
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)
