"""
Video generation and editing tasks for AI Video Tutor
"""
import time
import json
import asyncio
import os
from typing import Dict, Any, Optional
from celery import current_task
from app.core.celery_app import celery_app
from app.core.redis_client import redis_client

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

def _load_model(model_name: str):
    """Instantiate a text/VLM model by name using LiteLLM wrapper from agents.
    Fallback to a simple stub that echoes requests if import fails.
    """
    try:
        try:
            from agents.mllm_tools.litellm import LiteLLMWrapper  # type: ignore
        except Exception:
            # Some repos expose it at top-level mllm_tools
            from mllm_tools.litellm import LiteLLMWrapper  # type: ignore
        return LiteLLMWrapper(model_name=model_name, temperature=0.7, print_cost=True, verbose=False, use_langfuse=False)
    except Exception as e:  # pragma: no cover
        class _EchoModel:
            def __call__(self, *args, **kwargs):
                return json.dumps({"warning": f"LiteLLM not available: {e}", "args": str(args)})
        return _EchoModel()

@celery_app.task(bind=True, name="generate_video_task", queue="render")
def generate_video_task(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate video from project data
    
    Args:
        project_data: Dictionary containing:
            - project_id: Project identifier
            - session_id: Video session identifier  
            - scene_count: Number of scenes to generate
            - prompt: Text prompt for video generation
            
    Returns:
        dict: Task result with generated video information
    """
    task_id = self.request.id
    project_id = project_data.get('project_id')
    session_id = project_data.get('session_id')
    scene_count = project_data.get('scene_count', 1)
    prompt = project_data.get('prompt', '')
    
    try:
        # Update initial state
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': scene_count + 2,  # scenes + planning + finalization
                'status': 'Analyzing prompt and planning scenes...',
                'project_id': project_id,
                'session_id': session_id
            }
        )
        
        # Simulate prompt analysis
        time.sleep(2)
        
        # Generate each scene
        scenes = []
        for i in range(scene_count):
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': i + 1,
                    'total': scene_count + 2,
                    'status': f'Generating scene {i + 1} of {scene_count}...',
                    'project_id': project_id,
                    'session_id': session_id
                }
            )
            
            # Simulate scene generation (replace with actual agent calls)
            time.sleep(3)
            
            scene = {
                'scene_id': f"{session_id}_scene_{i}",
                'index': i,
                'title': f'Scene {i + 1}',
                'duration': 5.0,  # seconds
                'status': 'generated',
                's3_url': f'https://bucket.s3.amazonaws.com/scenes/{session_id}_scene_{i}.mp4',
                'source_code_url': f'https://bucket.s3.amazonaws.com/sources/{session_id}_scene_{i}.py'
            }
            scenes.append(scene)
        
        # Finalization
        self.update_state(
            state='PROGRESS',
            meta={
                'current': scene_count + 1,
                'total': scene_count + 2,
                'status': 'Finalizing video session...',
                'project_id': project_id,
                'session_id': session_id
            }
        )
        
        time.sleep(1)
        
        # Final result
        result = {
            'task_id': task_id,
            'project_id': project_id,
            'session_id': session_id,
            'status': 'completed',
            'scene_count': scene_count,
            'scenes': scenes,
            'total_duration': sum(scene['duration'] for scene in scenes),
            'timestamp': time.time()
        }
        
        # Store result in Redis
        redis_client.set_json(f"video_result:{task_id}", result, ex=3600)
        
        return result
        
    except Exception as e:
        # Handle errors
        error_result = {
            'task_id': task_id,
            'project_id': project_id,
            'session_id': session_id,
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }
        
        redis_client.set_json(f"video_result:{task_id}", error_result, ex=3600)
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True, name="edit_scene_task", queue="render")
def edit_scene_task(self, edit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Edit a specific scene based on user prompt
    
    Args:
        edit_data: Dictionary containing:
            - scene_id: Scene identifier to edit
            - project_id: Project identifier
            - edit_prompt: Description of changes to make
            
    Returns:
        dict: Task result with edited scene information
    """
    task_id = self.request.id
    scene_id = edit_data.get('scene_id')
    project_id = edit_data.get('project_id')
    edit_prompt = edit_data.get('edit_prompt', '')
    
    try:
        # Update initial state
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 1,
                'total': 4,
                'status': 'Loading original scene...',
                'scene_id': scene_id,
                'project_id': project_id
            }
        )
        
        time.sleep(1)
        
        # Analyze edit request
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 2,
                'total': 4,
                'status': 'Analyzing edit request...',
                'scene_id': scene_id,
                'project_id': project_id
            }
        )
        
        time.sleep(2)
        
        # Apply changes and re-render
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 3,
                'total': 4,
                'status': 'Re-rendering scene with changes...',
                'scene_id': scene_id,
                'project_id': project_id
            }
        )
        
        time.sleep(4)  # Simulate rendering
        
        # Upload new version
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 4,
                'total': 4,
                'status': 'Uploading new version...',
                'scene_id': scene_id,
                'project_id': project_id
            }
        )
        
        time.sleep(1)
        
        # Result
        result = {
            'task_id': task_id,
            'scene_id': scene_id,
            'project_id': project_id,
            'status': 'completed',
            'edit_prompt': edit_prompt,
            'new_version': 2,  # Increment version
            's3_url': f'https://bucket.s3.amazonaws.com/scenes/{scene_id}_v2.mp4',
            'source_code_url': f'https://bucket.s3.amazonaws.com/sources/{scene_id}_v2.py',
            'timestamp': time.time()
        }
        
        # Store result in Redis
        redis_client.set_json(f"edit_result:{task_id}", result, ex=3600)
        
        return result
        
    except Exception as e:
        error_result = {
            'task_id': task_id,
            'scene_id': scene_id,
            'project_id': project_id,
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }
        
        redis_client.set_json(f"edit_result:{task_id}", error_result, ex=3600)
        raise self.retry(exc=e, countdown=60, max_retries=3)


# ===================== New dedicated adapter-driven tasks =====================

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
        return {"topic": topic, "scene_number": scene_number, "version": version, "error": error, "code": final_code}
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)


@celery_app.task(bind=True, name="combine_videos_task", queue="render")
def combine_videos_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Combine rendered scene videos into a single mp4."""
    topic: str = payload["topic"]
    use_hw: bool = bool(payload.get("use_hardware_acceleration", False))

    try:
        adapter = VideoRenderAdapter()
        output_path = asyncio.run(adapter.combine_videos(topic=topic, use_hardware_acceleration=use_hw))
        return {"topic": topic, "output": output_path}
    except Exception as e:
        raise self.retry(exc=e, countdown=30, max_retries=2)
