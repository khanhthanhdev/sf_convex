"""
Background tasks for video generation and scene editing orchestration.

This module implements the orchestration functions that coordinate:
- Video generation pipeline
- Scene editing workflow
- S3 uploads and Convex synchronization
- Error handling and retry logic
"""
import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from celery import current_task
from app.core.celery_app import celery_app
from app.core.settings import settings
from app.services.convex_s3_sync import ConvexS3Sync, AssetSyncRequest, SyncOperation
from agents.src.storage.s3_storage import S3Storage
from backend.convex.types.schema import AssetType

# Configure logging
logger = logging.getLogger(__name__)

# Ensure project root is on path for importing agents
try:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except Exception as e:
    logger.warning(f"Failed to add project root to path: {e}")


def _load_video_generator():
    """Load and configure the VideoGenerator from agents module."""
    try:
        from agents.generate_video import VideoGenerator
        from agents.mllm_tools.litellm import LiteLLMWrapper
        
        # Default model configuration
        planner_model = LiteLLMWrapper(
            model_name="gpt-4o-mini",
            temperature=0.7,
            print_cost=True,
            verbose=False,
            use_langfuse=False
        )
        
        return VideoGenerator(
            planner_model=planner_model,
            output_dir="output",
            verbose=True,
            use_rag=False,
            use_visual_fix_code=False,
            max_scene_concurrency=3
        )
    except Exception as e:
        logger.error(f"Failed to load VideoGenerator: {e}")
        raise


def _get_s3_storage() -> S3Storage:
    """Get configured S3Storage instance."""
    if not settings.S3_BUCKET:
        raise ValueError("S3_BUCKET not configured in settings")
    
    return S3Storage(
        bucket=settings.S3_BUCKET,
        base_prefix=settings.S3_BASE_PREFIX
    )


def _get_convex_sync() -> ConvexS3Sync:
    """Get configured ConvexS3Sync instance."""
    return ConvexS3Sync()


def _update_task_progress(
    current: int,
    total: int,
    message: str,
    **extra_meta
) -> None:
    """Update task progress with metadata."""
    if current_task:
        progress = int((current / total) * 100) if total > 0 else 0
        estimated_completion = None
        
        # Calculate estimated completion time
        if progress > 0:
            elapsed = time.time() - current_task.request.eta if current_task.request.eta else 0
            if elapsed > 0:
                total_estimated = (elapsed / progress) * 100
                remaining = total_estimated - elapsed
                estimated_completion = (datetime.now() + timedelta(seconds=remaining)).isoformat()
        
        meta = {
            "current": current,
            "total": total,
            "progress": progress,
            "message": message,
            "current_step": message,
            "total_steps": total,
            "estimated_completion": estimated_completion,
            **extra_meta
        }
        
        current_task.update_state(
            state='PROGRESS',
            meta=meta
        )


@celery_app.task(bind=True, name="generate_video_orchestration_task", queue="render")
def generate_video_orchestration_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate the complete video generation pipeline.
    
    This function coordinates:
    1. Core video generation using existing agents
    2. Video chunking into individual scenes
    3. S3 upload for video chunks and source code
    4. Convex database updates with S3 URLs and status
    
    Args:
        task_data: Dictionary containing generation parameters
        
    Returns:
        Dictionary with generation results and S3 URLs
    """
    task_id = self.request.id
    project_id = task_data.get('project_id')
    content_prompt = task_data.get('content_prompt')
    scene_count = task_data.get('scene_count', 3)
    session_id = task_data.get('session_id') or str(uuid.uuid4())
    
    start_time = time.time()
    
    try:
        logger.info(f"Starting video generation orchestration for project {project_id}")
        
        # Initialize services
        s3_storage = _get_s3_storage()
        convex_sync = _get_convex_sync()
        video_generator = _load_video_generator()
        
        # Step 1: Initialize and plan scenes
        _update_task_progress(
            1, 6, "Analyzing prompt and planning scenes...",
            project_id=project_id, session_id=session_id
        )
        
        # Generate scene outline
        scene_outline = video_generator.generate_scene_outline(
            topic=content_prompt,
            description=f"Educational video with {scene_count} scenes",
            session_id=session_id
        )
        
        # Step 2: Generate implementation plans
        _update_task_progress(
            2, 6, "Generating detailed scene implementations...",
            project_id=project_id, session_id=session_id
        )
        
        implementation_plans = asyncio.run(
            video_generator.generate_scene_implementation_concurrently(
                topic=content_prompt,
                description=f"Educational video with {scene_count} scenes",
                plan=scene_outline,
                session_id=session_id
            )
        )
        
        # Step 3: Generate and render scenes
        _update_task_progress(
            3, 6, f"Rendering {len(implementation_plans)} scenes...",
            project_id=project_id, session_id=session_id
        )
        
        # Render all scenes
        asyncio.run(
            video_generator.render_video_fix_code(
                topic=content_prompt,
                description=f"Educational video with {scene_count} scenes",
                scene_outline=scene_outline,
                implementation_plans=implementation_plans,
                max_retries=3,
                session_id=session_id
            )
        )
        
        # Step 4: Upload individual scene assets to S3
        _update_task_progress(
            4, 6, "Uploading scene assets to S3...",
            project_id=project_id, session_id=session_id
        )
        
        scene_results = []
        file_prefix = content_prompt.lower().replace(' ', '_')[:50]
        
        for i, implementation_plan in enumerate(implementation_plans):
            scene_number = i + 1
            scene_id = f"{session_id}_scene_{scene_number}"
            
            try:
                # Upload scene assets
                scene_assets = _upload_scene_assets(
                    s3_storage=s3_storage,
                    convex_sync=convex_sync,
                    file_prefix=file_prefix,
                    scene_number=scene_number,
                    scene_id=scene_id,
                    session_id=session_id
                )
                
                scene_result = {
                    "scene_id": scene_id,
                    "scene_number": scene_number,
                    "title": f"Scene {scene_number}",
                    "duration": 10.0,  # Default duration, could be calculated
                    "status": "completed",
                    **scene_assets
                }
                
                scene_results.append(scene_result)
                logger.info(f"Successfully processed scene {scene_number}")
                
            except Exception as e:
                logger.error(f"Failed to process scene {scene_number}: {e}")
                scene_results.append({
                    "scene_id": scene_id,
                    "scene_number": scene_number,
                    "title": f"Scene {scene_number}",
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Step 5: Combine videos and upload combined assets
        _update_task_progress(
            5, 6, "Combining scenes and creating final video...",
            project_id=project_id, session_id=session_id
        )
        
        combined_assets = {}
        try:
            # Combine videos using the video generator
            video_generator.combine_videos(content_prompt)
            
            # Upload combined video
            combined_video_path = Path(video_generator.output_dir) / file_prefix / f"{file_prefix}_combined.mp4"
            if combined_video_path.exists():
                combined_key = f"{session_id}/combined_video.mp4"
                s3_key = s3_storage.upload_file(str(combined_video_path), combined_key)
                combined_assets["combined_video"] = s3_storage.url_for(s3_key)
                
                # Create and upload manifest
                manifest_data = {
                    "session_id": session_id,
                    "project_id": project_id,
                    "scene_count": len(scene_results),
                    "total_duration": sum(s.get("duration", 0) for s in scene_results),
                    "scenes": scene_results,
                    "generated_at": datetime.now().isoformat()
                }
                
                manifest_key = f"{session_id}/manifest.json"
                manifest_s3_key = s3_storage.write_manifest(session_id, manifest_data)
                combined_assets["manifest"] = s3_storage.url_for(manifest_s3_key)
                
        except Exception as e:
            logger.error(f"Failed to combine videos: {e}")
            combined_assets["error"] = str(e)
        
        # Step 6: Update Convex with session-level assets
        _update_task_progress(
            6, 6, "Finalizing and updating database...",
            project_id=project_id, session_id=session_id
        )
        
        if combined_assets and "error" not in combined_assets:
            try:
                asyncio.run(
                    convex_sync.sync_session_assets(session_id, combined_assets)
                )
            except Exception as e:
                logger.error(f"Failed to sync session assets: {e}")
        
        # Prepare final result
        generation_time = time.time() - start_time
        
        result = {
            "task_id": task_id,
            "project_id": project_id,
            "session_id": session_id,
            "status": "completed",
            "scene_count": len(scene_results),
            "scenes": scene_results,
            "total_duration": sum(s.get("duration", 0) for s in scene_results),
            "s3_combined_video_url": combined_assets.get("combined_video"),
            "s3_manifest_url": combined_assets.get("manifest"),
            "generation_time_seconds": generation_time,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Video generation completed for project {project_id} in {generation_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Video generation orchestration failed: {e}")
        
        # Return error result
        error_result = {
            "task_id": task_id,
            "project_id": project_id,
            "session_id": session_id,
            "status": "error",
            "error": str(e),
            "generation_time_seconds": time.time() - start_time,
            "completed_at": datetime.now().isoformat()
        }
        
        raise self.retry(exc=e, countdown=60, max_retries=2)


@celery_app.task(bind=True, name="edit_scene_orchestration_task", queue="render")
def edit_scene_orchestration_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate scene editing workflow.
    
    This function coordinates:
    1. Retrieving original Manim source code from S3
    2. Calling Code Generation Agent with edit prompt and MCP context
    3. Re-rendering the modified scene
    4. Uploading new version to S3
    5. Updating Convex with new URLs and status
    
    Args:
        task_data: Dictionary containing edit parameters
        
    Returns:
        Dictionary with edit results and new S3 URLs
    """
    task_id = self.request.id
    scene_id = task_data.get('scene_id')
    project_id = task_data.get('project_id')
    edit_prompt = task_data.get('edit_prompt')
    session_id = task_data.get('session_id')
    
    start_time = time.time()
    
    try:
        logger.info(f"Starting scene edit orchestration for scene {scene_id}")
        
        # Initialize services
        s3_storage = _get_s3_storage()
        convex_sync = _get_convex_sync()
        video_generator = _load_video_generator()
        
        # Step 1: Retrieve original source code from S3
        _update_task_progress(
            1, 5, "Retrieving original scene source code...",
            scene_id=scene_id, project_id=project_id
        )
        
        original_source_code = _retrieve_scene_source_code(s3_storage, scene_id)
        
        # Step 2: Generate modified code using edit prompt
        _update_task_progress(
            2, 5, "Generating modified scene code...",
            scene_id=scene_id, project_id=project_id
        )
        
        # Extract scene number from scene_id
        scene_number = _extract_scene_number(scene_id)
        
        # Use the code generator to modify the scene
        modified_code, generation_log = video_generator.code_generator.fix_code_errors(
            implementation_plan=f"Edit request: {edit_prompt}",
            code=original_source_code,
            error=f"User requested changes: {edit_prompt}",
            scene_trace_id=str(uuid.uuid4()),
            topic=f"Scene {scene_number} Edit",
            scene_number=scene_number,
            session_id=session_id or str(uuid.uuid4())
        )
        
        # Step 3: Render the modified scene
        _update_task_progress(
            3, 5, "Rendering modified scene...",
            scene_id=scene_id, project_id=project_id
        )
        
        # Create temporary directory for rendering
        temp_dir = Path("temp_render") / scene_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write modified code to temporary file
        temp_code_file = temp_dir / f"scene_{scene_number}_modified.py"
        with open(temp_code_file, 'w') as f:
            f.write(modified_code)
        
        # Render the scene (simplified - in practice would use full rendering pipeline)
        rendered_assets = _render_modified_scene(
            code_file=temp_code_file,
            scene_number=scene_number,
            output_dir=temp_dir
        )
        
        # Step 4: Upload new version to S3
        _update_task_progress(
            4, 5, "Uploading new scene version to S3...",
            scene_id=scene_id, project_id=project_id
        )
        
        # Determine new version number
        new_version = _get_next_scene_version(scene_id)
        
        # Upload modified assets
        new_scene_assets = _upload_modified_scene_assets(
            s3_storage=s3_storage,
            convex_sync=convex_sync,
            scene_id=scene_id,
            version=new_version,
            assets=rendered_assets,
            modified_code=modified_code
        )
        
        # Step 5: Update Convex with new version
        _update_task_progress(
            5, 5, "Updating database with new version...",
            scene_id=scene_id, project_id=project_id
        )
        
        # Sync with Convex
        asyncio.run(
            convex_sync.sync_scene_assets(
                scene_id,
                new_scene_assets,
                {
                    "version": new_version,
                    "edit_prompt": edit_prompt,
                    "modified_at": datetime.now().isoformat()
                }
            )
        )
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
        
        # Prepare result
        edit_time = time.time() - start_time
        
        result = {
            "task_id": task_id,
            "scene_id": scene_id,
            "project_id": project_id,
            "status": "completed",
            "edit_prompt": edit_prompt,
            "new_version": new_version,
            **new_scene_assets,
            "changes_made": ["Code modified based on edit prompt"],
            "render_attempts": 1,
            "edit_time_seconds": edit_time,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Scene edit completed for {scene_id} in {edit_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Scene edit orchestration failed: {e}")
        
        error_result = {
            "task_id": task_id,
            "scene_id": scene_id,
            "project_id": project_id,
            "status": "error",
            "error": str(e),
            "edit_time_seconds": time.time() - start_time,
            "completed_at": datetime.now().isoformat()
        }
        
        raise self.retry(exc=e, countdown=30, max_retries=2)


# Helper functions

def _upload_scene_assets(
    s3_storage: S3Storage,
    convex_sync: ConvexS3Sync,
    file_prefix: str,
    scene_number: int,
    scene_id: str,
    session_id: str
) -> Dict[str, str]:
    """Upload scene assets to S3 and sync with Convex."""
    assets = {}
    
    # Define expected asset paths
    scene_dir = Path("output") / file_prefix / f"scene{scene_number}"
    
    # Video file
    video_files = list(scene_dir.glob("*.mp4"))
    if video_files:
        video_key = f"{session_id}/scenes/scene_{scene_number}.mp4"
        s3_key, s3_url = s3_storage.upload_with_convex_sync(
            str(video_files[0]),
            video_key,
            convex_sync,
            scene_id,
            "scene",
            AssetType.VIDEO_CHUNK
        )
        assets["s3_video_url"] = s3_url
    
    # Source code
    code_dir = scene_dir / "code"
    code_files = list(code_dir.glob("*.py")) if code_dir.exists() else []
    if code_files:
        # Get the latest version
        latest_code = max(code_files, key=lambda p: p.stat().st_mtime)
        source_key = f"{session_id}/sources/scene_{scene_number}.py"
        s3_key, s3_url = s3_storage.upload_with_convex_sync(
            str(latest_code),
            source_key,
            convex_sync,
            scene_id,
            "scene",
            AssetType.SOURCE_CODE
        )
        assets["s3_source_url"] = s3_url
    
    return assets


def _retrieve_scene_source_code(s3_storage: S3Storage, scene_id: str) -> str:
    """Retrieve original source code for a scene from S3."""
    # This is a simplified implementation
    # In practice, you'd query Convex to get the S3 key for the source code
    # then download it from S3
    
    # For now, return a placeholder
    return """
from manim import *

class SceneExample(Scene):
    def construct(self):
        text = Text("Original Scene")
        self.play(Write(text))
        self.wait(2)
"""


def _extract_scene_number(scene_id: str) -> int:
    """Extract scene number from scene ID."""
    # Assuming scene_id format like "session_id_scene_1"
    parts = scene_id.split('_')
    for i, part in enumerate(parts):
        if part == "scene" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 1  # Default fallback


def _render_modified_scene(
    code_file: Path,
    scene_number: int,
    output_dir: Path
) -> Dict[str, Path]:
    """Render a modified scene and return asset paths."""
    # This is a simplified implementation
    # In practice, you'd use the full Manim rendering pipeline
    
    # Create dummy output files for demonstration
    video_file = output_dir / f"scene_{scene_number}_modified.mp4"
    thumbnail_file = output_dir / f"scene_{scene_number}_thumbnail.png"
    
    # Create empty files (in practice these would be actual rendered assets)
    video_file.touch()
    thumbnail_file.touch()
    
    return {
        "video": video_file,
        "thumbnail": thumbnail_file
    }


def _get_next_scene_version(scene_id: str) -> int:
    """Get the next version number for a scene."""
    # This would typically query Convex to get the current version
    # For now, return a simple increment
    return 2


def _upload_modified_scene_assets(
    s3_storage: S3Storage,
    convex_sync: ConvexS3Sync,
    scene_id: str,
    version: int,
    assets: Dict[str, Path],
    modified_code: str
) -> Dict[str, str]:
    """Upload modified scene assets to S3."""
    uploaded_assets = {}
    
    # Upload video
    if "video" in assets and assets["video"].exists():
        video_key = f"{scene_id}/v{version}/video.mp4"
        s3_key, s3_url = s3_storage.upload_with_convex_sync(
            str(assets["video"]),
            video_key,
            convex_sync,
            scene_id,
            "scene",
            AssetType.VIDEO_CHUNK
        )
        uploaded_assets["s3_video_url"] = s3_url
    
    # Upload source code
    temp_code_file = Path(f"temp_{scene_id}_v{version}.py")
    try:
        with open(temp_code_file, 'w') as f:
            f.write(modified_code)
        
        source_key = f"{scene_id}/v{version}/source.py"
        s3_key, s3_url = s3_storage.upload_with_convex_sync(
            str(temp_code_file),
            source_key,
            convex_sync,
            scene_id,
            "scene",
            AssetType.SOURCE_CODE
        )
        uploaded_assets["s3_source_url"] = s3_url
        
    finally:
        # Clean up temp file
        if temp_code_file.exists():
            temp_code_file.unlink()
    
    # Upload thumbnail if available
    if "thumbnail" in assets and assets["thumbnail"].exists():
        thumbnail_key = f"{scene_id}/v{version}/thumbnail.png"
        s3_key, s3_url = s3_storage.upload_with_convex_sync(
            str(assets["thumbnail"]),
            thumbnail_key,
            convex_sync,
            scene_id,
            "scene",
            AssetType.THUMBNAIL
        )
        uploaded_assets["s3_thumbnail_url"] = s3_url
    
    return uploaded_assets