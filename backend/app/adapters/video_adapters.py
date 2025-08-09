"""
Thin adapter layer to bridge FastAPI tasks/endpoints with core agents.

These adapters are intentionally small and dependency-injected. They avoid
hard dependencies on specific model wrappers so the API server can decide
which LLM/VLM providers to use at runtime.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Best-effort import of project root so that `agents` package is importable
try:
    # If running inside FastAPI app process, PYTHONPATH may not include project root
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in os.sys.path:
        os.sys.path.append(str(PROJECT_ROOT))
except Exception:
    pass

# Core agents
try:
    from agents.src.core.video_planner import EnhancedVideoPlanner
    from agents.src.core.code_generator import CodeGenerator
    from agents.src.core.video_renderer import OptimizedVideoRenderer
except Exception as e:  # pragma: no cover - defensive import
    raise ImportError(
        f"Failed to import core agents from 'agents'. Ensure PYTHONPATH includes project root. Error: {e}"
    )

# Settings (optional fields with sensible defaults)
try:
    from app.core.settings import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # Adapters can be created without global settings


def _get_setting(attr: str, default: Any) -> Any:
    """Safely read setting attribute with a default fallback."""
    if settings is None:
        return default
    return getattr(settings, attr, default)


def _sanitize_topic_to_prefix(topic: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", topic.lower())


class VideoPlannerAdapter:
    """Adapter for planning and implementation generation."""

    def __init__(
        self,
        *,
        planner_model: Any,
        helper_model: Optional[Any] = None,
        output_dir: Optional[str] = None,
        use_context_learning: Optional[bool] = None,
        context_learning_path: Optional[str] = None,
        use_rag: Optional[bool] = None,
        chroma_db_path: Optional[str] = None,
        manim_docs_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_langfuse: Optional[bool] = None,
        max_scene_concurrency: Optional[int] = None,
        max_step_concurrency: Optional[int] = None,
        enable_caching: Optional[bool] = None,
    ) -> None:
        if planner_model is None:
            raise ValueError("planner_model is required")

        self.planner = EnhancedVideoPlanner(
            planner_model=planner_model,
            helper_model=helper_model or planner_model,
            output_dir=output_dir or _get_setting("OUTPUT_DIR", "output"),
            print_response=_get_setting("DEBUG", True),
            use_context_learning=bool(_get_setting("USE_CONTEXT_LEARNING", use_context_learning or False)),
            context_learning_path=context_learning_path or _get_setting("CONTEXT_LEARNING_PATH", "data/context_learning"),
            use_rag=bool(_get_setting("USE_RAG", use_rag or False)),
            session_id=None,
            chroma_db_path=chroma_db_path or _get_setting("CHROMA_DB_PATH", "data/rag/chroma_db"),
            manim_docs_path=manim_docs_path or _get_setting("MANIM_DOCS_PATH", "data/rag/manim_docs"),
            embedding_model=embedding_model or _get_setting("EMBEDDING_MODEL", "text-embedding-ada-002"),
            use_langfuse=bool(_get_setting("USE_LANGFUSE", use_langfuse or True)),
            max_scene_concurrency=int(_get_setting("MAX_SCENE_CONCURRENCY", max_scene_concurrency or 3)),
            max_step_concurrency=int(_get_setting("MAX_STEP_CONCURRENCY", max_step_concurrency or 2)),
            enable_caching=bool(_get_setting("ENABLE_PLANNER_CACHING", enable_caching or True)),
        )

    async def generate_scene_outline(self, topic: str, description: str, session_id: str) -> str:
        return await self.planner.generate_scene_outline(topic, description, session_id)

    async def generate_implementation(
        self,
        topic: str,
        description: str,
        plan_xml: str,
        session_id: str,
    ) -> List[str]:
        return await self.planner.generate_scene_implementation_concurrently_enhanced(
            topic, description, plan_xml, session_id
        )


class CodeGenAdapter:
    """Adapter for scene code generation and fixes."""

    def __init__(
        self,
        *,
        scene_model: Any,
        helper_model: Optional[Any] = None,
        output_dir: Optional[str] = None,
        use_rag: Optional[bool] = None,
        use_context_learning: Optional[bool] = None,
        context_learning_path: Optional[str] = None,
        chroma_db_path: Optional[str] = None,
        manim_docs_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_visual_fix_code: Optional[bool] = None,
        use_langfuse: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if scene_model is None:
            raise ValueError("scene_model is required")

        self.codegen = CodeGenerator(
            scene_model=scene_model,
            helper_model=helper_model or scene_model,
            output_dir=output_dir or _get_setting("OUTPUT_DIR", "output"),
            print_response=_get_setting("DEBUG", True),
            use_rag=bool(_get_setting("USE_RAG", use_rag or False)),
            use_context_learning=bool(_get_setting("USE_CONTEXT_LEARNING", use_context_learning or False)),
            context_learning_path=context_learning_path or _get_setting("CONTEXT_LEARNING_PATH", "data/context_learning"),
            chroma_db_path=chroma_db_path or _get_setting("CHROMA_DB_PATH", "data/rag/chroma_db"),
            manim_docs_path=manim_docs_path or _get_setting("MANIM_DOCS_PATH", "data/rag/manim_docs"),
            embedding_model=embedding_model or _get_setting("EMBEDDING_MODEL", "azure/text-embedding-3-large"),
            use_visual_fix_code=bool(_get_setting("USE_VISUAL_FIX_CODE", use_visual_fix_code or False)),
            use_langfuse=bool(_get_setting("USE_LANGFUSE", use_langfuse or True)),
            session_id=session_id,
        )

    def generate_code(
        self,
        *,
        topic: str,
        description: str,
        scene_outline: str,
        implementation: str,
        scene_number: int,
        session_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        return self.codegen.generate_manim_code(
            topic=topic,
            description=description,
            scene_outline=scene_outline,
            scene_implementation=implementation,
            scene_number=scene_number,
            scene_trace_id=None,
            session_id=session_id,
        )

    def fix_code_errors(
        self,
        *,
        implementation_plan: str,
        code: str,
        error: str,
        scene_trace_id: str,
        topic: str,
        scene_number: int,
        session_id: str,
    ) -> Tuple[str, str]:
        return self.codegen.fix_code_errors(
            implementation_plan=implementation_plan,
            code=code,
            error=error,
            scene_trace_id=scene_trace_id,
            topic=topic,
            scene_number=scene_number,
            session_id=session_id,
        )


class VideoRenderAdapter:
    """Adapter for scene rendering and video combination."""

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        use_visual_fix_code: Optional[bool] = None,
        max_concurrent_renders: Optional[int] = None,
        enable_caching: Optional[bool] = None,
        default_quality: Optional[str] = None,
        use_gpu_acceleration: Optional[bool] = None,
        preview_mode: Optional[bool] = None,
    ) -> None:
        self.output_dir = output_dir or _get_setting("OUTPUT_DIR", "output")
        self.renderer = OptimizedVideoRenderer(
            output_dir=self.output_dir,
            print_response=_get_setting("DEBUG", True),
            use_visual_fix_code=bool(_get_setting("USE_VISUAL_FIX_CODE", use_visual_fix_code or False)),
            max_concurrent_renders=int(_get_setting("MAX_CONCURRENT_RENDERS", max_concurrent_renders or 2)),
            enable_caching=bool(_get_setting("RENDER_CACHE_ENABLED", enable_caching or True)),
            default_quality=str(_get_setting("DEFAULT_RENDER_QUALITY", default_quality or "medium")),
            use_gpu_acceleration=bool(_get_setting("USE_GPU", use_gpu_acceleration or False)),
            preview_mode=bool(_get_setting("PREVIEW_MODE", preview_mode or False)),
        )

    async def render_scene(
        self,
        *,
        code: str,
        topic: str,
        scene_number: int,
        version: int,
        session_id: Optional[str] = None,
        quality: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        file_prefix = _sanitize_topic_to_prefix(topic)
        code_dir = str(Path(self.output_dir) / file_prefix / f"scene{scene_number}" / "code")
        media_dir = str(Path(self.output_dir) / file_prefix / "media")

        return await self.renderer.render_scene_optimized(
            code=code,
            file_prefix=file_prefix,
            curr_scene=scene_number,
            curr_version=version,
            code_dir=code_dir,
            media_dir=media_dir,
            quality=quality or str(_get_setting("DEFAULT_RENDER_QUALITY", "medium")),
            session_id=session_id,
        )

    async def combine_videos(self, topic: str, *, use_hardware_acceleration: Optional[bool] = None) -> str:
        return await self.renderer.combine_videos_optimized(
            topic=topic,
            use_hardware_acceleration=bool(
                _get_setting("USE_GPU", use_hardware_acceleration if use_hardware_acceleration is not None else False)
            ),
        )

    def cleanup_cache(self, max_age_days: int = 7) -> None:
        self.renderer.cleanup_cache(max_age_days=max_age_days)


__all__ = [
    "VideoPlannerAdapter",
    "CodeGenAdapter",
    "VideoRenderAdapter",
]


