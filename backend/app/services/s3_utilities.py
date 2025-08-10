"""
S3 utility functions for video generation and scene editing.

This module provides helper functions for:
- Uploading video files and source code to S3
- Generating public URLs for S3 objects
- Managing S3 object versioning
- Batch operations for efficiency
"""
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from agents.src.storage.s3_storage import S3Storage
from app.core.settings import settings
from app.services.convex_s3_sync import ConvexS3Sync, AssetType

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class S3UploadResult:
    """Result of an S3 upload operation."""
    success: bool
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    public_url: Optional[str] = None
    file_size: Optional[int] = None
    content_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class S3BatchUploadResult:
    """Result of a batch S3 upload operation."""
    successful_uploads: List[S3UploadResult]
    failed_uploads: List[S3UploadResult]
    total_files: int
    total_size_bytes: int


class S3VideoUtilities:
    """
    Utility class for S3 operations specific to video generation workflow.
    
    This class provides high-level methods for uploading video assets,
    managing versions, and coordinating with Convex database updates.
    """
    
    def __init__(self, s3_storage: Optional[S3Storage] = None, convex_sync: Optional[ConvexS3Sync] = None):
        """
        Initialize S3 utilities.
        
        Args:
            s3_storage: Optional S3Storage instance. Creates default if not provided.
            convex_sync: Optional ConvexS3Sync instance. Creates default if not provided.
        """
        self.s3_storage = s3_storage or self._create_default_s3_storage()
        self.convex_sync = convex_sync or ConvexS3Sync()
    
    def _create_default_s3_storage(self) -> S3Storage:
        """Create default S3Storage instance from settings."""
        if not settings.S3_BUCKET:
            raise ValueError("S3_BUCKET not configured in settings")
        
        return S3Storage(
            bucket=settings.S3_BUCKET,
            base_prefix=settings.S3_BASE_PREFIX
        )
    
    def upload_video_chunk(
        self,
        video_file_path: str,
        scene_id: str,
        session_id: str,
        scene_number: int,
        version: int = 1
    ) -> S3UploadResult:
        """
        Upload a video chunk to S3 with proper naming and metadata.
        
        Args:
            video_file_path: Local path to the video file
            scene_id: Unique scene identifier
            session_id: Video session identifier
            scene_number: Scene number in sequence
            version: Scene version number
            
        Returns:
            S3UploadResult with upload details
        """
        try:
            logger.info(f"Uploading video chunk for scene {scene_id}")
            
            # Validate file exists
            video_path = Path(video_file_path)
            if not video_path.exists():
                return S3UploadResult(
                    success=False,
                    error_message=f"Video file not found: {video_file_path}"
                )
            
            # Generate S3 key
            s3_key = f"{session_id}/scenes/scene_{scene_number}_v{version}.mp4"
            
            # Get file metadata
            file_size = video_path.stat().st_size
            content_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
            
            # Upload with Convex sync
            s3_key, s3_url = self.s3_storage.upload_with_convex_sync(
                local_path=str(video_path),
                key=s3_key,
                convex_sync=self.convex_sync,
                entity_id=scene_id,
                entity_type="scene",
                asset_type=AssetType.VIDEO_CHUNK,
                extra_args={
                    "Metadata": {
                        "scene_id": scene_id,
                        "session_id": session_id,
                        "scene_number": str(scene_number),
                        "version": str(version),
                        "asset_type": "video_chunk"
                    }
                }
            )
            
            # Generate public URL
            public_url = self.s3_storage.url_for(s3_key)
            
            logger.info(f"Successfully uploaded video chunk: {s3_url}")
            
            return S3UploadResult(
                success=True,
                s3_key=s3_key,
                s3_url=s3_url,
                public_url=public_url,
                file_size=file_size,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Failed to upload video chunk: {e}")
            return S3UploadResult(
                success=False,
                error_message=str(e)
            )
    
    def upload_source_code(
        self,
        source_file_path: str,
        scene_id: str,
        session_id: str,
        scene_number: int,
        version: int = 1
    ) -> S3UploadResult:
        """
        Upload Manim source code to S3 with proper naming and metadata.
        
        Args:
            source_file_path: Local path to the source code file
            scene_id: Unique scene identifier
            session_id: Video session identifier
            scene_number: Scene number in sequence
            version: Scene version number
            
        Returns:
            S3UploadResult with upload details
        """
        try:
            logger.info(f"Uploading source code for scene {scene_id}")
            
            # Validate file exists
            source_path = Path(source_file_path)
            if not source_path.exists():
                return S3UploadResult(
                    success=False,
                    error_message=f"Source file not found: {source_file_path}"
                )
            
            # Generate S3 key
            s3_key = f"{session_id}/sources/scene_{scene_number}_v{version}.py"
            
            # Get file metadata
            file_size = source_path.stat().st_size
            content_type = "text/x-python"
            
            # Upload with Convex sync
            s3_key, s3_url = self.s3_storage.upload_with_convex_sync(
                local_path=str(source_path),
                key=s3_key,
                convex_sync=self.convex_sync,
                entity_id=scene_id,
                entity_type="scene",
                asset_type=AssetType.SOURCE_CODE,
                extra_args={
                    "Metadata": {
                        "scene_id": scene_id,
                        "session_id": session_id,
                        "scene_number": str(scene_number),
                        "version": str(version),
                        "asset_type": "source_code"
                    }
                }
            )
            
            # Generate public URL
            public_url = self.s3_storage.url_for(s3_key)
            
            logger.info(f"Successfully uploaded source code: {s3_url}")
            
            return S3UploadResult(
                success=True,
                s3_key=s3_key,
                s3_url=s3_url,
                public_url=public_url,
                file_size=file_size,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Failed to upload source code: {e}")
            return S3UploadResult(
                success=False,
                error_message=str(e)
            )
    
    def upload_combined_video(
        self,
        video_file_path: str,
        session_id: str,
        project_id: str
    ) -> S3UploadResult:
        """
        Upload combined video to S3 for a complete video session.
        
        Args:
            video_file_path: Local path to the combined video file
            session_id: Video session identifier
            project_id: Project identifier
            
        Returns:
            S3UploadResult with upload details
        """
        try:
            logger.info(f"Uploading combined video for session {session_id}")
            
            # Validate file exists
            video_path = Path(video_file_path)
            if not video_path.exists():
                return S3UploadResult(
                    success=False,
                    error_message=f"Combined video file not found: {video_file_path}"
                )
            
            # Generate S3 key
            s3_key = f"{session_id}/combined_video.mp4"
            
            # Get file metadata
            file_size = video_path.stat().st_size
            content_type = "video/mp4"
            
            # Upload with Convex sync
            s3_key, s3_url = self.s3_storage.upload_with_convex_sync(
                local_path=str(video_path),
                key=s3_key,
                convex_sync=self.convex_sync,
                entity_id=session_id,
                entity_type="session",
                asset_type=AssetType.COMBINED_VIDEO,
                extra_args={
                    "Metadata": {
                        "session_id": session_id,
                        "project_id": project_id,
                        "asset_type": "combined_video"
                    }
                }
            )
            
            # Generate public URL
            public_url = self.s3_storage.url_for(s3_key)
            
            logger.info(f"Successfully uploaded combined video: {s3_url}")
            
            return S3UploadResult(
                success=True,
                s3_key=s3_key,
                s3_url=s3_url,
                public_url=public_url,
                file_size=file_size,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Failed to upload combined video: {e}")
            return S3UploadResult(
                success=False,
                error_message=str(e)
            )
    
    def upload_manifest(
        self,
        manifest_data: Dict[str, Any],
        session_id: str,
        project_id: str
    ) -> S3UploadResult:
        """
        Upload video session manifest to S3.
        
        Args:
            manifest_data: Dictionary containing session manifest data
            session_id: Video session identifier
            project_id: Project identifier
            
        Returns:
            S3UploadResult with upload details
        """
        try:
            logger.info(f"Uploading manifest for session {session_id}")
            
            # Generate S3 key
            s3_key = f"{session_id}/manifest.json"
            
            # Upload manifest using S3Storage method
            full_s3_key = self.s3_storage.write_manifest(session_id, manifest_data)
            s3_url = self.s3_storage.url_for(full_s3_key)
            public_url = self.s3_storage.url_for(full_s3_key)
            
            # Sync with Convex
            import asyncio
            asyncio.run(
                self.convex_sync.sync_session_assets(
                    session_id,
                    {"manifest": s3_url}
                )
            )
            
            logger.info(f"Successfully uploaded manifest: {s3_url}")
            
            return S3UploadResult(
                success=True,
                s3_key=full_s3_key,
                s3_url=s3_url,
                public_url=public_url,
                content_type="application/json"
            )
            
        except Exception as e:
            logger.error(f"Failed to upload manifest: {e}")
            return S3UploadResult(
                success=False,
                error_message=str(e)
            )
    
    def batch_upload_scene_assets(
        self,
        scene_assets: List[Dict[str, Any]]
    ) -> S3BatchUploadResult:
        """
        Upload multiple scene assets in batch for efficiency.
        
        Args:
            scene_assets: List of dictionaries containing asset information:
                - asset_type: "video" or "source"
                - file_path: Local path to file
                - scene_id: Scene identifier
                - session_id: Session identifier
                - scene_number: Scene number
                - version: Version number
                
        Returns:
            S3BatchUploadResult with batch operation results
        """
        logger.info(f"Starting batch upload of {len(scene_assets)} scene assets")
        
        successful_uploads = []
        failed_uploads = []
        total_size_bytes = 0
        
        for asset_info in scene_assets:
            try:
                asset_type = asset_info.get("asset_type")
                
                if asset_type == "video":
                    result = self.upload_video_chunk(
                        video_file_path=asset_info["file_path"],
                        scene_id=asset_info["scene_id"],
                        session_id=asset_info["session_id"],
                        scene_number=asset_info["scene_number"],
                        version=asset_info.get("version", 1)
                    )
                elif asset_type == "source":
                    result = self.upload_source_code(
                        source_file_path=asset_info["file_path"],
                        scene_id=asset_info["scene_id"],
                        session_id=asset_info["session_id"],
                        scene_number=asset_info["scene_number"],
                        version=asset_info.get("version", 1)
                    )
                else:
                    result = S3UploadResult(
                        success=False,
                        error_message=f"Unknown asset type: {asset_type}"
                    )
                
                if result.success:
                    successful_uploads.append(result)
                    if result.file_size:
                        total_size_bytes += result.file_size
                else:
                    failed_uploads.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to upload asset {asset_info}: {e}")
                failed_uploads.append(S3UploadResult(
                    success=False,
                    error_message=str(e)
                ))
        
        logger.info(f"Batch upload completed: {len(successful_uploads)} successful, {len(failed_uploads)} failed")
        
        return S3BatchUploadResult(
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            total_files=len(scene_assets),
            total_size_bytes=total_size_bytes
        )
    
    def generate_presigned_urls(
        self,
        s3_keys: List[str],
        expires_in: int = 3600
    ) -> Dict[str, str]:
        """
        Generate presigned URLs for multiple S3 objects.
        
        Args:
            s3_keys: List of S3 keys to generate URLs for
            expires_in: URL expiration time in seconds
            
        Returns:
            Dictionary mapping S3 keys to presigned URLs
        """
        logger.info(f"Generating presigned URLs for {len(s3_keys)} objects")
        
        presigned_urls = {}
        
        for s3_key in s3_keys:
            try:
                url = self.s3_storage.generate_presigned_url(s3_key, expires_in)
                presigned_urls[s3_key] = url
            except Exception as e:
                logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
                presigned_urls[s3_key] = None
        
        return presigned_urls
    
    def cleanup_scene_assets(
        self,
        scene_id: str,
        versions_to_keep: int = 3
    ) -> Dict[str, Any]:
        """
        Clean up old versions of scene assets, keeping only recent versions.
        
        Args:
            scene_id: Scene identifier
            versions_to_keep: Number of recent versions to retain
            
        Returns:
            Dictionary with cleanup results
        """
        logger.info(f"Cleaning up old assets for scene {scene_id}")
        
        try:
            # Get asset manifest for the scene
            scene_prefix = f"scenes/{scene_id}/"
            manifest = self.s3_storage.get_asset_manifest(scene_prefix)
            
            # Group assets by version
            version_groups = {}
            for asset in manifest["assets"]:
                # Extract version from key (assuming format includes version)
                key_parts = asset["key"].split("/")
                version = None
                for part in key_parts:
                    if part.startswith("v") and part[1:].isdigit():
                        version = int(part[1:])
                        break
                
                if version is not None:
                    if version not in version_groups:
                        version_groups[version] = []
                    version_groups[version].append(asset["key"])
            
            # Determine versions to delete
            sorted_versions = sorted(version_groups.keys(), reverse=True)
            versions_to_delete = sorted_versions[versions_to_keep:]
            
            # Delete old versions
            keys_to_delete = []
            for version in versions_to_delete:
                keys_to_delete.extend(version_groups[version])
            
            if keys_to_delete:
                cleanup_success = self.s3_storage.cleanup_with_convex_sync(
                    keys_to_delete,
                    self.convex_sync,
                    scene_id,
                    "scene"
                )
                
                return {
                    "success": cleanup_success,
                    "deleted_versions": versions_to_delete,
                    "deleted_files": len(keys_to_delete),
                    "retained_versions": sorted_versions[:versions_to_keep]
                }
            else:
                return {
                    "success": True,
                    "deleted_versions": [],
                    "deleted_files": 0,
                    "message": "No old versions to clean up"
                }
                
        except Exception as e:
            logger.error(f"Failed to cleanup scene assets: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Standalone utility functions

def get_s3_utilities() -> S3VideoUtilities:
    """Get a configured S3VideoUtilities instance."""
    return S3VideoUtilities()


def upload_video_file(
    file_path: str,
    s3_key: str,
    metadata: Optional[Dict[str, str]] = None
) -> S3UploadResult:
    """
    Simple utility function to upload a video file to S3.
    
    Args:
        file_path: Local path to video file
        s3_key: S3 key for the uploaded file
        metadata: Optional metadata to attach
        
    Returns:
        S3UploadResult with upload details
    """
    try:
        s3_storage = S3Storage(
            bucket=settings.S3_BUCKET,
            base_prefix=settings.S3_BASE_PREFIX
        )
        
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata
        
        full_s3_key = s3_storage.upload_file(file_path, s3_key, extra_args)
        s3_url = s3_storage.url_for(full_s3_key)
        public_url = s3_storage.url_for(full_s3_key)
        
        file_size = Path(file_path).stat().st_size
        content_type = mimetypes.guess_type(file_path)[0] or "video/mp4"
        
        return S3UploadResult(
            success=True,
            s3_key=full_s3_key,
            s3_url=s3_url,
            public_url=public_url,
            file_size=file_size,
            content_type=content_type
        )
        
    except Exception as e:
        return S3UploadResult(
            success=False,
            error_message=str(e)
        )


def generate_public_url(s3_key: str) -> str:
    """
    Generate a public URL for an S3 object.
    
    Args:
        s3_key: S3 key for the object
        
    Returns:
        Public URL string
    """
    s3_storage = S3Storage(
        bucket=settings.S3_BUCKET,
        base_prefix=settings.S3_BASE_PREFIX
    )
    
    return s3_storage.url_for(s3_key)