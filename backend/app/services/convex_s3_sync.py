"""
ConvexS3Sync service for synchronizing S3 assets with Convex database.

This service orchestrates synchronization between S3 operations and Convex database updates,
ensuring that asset metadata is properly maintained across both systems.
"""
import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from convex import ConvexClient
from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Import types from schema
from backend.convex.types.schema import AssetType, S3Asset, AssetReference


class SyncOperation(str, Enum):
    """Types of sync operations."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class AssetSyncRequest:
    """Request object for asset synchronization."""
    entity_id: str
    entity_type: str  # "scene" or "session"
    s3_key: str
    s3_url: str
    asset_type: AssetType
    content_type: str = "application/octet-stream"
    size: int = 0
    checksum: str = ""
    metadata: Optional[Dict[str, Any]] = None
    operation: SyncOperation = SyncOperation.CREATE


@dataclass
class BatchSyncResponse:
    """Response object for batch synchronization operations."""
    successful: List[str]  # entity_ids
    failed: List[Tuple[str, str]]  # (entity_id, error_message)
    total_processed: int


class ConvexS3Sync:
    """
    Service for synchronizing S3 assets with Convex database.
    
    This service handles:
    - Scene asset synchronization
    - Video session asset synchronization  
    - Batch operations for efficiency
    - Error handling and retry logic
    """
    
    def __init__(self, convex_client: Optional[ConvexClient] = None):
        """
        Initialize the ConvexS3Sync service.
        
        Args:
            convex_client: Optional Convex client instance. If not provided,
                          will create one using settings.
        """
        self.settings = get_settings()
        self.convex_client = convex_client or ConvexClient(
            self.settings.convex_deployment_url
        )
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        
    async def sync_scene_assets(
        self,
        scene_id: str,
        s3_assets: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update scene records with S3 asset URLs and metadata.
        
        Args:
            scene_id: ID of the scene to update
            s3_assets: Dictionary mapping asset types to S3 URLs
                      e.g., {"video": "s3://bucket/video.mp4", "source": "s3://bucket/code.py"}
            metadata: Optional metadata to include with assets
            
        Returns:
            Updated scene document
            
        Raises:
            Exception: If sync operation fails after retries
        """
        logger.info(f"Syncing scene assets for scene {scene_id}")
        
        try:
            # Convert s3_assets to S3Asset objects
            asset_updates = {}
            current_time = datetime.now().timestamp()
            
            for asset_key, s3_url in s3_assets.items():
                # Map asset keys to proper asset types
                asset_type = self._map_asset_key_to_type(asset_key)
                if not asset_type:
                    logger.warning(f"Unknown asset key: {asset_key}")
                    continue
                    
                # Extract S3 key from URL
                s3_key = self._extract_s3_key_from_url(s3_url)
                
                # Create S3Asset object
                s3_asset = S3Asset(
                    s3Key=s3_key,
                    s3Url=s3_url,
                    contentType=self._guess_content_type(s3_key),
                    size=metadata.get(f"{asset_key}_size", 0) if metadata else 0,
                    checksum=metadata.get(f"{asset_key}_checksum", "") if metadata else "",
                    uploadedAt=current_time
                )
                
                # Map to scene asset field
                field_name = self._map_asset_type_to_scene_field(asset_type)
                if field_name:
                    asset_updates[field_name] = s3_asset.dict()
            
            # Update assets status
            asset_updates["assetsStatus"] = "ready"
            asset_updates["assetsErrorMessage"] = None
            asset_updates["assetVersion"] = asset_updates.get("assetVersion", 1) + 1
            
            # Perform the update with retry logic
            return await self._retry_operation(
                self._update_scene_document,
                scene_id,
                asset_updates
            )
            
        except Exception as e:
            logger.error(f"Failed to sync scene assets for {scene_id}: {str(e)}")
            # Update scene with error status
            await self._update_scene_error_status(scene_id, str(e))
            raise
    
    async def sync_session_assets(
        self,
        session_id: str,
        combined_assets: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Update video session records with combined asset URLs.
        
        Args:
            session_id: ID of the video session to update
            combined_assets: Dictionary mapping asset types to S3 URLs
                           e.g., {"combined_video": "s3://bucket/final.mp4", "manifest": "s3://bucket/manifest.json"}
            
        Returns:
            Updated video session document
            
        Raises:
            Exception: If sync operation fails after retries
        """
        logger.info(f"Syncing session assets for session {session_id}")
        
        try:
            asset_updates = {}
            current_time = datetime.now().timestamp()
            total_size = 0
            asset_count = 0
            
            for asset_key, s3_url in combined_assets.items():
                # Extract S3 key from URL
                s3_key = self._extract_s3_key_from_url(s3_url)
                
                # Create S3Asset object
                s3_asset = S3Asset(
                    s3Key=s3_key,
                    s3Url=s3_url,
                    contentType=self._guess_content_type(s3_key),
                    size=0,  # Size will be updated separately if available
                    checksum="",  # Checksum will be updated separately if available
                    uploadedAt=current_time
                )
                
                # Map to session asset field
                if asset_key == "combined_video":
                    asset_updates["combinedVideoAsset"] = s3_asset.dict()
                    asset_count += 1
                elif asset_key == "combined_subtitle":
                    asset_updates["combinedSubtitleAsset"] = s3_asset.dict()
                    asset_count += 1
                elif asset_key == "manifest":
                    asset_updates["manifestAsset"] = s3_asset.dict()
                    asset_count += 1
            
            # Update summary fields
            asset_updates["totalAssetSize"] = total_size
            asset_updates["assetCount"] = asset_count
            asset_updates["assetsStatus"] = "ready"
            asset_updates["assetsErrorMessage"] = None
            
            # Perform the update with retry logic
            return await self._retry_operation(
                self._update_session_document,
                session_id,
                asset_updates
            )
            
        except Exception as e:
            logger.error(f"Failed to sync session assets for {session_id}: {str(e)}")
            # Update session with error status
            await self._update_session_error_status(session_id, str(e))
            raise
    
    async def batch_sync_assets(
        self,
        asset_batch: List[AssetSyncRequest]
    ) -> BatchSyncResponse:
        """
        Handle bulk asset synchronization operations efficiently.
        
        Args:
            asset_batch: List of asset sync requests to process
            
        Returns:
            BatchSyncResponse with results of the batch operation
        """
        logger.info(f"Processing batch sync of {len(asset_batch)} assets")
        
        successful = []
        failed = []
        
        # Group requests by entity type for efficient processing
        scene_requests = [req for req in asset_batch if req.entity_type == "scene"]
        session_requests = [req for req in asset_batch if req.entity_type == "session"]
        
        # Process scene requests
        for request in scene_requests:
            try:
                s3_assets = {request.asset_type.value: request.s3_url}
                metadata = {
                    f"{request.asset_type.value}_size": request.size,
                    f"{request.asset_type.value}_checksum": request.checksum
                }
                if request.metadata:
                    metadata.update(request.metadata)
                
                await self.sync_scene_assets(
                    request.entity_id,
                    s3_assets,
                    metadata
                )
                successful.append(request.entity_id)
                
            except Exception as e:
                logger.error(f"Failed to sync scene asset {request.entity_id}: {str(e)}")
                failed.append((request.entity_id, str(e)))
        
        # Process session requests
        for request in session_requests:
            try:
                combined_assets = {request.asset_type.value: request.s3_url}
                
                await self.sync_session_assets(
                    request.entity_id,
                    combined_assets
                )
                successful.append(request.entity_id)
                
            except Exception as e:
                logger.error(f"Failed to sync session asset {request.entity_id}: {str(e)}")
                failed.append((request.entity_id, str(e)))
        
        return BatchSyncResponse(
            successful=successful,
            failed=failed,
            total_processed=len(asset_batch)
        )
    
    async def cleanup_orphaned_assets(self, project_id: str) -> int:
        """
        Remove references to deleted S3 assets for a project.
        
        Args:
            project_id: ID of the project to clean up
            
        Returns:
            Number of orphaned asset references cleaned up
        """
        logger.info(f"Cleaning up orphaned assets for project {project_id}")
        
        try:
            # This would typically involve:
            # 1. Query all scenes and sessions for the project
            # 2. Check if referenced S3 objects still exist
            # 3. Remove references to missing objects
            # 4. Update asset counts and status
            
            # For now, return 0 as this is a placeholder implementation
            # The actual implementation would require S3 client integration
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned assets for project {project_id}: {str(e)}")
            raise
    
    # Private helper methods
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry logic.
        
        Args:
            operation: The async operation to retry
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If operation fails after all retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Operation failed after {self.max_retries} retries: {str(e)}")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def _update_scene_document(self, scene_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a scene document in Convex."""
        current_time = datetime.now().timestamp()
        updates["updatedAt"] = current_time
        
        return self.convex_client.mutation(
            "updateDocument",
            {
                "collection": "scenes",
                "id": scene_id,
                "updates": updates
            }
        )
    
    async def _update_session_document(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a video session document in Convex."""
        current_time = datetime.now().timestamp()
        updates["updatedAt"] = current_time
        
        return self.convex_client.mutation(
            "updateDocument",
            {
                "collection": "videoSessions",
                "id": session_id,
                "updates": updates
            }
        )
    
    async def _update_scene_error_status(self, scene_id: str, error_message: str):
        """Update scene with error status."""
        try:
            await self._update_scene_document(
                scene_id,
                {
                    "assetsStatus": "error",
                    "assetsErrorMessage": error_message
                }
            )
        except Exception as e:
            logger.error(f"Failed to update scene error status: {str(e)}")
    
    async def _update_session_error_status(self, session_id: str, error_message: str):
        """Update session with error status."""
        try:
            await self._update_session_document(
                session_id,
                {
                    "assetsStatus": "error",
                    "assetsErrorMessage": error_message
                }
            )
        except Exception as e:
            logger.error(f"Failed to update session error status: {str(e)}")
    
    def _map_asset_key_to_type(self, asset_key: str) -> Optional[AssetType]:
        """Map asset key to AssetType enum."""
        mapping = {
            "video": AssetType.VIDEO_CHUNK,
            "video_chunk": AssetType.VIDEO_CHUNK,
            "source": AssetType.SOURCE_CODE,
            "source_code": AssetType.SOURCE_CODE,
            "thumbnail": AssetType.THUMBNAIL,
            "subtitle": AssetType.SUBTITLE,
            "combined_video": AssetType.COMBINED_VIDEO,
            "manifest": AssetType.MANIFEST
        }
        return mapping.get(asset_key.lower())
    
    def _map_asset_type_to_scene_field(self, asset_type: AssetType) -> Optional[str]:
        """Map AssetType to scene field name."""
        mapping = {
            AssetType.VIDEO_CHUNK: "videoAsset",
            AssetType.SOURCE_CODE: "sourceCodeAsset",
            AssetType.THUMBNAIL: "thumbnailAsset",
            AssetType.SUBTITLE: "subtitleAsset"
        }
        return mapping.get(asset_type)
    
    def _extract_s3_key_from_url(self, s3_url: str) -> str:
        """Extract S3 key from S3 URL."""
        # Handle different S3 URL formats
        if s3_url.startswith("s3://"):
            # s3://bucket/key format
            parts = s3_url[5:].split("/", 1)
            return parts[1] if len(parts) > 1 else ""
        elif "amazonaws.com" in s3_url:
            # https://bucket.s3.region.amazonaws.com/key format
            parts = s3_url.split("/")
            return "/".join(parts[3:]) if len(parts) > 3 else ""
        else:
            # Assume it's already a key or custom format
            return s3_url.split("/")[-1] if "/" in s3_url else s3_url
    
    def _guess_content_type(self, s3_key: str) -> str:
        """Guess content type from S3 key extension."""
        import mimetypes
        content_type, _ = mimetypes.guess_type(s3_key)
        return content_type or "application/octet-stream"