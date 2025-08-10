"""
Asset Manager service for high-level asset operations.

This service provides a unified interface for managing asset lifecycle across
S3 and Convex, coordinating uploads, retrievals, and cleanup operations while
maintaining data consistency between storage systems.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from agents.src.storage.s3_storage import S3Storage
from .convex_s3_sync import ConvexS3Sync, AssetSyncRequest, SyncOperation
from backend.convex.types.schema import AssetType, AssetReference, UrlType, S3Asset
from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class AssetManager:
    """
    High-level service for managing asset lifecycle across S3 and Convex.
    
    This service coordinates asset uploads with proper metadata, handles asset
    versioning and cleanup, provides unified asset access interface, and manages
    asset permissions and access control.
    
    Key responsibilities:
    - Coordinate asset uploads with proper metadata
    - Handle asset versioning and cleanup
    - Provide unified asset access interface
    - Manage asset permissions and access control
    """
    
    def __init__(self, s3_storage: Optional[S3Storage] = None, convex_sync: Optional[ConvexS3Sync] = None):
        """
        Initialize the AssetManager with S3Storage and ConvexS3Sync dependencies.
        
        Args:
            s3_storage: S3Storage service instance. If not provided, creates one using settings.
            convex_sync: ConvexS3Sync service instance. If not provided, creates one using settings.
        """
        self.settings = get_settings()
        
        # Initialize S3Storage if not provided
        if s3_storage is None:
            bucket = self.settings.s3_bucket_name
            base_prefix = getattr(self.settings, 's3_base_prefix', '')
            self.s3_storage = S3Storage(bucket=bucket, base_prefix=base_prefix)
        else:
            self.s3_storage = s3_storage
            
        # Initialize ConvexS3Sync if not provided
        if convex_sync is None:
            self.convex_sync = ConvexS3Sync()
        else:
            self.convex_sync = convex_sync
            
        logger.info("AssetManager initialized with S3Storage and ConvexS3Sync")
    
    async def store_scene_asset(
        self,
        scene_id: str,
        asset_type: AssetType,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssetReference:
        """
        Store a scene asset with coordinated S3 upload and Convex synchronization.
        
        This method handles the complete asset storage workflow:
        1. Validates the input file and parameters
        2. Generates appropriate S3 key based on scene and asset type
        3. Uploads file to S3 with proper metadata
        4. Synchronizes asset information with Convex
        5. Returns comprehensive asset reference
        
        Args:
            scene_id: ID of the scene this asset belongs to
            asset_type: Type of asset being stored (video, source code, etc.)
            file_path: Path to the local file to upload
            metadata: Optional additional metadata to store with the asset
            
        Returns:
            AssetReference object containing complete asset information
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If invalid parameters are provided
            Exception: If upload or sync operations fail
        """
        logger.info(f"Storing {asset_type.value} asset for scene {scene_id}")
        
        # Validate inputs
        if not scene_id:
            raise ValueError("scene_id is required")
        if not file_path:
            raise ValueError("file_path is required")
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        try:
            # Generate S3 key for the asset
            s3_key = self._generate_scene_asset_key(scene_id, asset_type, file_path_obj.name)
            
            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata.update({
                'scene_id': scene_id,
                'asset_type': asset_type.value,
                'original_filename': file_path_obj.name,
                'uploaded_by': 'asset_manager',
                'upload_timestamp': datetime.now().isoformat()
            })
            
            # Upload file to S3 with Convex synchronization
            s3_key_result, s3_url = self.s3_storage.upload_with_convex_sync(
                local_path=str(file_path),
                key=s3_key,
                convex_sync=self.convex_sync,
                entity_id=scene_id,
                entity_type="scene",
                asset_type=asset_type,
                extra_args={'Metadata': file_metadata}
            )
            
            # Calculate additional asset information
            file_size = file_path_obj.stat().st_size
            checksum = self.s3_storage._calculate_file_checksum(str(file_path))
            content_type = self.s3_storage._guess_content_type(str(file_path)) or "application/octet-stream"
            current_time = datetime.now().timestamp()
            
            # Create AssetReference object
            asset_reference = AssetReference(
                id=f"{scene_id}_{asset_type.value}_{int(current_time)}",
                entityId=scene_id,
                entityType="scene",
                assetType=asset_type,
                s3Key=s3_key_result,
                s3Url=s3_url,
                contentType=content_type,
                size=file_size,
                checksum=checksum,
                version=1,  # Initial version
                createdAt=current_time,
                metadata=file_metadata
            )
            
            logger.info(f"Successfully stored {asset_type.value} asset for scene {scene_id}: {s3_key_result}")
            return asset_reference
            
        except Exception as e:
            logger.error(f"Failed to store scene asset {asset_type.value} for scene {scene_id}: {str(e)}")
            raise
    
    async def get_scene_assets(self, scene_id: str) -> List[AssetReference]:
        """
        Retrieve comprehensive asset information for a scene.
        
        This method queries Convex to get all assets associated with a scene
        and returns them as AssetReference objects with complete metadata.
        
        Args:
            scene_id: ID of the scene to retrieve assets for
            
        Returns:
            List of AssetReference objects for all scene assets
            
        Raises:
            ValueError: If scene_id is not provided
            Exception: If retrieval operation fails
        """
        if not scene_id:
            raise ValueError("scene_id is required")
            
        logger.info(f"Retrieving assets for scene {scene_id}")
        
        try:
            # Query Convex for scene document
            scene_doc = await self._get_scene_document(scene_id)
            if not scene_doc:
                logger.warning(f"Scene {scene_id} not found")
                return []
            
            assets = []
            current_time = datetime.now().timestamp()
            
            # Extract assets from scene document
            asset_fields = {
                'videoAsset': AssetType.VIDEO_CHUNK,
                'sourceCodeAsset': AssetType.SOURCE_CODE,
                'thumbnailAsset': AssetType.THUMBNAIL,
                'subtitleAsset': AssetType.SUBTITLE
            }
            
            for field_name, asset_type in asset_fields.items():
                s3_asset_data = scene_doc.get(field_name)
                if s3_asset_data:
                    # Convert S3Asset data to AssetReference
                    asset_reference = AssetReference(
                        id=f"{scene_id}_{asset_type.value}",
                        entityId=scene_id,
                        entityType="scene",
                        assetType=asset_type,
                        s3Key=s3_asset_data['s3Key'],
                        s3Url=s3_asset_data['s3Url'],
                        contentType=s3_asset_data.get('contentType', 'application/octet-stream'),
                        size=s3_asset_data.get('size', 0),
                        checksum=s3_asset_data.get('checksum', ''),
                        version=scene_doc.get('assetVersion', 1),
                        createdAt=s3_asset_data.get('uploadedAt', current_time),
                        metadata={
                            'scene_id': scene_id,
                            'asset_status': scene_doc.get('assetsStatus', 'unknown'),
                            'asset_version': scene_doc.get('assetVersion', 1)
                        }
                    )
                    assets.append(asset_reference)
            
            logger.info(f"Retrieved {len(assets)} assets for scene {scene_id}")
            return assets
            
        except Exception as e:
            logger.error(f"Failed to retrieve assets for scene {scene_id}: {str(e)}")
            raise
    
    async def cleanup_scene_assets(self, scene_id: str) -> bool:
        """
        Clean up all assets associated with a scene for lifecycle management.
        
        This method performs coordinated cleanup of all scene assets:
        1. Retrieves all current scene assets
        2. Deletes files from S3
        3. Updates Convex records to reflect cleanup
        4. Handles partial failures gracefully
        
        Args:
            scene_id: ID of the scene to clean up assets for
            
        Returns:
            True if all cleanup operations succeeded, False if any failures occurred
            
        Raises:
            ValueError: If scene_id is not provided
            Exception: If critical cleanup operations fail
        """
        if not scene_id:
            raise ValueError("scene_id is required")
            
        logger.info(f"Starting cleanup of assets for scene {scene_id}")
        
        try:
            # Get all current scene assets
            assets = await self.get_scene_assets(scene_id)
            
            if not assets:
                logger.info(f"No assets found for scene {scene_id}, cleanup complete")
                return True
            
            # Extract S3 keys for deletion
            s3_keys = []
            for asset in assets:
                # Convert full S3 key to relative key for cleanup method
                relative_key = asset.s3Key
                if self.s3_storage.base_prefix and asset.s3Key.startswith(self.s3_storage.base_prefix + "/"):
                    relative_key = asset.s3Key[len(self.s3_storage.base_prefix) + 1:]
                s3_keys.append(relative_key)
            
            logger.info(f"Cleaning up {len(s3_keys)} assets for scene {scene_id}")
            
            # Perform coordinated cleanup
            cleanup_success = self.s3_storage.cleanup_with_convex_sync(
                keys=s3_keys,
                convex_sync=self.convex_sync,
                entity_id=scene_id,
                entity_type="scene"
            )
            
            if cleanup_success:
                logger.info(f"Successfully cleaned up all assets for scene {scene_id}")
            else:
                logger.warning(f"Some assets failed to clean up for scene {scene_id}")
            
            return cleanup_success
            
        except Exception as e:
            logger.error(f"Failed to cleanup assets for scene {scene_id}: {str(e)}")
            raise
    
    async def get_asset_url(
        self,
        asset_id: str,
        url_type: UrlType = UrlType.PUBLIC,
        expiration: Optional[int] = None
    ) -> str:
        """
        Generate URLs for asset access with different types (public, presigned).
        
        This method provides flexible URL generation for assets based on the
        requested URL type and security requirements.
        
        Args:
            asset_id: ID of the asset to generate URL for
            url_type: Type of URL to generate (public or presigned)
            expiration: Optional expiration time in seconds for presigned URLs
            
        Returns:
            URL string for accessing the asset
            
        Raises:
            ValueError: If asset_id is not provided or asset not found
            Exception: If URL generation fails
        """
        if not asset_id:
            raise ValueError("asset_id is required")
            
        logger.info(f"Generating {url_type.value} URL for asset {asset_id}")
        
        try:
            # Parse asset_id to extract entity information
            # Asset ID format: {entity_id}_{asset_type}[_{timestamp}]
            parts = asset_id.split('_')
            if len(parts) < 2:
                raise ValueError(f"Invalid asset_id format: {asset_id}")
            
            entity_id = parts[0]
            asset_type_str = parts[1]
            
            # Find the asset by querying scene assets
            # (This is a simplified approach - in production you might want a more efficient lookup)
            assets = await self.get_scene_assets(entity_id)
            
            target_asset = None
            for asset in assets:
                if asset.assetType.value == asset_type_str:
                    target_asset = asset
                    break
            
            if not target_asset:
                raise ValueError(f"Asset not found: {asset_id}")
            
            # Generate URL based on type
            if url_type == UrlType.PUBLIC:
                # Use the stored S3 URL or generate public URL
                if target_asset.s3Url:
                    return target_asset.s3Url
                else:
                    # Generate public URL from S3 key
                    relative_key = target_asset.s3Key
                    if self.s3_storage.base_prefix and target_asset.s3Key.startswith(self.s3_storage.base_prefix + "/"):
                        relative_key = target_asset.s3Key[len(self.s3_storage.base_prefix) + 1:]
                    return self.s3_storage.url_for(relative_key, prefer_presigned=False)
                    
            elif url_type == UrlType.PRESIGNED:
                # Generate presigned URL
                relative_key = target_asset.s3Key
                if self.s3_storage.base_prefix and target_asset.s3Key.startswith(self.s3_storage.base_prefix + "/"):
                    relative_key = target_asset.s3Key[len(self.s3_storage.base_prefix) + 1:]
                
                if expiration:
                    return self.s3_storage.generate_presigned_url(relative_key, expires=expiration)
                else:
                    return self.s3_storage.url_for(relative_key, prefer_presigned=True)
            
            else:
                raise ValueError(f"Unsupported URL type: {url_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate {url_type.value} URL for asset {asset_id}: {str(e)}")
            raise
    
    # Additional utility methods for session assets
    
    async def store_session_asset(
        self,
        session_id: str,
        asset_type: AssetType,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssetReference:
        """
        Store a session asset (combined video, manifest, etc.).
        
        Similar to store_scene_asset but for video session assets.
        """
        logger.info(f"Storing {asset_type.value} asset for session {session_id}")
        
        # Validate inputs
        if not session_id:
            raise ValueError("session_id is required")
        if not file_path:
            raise ValueError("file_path is required")
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Generate S3 key for the session asset
            s3_key = self._generate_session_asset_key(session_id, asset_type, file_path_obj.name)
            
            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata.update({
                'session_id': session_id,
                'asset_type': asset_type.value,
                'original_filename': file_path_obj.name,
                'uploaded_by': 'asset_manager',
                'upload_timestamp': datetime.now().isoformat()
            })
            
            # Upload file to S3 with Convex synchronization
            s3_key_result, s3_url = self.s3_storage.upload_with_convex_sync(
                local_path=str(file_path),
                key=s3_key,
                convex_sync=self.convex_sync,
                entity_id=session_id,
                entity_type="session",
                asset_type=asset_type,
                extra_args={'Metadata': file_metadata}
            )
            
            # Calculate additional asset information
            file_size = file_path_obj.stat().st_size
            checksum = self.s3_storage._calculate_file_checksum(str(file_path))
            content_type = self.s3_storage._guess_content_type(str(file_path)) or "application/octet-stream"
            current_time = datetime.now().timestamp()
            
            # Create AssetReference object
            asset_reference = AssetReference(
                id=f"{session_id}_{asset_type.value}_{int(current_time)}",
                entityId=session_id,
                entityType="session",
                assetType=asset_type,
                s3Key=s3_key_result,
                s3Url=s3_url,
                contentType=content_type,
                size=file_size,
                checksum=checksum,
                version=1,
                createdAt=current_time,
                metadata=file_metadata
            )
            
            logger.info(f"Successfully stored {asset_type.value} asset for session {session_id}: {s3_key_result}")
            return asset_reference
            
        except Exception as e:
            logger.error(f"Failed to store session asset {asset_type.value} for session {session_id}: {str(e)}")
            raise
    
    async def get_session_assets(self, session_id: str) -> List[AssetReference]:
        """Retrieve all assets for a video session."""
        if not session_id:
            raise ValueError("session_id is required")
            
        logger.info(f"Retrieving assets for session {session_id}")
        
        try:
            # Query Convex for session document
            session_doc = await self._get_session_document(session_id)
            if not session_doc:
                logger.warning(f"Session {session_id} not found")
                return []
            
            assets = []
            current_time = datetime.now().timestamp()
            
            # Extract assets from session document
            asset_fields = {
                'combinedVideoAsset': AssetType.COMBINED_VIDEO,
                'combinedSubtitleAsset': AssetType.SUBTITLE,
                'manifestAsset': AssetType.MANIFEST
            }
            
            for field_name, asset_type in asset_fields.items():
                s3_asset_data = session_doc.get(field_name)
                if s3_asset_data:
                    asset_reference = AssetReference(
                        id=f"{session_id}_{asset_type.value}",
                        entityId=session_id,
                        entityType="session",
                        assetType=asset_type,
                        s3Key=s3_asset_data['s3Key'],
                        s3Url=s3_asset_data['s3Url'],
                        contentType=s3_asset_data.get('contentType', 'application/octet-stream'),
                        size=s3_asset_data.get('size', 0),
                        checksum=s3_asset_data.get('checksum', ''),
                        version=1,
                        createdAt=s3_asset_data.get('uploadedAt', current_time),
                        metadata={
                            'session_id': session_id,
                            'asset_status': session_doc.get('assetsStatus', 'unknown'),
                            'total_asset_size': session_doc.get('totalAssetSize', 0),
                            'asset_count': session_doc.get('assetCount', 0)
                        }
                    )
                    assets.append(asset_reference)
            
            logger.info(f"Retrieved {len(assets)} assets for session {session_id}")
            return assets
            
        except Exception as e:
            logger.error(f"Failed to retrieve assets for session {session_id}: {str(e)}")
            raise
    
    # Private helper methods
    
    def _generate_scene_asset_key(self, scene_id: str, asset_type: AssetType, filename: str) -> str:
        """Generate S3 key for scene asset."""
        # Format: scenes/{scene_id}/{asset_type}/{filename}
        sanitized_filename = filename.replace(' ', '_').replace('/', '_')
        return f"scenes/{scene_id}/{asset_type.value}/{sanitized_filename}"
    
    def _generate_session_asset_key(self, session_id: str, asset_type: AssetType, filename: str) -> str:
        """Generate S3 key for session asset."""
        # Format: sessions/{session_id}/{asset_type}/{filename}
        sanitized_filename = filename.replace(' ', '_').replace('/', '_')
        return f"sessions/{session_id}/{asset_type.value}/{sanitized_filename}"
    
    async def _get_scene_document(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve scene document from Convex."""
        try:
            # Use the convex_sync client to query the scene
            result = self.convex_sync.convex_client.query(
                "getDocument",
                {
                    "collection": "scenes",
                    "id": scene_id
                }
            )
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve scene document {scene_id}: {str(e)}")
            return None
    
    async def _get_session_document(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session document from Convex."""
        try:
            # Use the convex_sync client to query the session
            result = self.convex_sync.convex_client.query(
                "getDocument",
                {
                    "collection": "videoSessions",
                    "id": session_id
                }
            )
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve session document {session_id}: {str(e)}")
            return None