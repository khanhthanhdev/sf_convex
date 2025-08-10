"""
S3 storage abstraction for uploading and serving agent outputs.

Implements the interface defined in docs/s3_storage_plan.md (Step 3):
- S3Storage(bucket, base_prefix)
- upload_file
- upload_bytes
- upload_dir
- generate_presigned_url
- url_for
- write_manifest (optional)

Enhanced with Convex integration methods:
- upload_with_convex_sync
- get_asset_manifest
- cleanup_with_convex_sync

This class intentionally reads minimal deployment settings from environment
variables to remain lightweight and reusable across agents without importing
backend settings directly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .s3_client import get_s3_client, get_s3_transfer_config

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from backend.app.services.convex_s3_sync import ConvexS3Sync, AssetType

# Configure logging
logger = logging.getLogger(__name__)


def _norm_join(*parts: str) -> str:
    # Join with POSIX-style separators for S3 keys
    joined = "/".join([p.strip("/") for p in parts if p is not None and str(p) != ""])
    return joined


def _guess_content_type(path_or_key: str) -> Optional[str]:
    ctype, _ = mimetypes.guess_type(path_or_key)
    return ctype


# ------------------------ Step 4: Key Strategy ------------------------
def sanitize_topic_to_prefix(topic: str) -> str:
    """Sanitize topic name to kebab-case prefix used as S3 root.

    - Lowercase
    - Replace any non-alphanumeric with '-'
    - Collapse multiple '-' and trim leading/trailing '-'
    """
    s = topic.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def key_for_local_path(local_path: str, *, topic_name: str, output_dir: str) -> str:
    """Derive S3 key from a local absolute/relative path by mirroring structure.

    Rule (per plan): strip local `output_dir` from absolute path; prepend
    `topic-name/` (sanitized to kebab-case) to the remaining relative path.
    """
    lp = Path(local_path).resolve()
    od = Path(output_dir).resolve()
    try:
        rel = lp.relative_to(od)
    except ValueError as e:
        raise ValueError(f"Local path {lp} is not under output_dir {od}") from e

    topic_prefix = sanitize_topic_to_prefix(topic_name)
    return _norm_join(topic_prefix, rel.as_posix())


class S3Storage:
    """
    High-level S3 storage helper.

    Args:
        bucket: Target S3 bucket name.
        base_prefix: Optional base prefix applied to all keys (e.g., "projects/").
    """

    def __init__(self, bucket: str, base_prefix: str = "") -> None:
        if not bucket:
            raise ValueError("bucket is required")
        self.bucket = bucket
        self.base_prefix = base_prefix.strip("/")

        # Configure S3 client and transfer config
        self.client = get_s3_client()
        self.transfer_config = get_s3_transfer_config()

        # Optional behavior/env
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.public_base_url = os.getenv("S3_PUBLIC_BASE_URL")
        self.kms_key_id = os.getenv("S3_KMS_KEY_ID")
        self.presign_expiration = int(os.getenv("S3_PRESIGN_EXPIRATION", "3600"))
        self.max_concurrency = int(os.getenv("S3_MAX_CONCURRENCY", "8"))

    # ---------------------------- key helpers ----------------------------
    def _full_key(self, key: str) -> str:
        if self.base_prefix:
            return _norm_join(self.base_prefix, key)
        return key.strip("/")

    # ----------------------------- uploads ------------------------------
    def upload_file(self, local_path: str, key: str, extra_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a local file to S3 using high-performance TransferConfig.

        Returns the S3 key actually used (including base_prefix).
        """
        if not Path(local_path).is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = self._full_key(key)

        # Build ExtraArgs with content-type and optional SSE-KMS
        extra: Dict[str, Any] = dict(extra_args or {})
        ctype = _guess_content_type(local_path) or _guess_content_type(full_key)
        if ctype:
            extra.setdefault("ContentType", ctype)
        if self.kms_key_id:
            extra.setdefault("ServerSideEncryption", "aws:kms")
            extra.setdefault("SSEKMSKeyId", self.kms_key_id)

        self.client.upload_file(
            Filename=str(local_path),
            Bucket=self.bucket,
            Key=full_key,
            ExtraArgs=extra if extra else None,
            Config=self.transfer_config,
        )
        return full_key

    def upload_bytes(self, data: bytes, key: str, extra_args: Optional[Dict[str, Any]] = None) -> str:
        """Upload raw bytes to S3, setting content-type when possible."""
        full_key = self._full_key(key)
        extra: Dict[str, Any] = dict(extra_args or {})
        ctype = _guess_content_type(full_key)
        if ctype:
            extra.setdefault("ContentType", ctype)
        if self.kms_key_id:
            extra.setdefault("ServerSideEncryption", "aws:kms")
            extra.setdefault("SSEKMSKeyId", self.kms_key_id)

        # put_object is efficient for small/medium payloads
        self.client.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=data,
            **({} if not extra else extra),
        )
        return full_key

    def upload_dir(self, local_root: str, key_prefix: str) -> Tuple[int, int]:
        """
        Recursively upload a directory to S3.

        Args:
            local_root: Path to local directory to upload.
            key_prefix: Prefix under which files will be stored (relative to base_prefix).
        Returns:
            (files_count, errors_count)
        """
        root_path = Path(local_root)
        if not root_path.is_dir():
            raise NotADirectoryError(f"Local directory not found: {local_root}")

        files: list[Path] = [p for p in root_path.rglob("*") if p.is_file()]
        errors = 0

        def _upload_one(p: Path) -> None:
            rel = p.relative_to(root_path).as_posix()
            k = _norm_join(key_prefix, rel)
            self.upload_file(str(p), k)

        # Use limited concurrency; boto3 handles per-transfer parallelism too.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {executor.submit(_upload_one, p): p for p in files}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    errors += 1

        return (len(files), errors)

    # ------------------------- URL generation ---------------------------
    def generate_presigned_url(self, key: str, expires: Optional[int] = None) -> str:
        """Generate a presigned GET URL for a key."""
        full_key = self._full_key(key)
        expiration = int(expires or self.presign_expiration)
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": full_key},
            ExpiresIn=expiration,
        )

    def url_for(self, key: str, *, prefer_presigned: bool = False) -> str:
        """
        Return a URL for accessing the key.
        - If S3_PUBLIC_BASE_URL is set, return "{PUBLIC_BASE_URL}/{full_key}".
        - Else, when prefer_presigned=True, return a presigned GET.
        - Else, return the standard virtual-hosted–style S3 URL.
        """
        full_key = self._full_key(key)
        if self.public_base_url:
            return _norm_join(self.public_base_url, full_key)
        if prefer_presigned:
            return self.generate_presigned_url(full_key)
        # Virtual-hosted–style URL
        return f"https://{self.bucket}.s3.{self.aws_region}.amazonaws.com/{full_key}"

    # --------------------------- manifest -------------------------------
    def write_manifest(self, topic_prefix: str, manifest_dict: Dict[str, Any]) -> str:
        """Write a manifest.json under the given topic prefix."""
        key = _norm_join(topic_prefix, "manifest.json")
        data = json.dumps(manifest_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.upload_bytes(data, key, extra_args={"ContentType": "application/json"})

    # ------------------------ Convex Integration Methods ------------------------

    def upload_with_convex_sync(
        self,
        local_path: str,
        key: str,
        convex_sync: "ConvexS3Sync",
        entity_id: str,
        entity_type: str,
        asset_type: "AssetType",
        extra_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Upload a file to S3 and synchronize metadata with Convex.
        
        This method combines S3 upload with immediate Convex synchronization,
        ensuring that asset metadata is properly maintained across both systems.
        
        Args:
            local_path: Path to the local file to upload
            key: S3 key for the uploaded file
            convex_sync: ConvexS3Sync service instance for database synchronization
            entity_id: ID of the entity (scene or session) this asset belongs to
            entity_type: Type of entity ("scene" or "session")
            asset_type: Type of asset being uploaded
            extra_args: Optional extra arguments for S3 upload
            
        Returns:
            Tuple of (s3_key, s3_url) for the uploaded file
            
        Raises:
            FileNotFoundError: If local file doesn't exist
            Exception: If S3 upload or Convex sync fails
        """
        try:
            # First, upload the file to S3
            logger.info(f"Uploading {local_path} to S3 key {key}")
            full_key = self.upload_file(local_path, key, extra_args)
            
            # Generate the URL for the uploaded file
            s3_url = self.url_for(full_key)
            
            # Calculate file metadata
            file_path = Path(local_path)
            file_size = file_path.stat().st_size
            file_checksum = self._calculate_file_checksum(local_path)
            content_type = _guess_content_type(local_path) or _guess_content_type(full_key)
            
            # Prepare sync request
            from backend.app.services.convex_s3_sync import AssetSyncRequest, SyncOperation
            
            sync_request = AssetSyncRequest(
                entity_id=entity_id,
                entity_type=entity_type,
                s3_key=full_key,
                s3_url=s3_url,
                asset_type=asset_type,
                content_type=content_type or "application/octet-stream",
                size=file_size,
                checksum=file_checksum,
                operation=SyncOperation.CREATE
            )
            
            # Synchronize with Convex
            logger.info(f"Syncing asset metadata to Convex for entity {entity_id}")
            
            if entity_type == "scene":
                # For scenes, use the scene-specific sync method
                s3_assets = {asset_type.value: s3_url}
                metadata = {
                    f"{asset_type.value}_size": file_size,
                    f"{asset_type.value}_checksum": file_checksum,
                    f"{asset_type.value}_content_type": content_type
                }
                
                # Use asyncio to run the async method
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.run_until_complete(
                    convex_sync.sync_scene_assets(entity_id, s3_assets, metadata)
                )
                
            elif entity_type == "session":
                # For sessions, use the session-specific sync method
                combined_assets = {asset_type.value: s3_url}
                
                # Use asyncio to run the async method
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.run_until_complete(
                    convex_sync.sync_session_assets(entity_id, combined_assets)
                )
            
            logger.info(f"Successfully uploaded and synced asset {full_key}")
            return (full_key, s3_url)
            
        except Exception as e:
            logger.error(f"Failed to upload with Convex sync: {str(e)}")
            
            # If Convex sync fails but S3 upload succeeded, we should clean up S3
            # or mark the asset for retry
            try:
                if 'full_key' in locals():
                    logger.warning(f"S3 upload succeeded but Convex sync failed. Asset {full_key} may be orphaned.")
                    # In a production system, you might want to:
                    # 1. Queue the sync operation for retry
                    # 2. Mark the asset as "sync_pending" in a separate tracking system
                    # 3. Clean up the S3 object if sync is critical
            except Exception as cleanup_error:
                logger.error(f"Failed to handle cleanup after sync failure: {str(cleanup_error)}")
            
            raise

    def get_asset_manifest(self, key_prefix: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive asset information for all objects under a key prefix.
        
        This method provides detailed information about all assets stored under
        a specific S3 key prefix, including metadata, sizes, and URLs.
        
        Args:
            key_prefix: S3 key prefix to search for assets
            
        Returns:
            Dictionary containing comprehensive asset information:
            {
                "assets": [
                    {
                        "key": "path/to/file.mp4",
                        "url": "https://...",
                        "size": 1024,
                        "content_type": "video/mp4",
                        "last_modified": "2024-01-01T00:00:00Z",
                        "etag": "abc123",
                        "checksum": "sha256:..."
                    }
                ],
                "total_size": 2048,
                "total_count": 2,
                "prefix": "path/to/",
                "generated_at": "2024-01-01T00:00:00Z"
            }
            
        Raises:
            Exception: If S3 listing operation fails
        """
        try:
            logger.info(f"Retrieving asset manifest for prefix: {key_prefix}")
            
            full_prefix = self._full_key(key_prefix)
            assets = []
            total_size = 0
            
            # Use paginator to handle large numbers of objects
            paginator = self.client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=full_prefix
            )
            
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    last_modified = obj['LastModified'].isoformat()
                    etag = obj['ETag'].strip('"')
                    
                    # Generate URL for the asset
                    relative_key = key
                    if self.base_prefix and key.startswith(self.base_prefix + "/"):
                        relative_key = key[len(self.base_prefix) + 1:]
                    
                    asset_url = self.url_for(relative_key)
                    
                    # Guess content type from key
                    content_type = _guess_content_type(key) or "application/octet-stream"
                    
                    # Try to get additional metadata
                    try:
                        head_response = self.client.head_object(Bucket=self.bucket, Key=key)
                        metadata = head_response.get('Metadata', {})
                        content_type = head_response.get('ContentType', content_type)
                        
                        # Calculate checksum if available
                        checksum = metadata.get('checksum', f"etag:{etag}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get metadata for {key}: {str(e)}")
                        checksum = f"etag:{etag}"
                        metadata = {}
                    
                    asset_info = {
                        "key": key,
                        "relative_key": relative_key,
                        "url": asset_url,
                        "size": size,
                        "content_type": content_type,
                        "last_modified": last_modified,
                        "etag": etag,
                        "checksum": checksum,
                        "metadata": metadata
                    }
                    
                    assets.append(asset_info)
                    total_size += size
            
            manifest = {
                "assets": assets,
                "total_size": total_size,
                "total_count": len(assets),
                "prefix": full_prefix,
                "generated_at": self._get_current_timestamp()
            }
            
            logger.info(f"Generated manifest for {len(assets)} assets under prefix {key_prefix}")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to get asset manifest for prefix {key_prefix}: {str(e)}")
            raise

    def cleanup_with_convex_sync(
        self,
        keys: List[str],
        convex_sync: "ConvexS3Sync",
        entity_id: str,
        entity_type: str = "scene"
    ) -> bool:
        """
        Delete files from S3 and update Convex records accordingly.
        
        This method performs coordinated cleanup of assets, ensuring that both
        S3 objects and Convex database records are properly updated.
        
        Args:
            keys: List of S3 keys to delete
            convex_sync: ConvexS3Sync service instance for database synchronization
            entity_id: ID of the entity these assets belong to
            entity_type: Type of entity ("scene" or "session")
            
        Returns:
            True if all operations succeeded, False if any failures occurred
            
        Raises:
            Exception: If critical cleanup operations fail
        """
        if not keys:
            logger.info("No keys provided for cleanup")
            return True
            
        try:
            logger.info(f"Starting cleanup of {len(keys)} assets for {entity_type} {entity_id}")
            
            # Convert relative keys to full keys
            full_keys = [self._full_key(key) for key in keys]
            
            # Track which deletions succeed/fail
            successful_deletions = []
            failed_deletions = []
            
            # Delete objects from S3
            if len(full_keys) == 1:
                # Single object deletion
                try:
                    self.client.delete_object(Bucket=self.bucket, Key=full_keys[0])
                    successful_deletions.append(full_keys[0])
                    logger.info(f"Successfully deleted S3 object: {full_keys[0]}")
                except Exception as e:
                    logger.error(f"Failed to delete S3 object {full_keys[0]}: {str(e)}")
                    failed_deletions.append((full_keys[0], str(e)))
            else:
                # Batch deletion for multiple objects
                delete_objects = [{"Key": key} for key in full_keys]
                
                try:
                    response = self.client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": delete_objects}
                    )
                    
                    # Track successful deletions
                    for deleted in response.get("Deleted", []):
                        successful_deletions.append(deleted["Key"])
                        logger.info(f"Successfully deleted S3 object: {deleted['Key']}")
                    
                    # Track failed deletions
                    for error in response.get("Errors", []):
                        failed_deletions.append((error["Key"], error["Message"]))
                        logger.error(f"Failed to delete S3 object {error['Key']}: {error['Message']}")
                        
                except Exception as e:
                    logger.error(f"Failed to perform batch deletion: {str(e)}")
                    # If batch deletion fails, treat all as failed
                    failed_deletions.extend([(key, str(e)) for key in full_keys])
            
            # Update Convex records to reflect the cleanup
            if successful_deletions:
                try:
                    logger.info(f"Updating Convex records after successful deletion of {len(successful_deletions)} assets")
                    
                    # Prepare asset updates to clear the deleted assets
                    asset_updates = {}
                    
                    if entity_type == "scene":
                        # For scenes, we need to clear specific asset fields
                        # This is a simplified approach - in practice, you'd want to
                        # identify which specific asset fields to clear based on the deleted keys
                        asset_updates = {
                            "assetsStatus": "ready",  # Maintain ready status after cleanup
                            "assetsErrorMessage": None,
                            "assetVersion": asset_updates.get("assetVersion", 1) + 1
                        }
                        
                        # Use asyncio to run the async method
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        # Update the scene document
                        loop.run_until_complete(
                            convex_sync._update_scene_document(entity_id, asset_updates)
                        )
                        
                    elif entity_type == "session":
                        # For sessions, clear combined asset fields as needed
                        asset_updates = {
                            "assetsStatus": "ready",
                            "assetsErrorMessage": None,
                            "assetCount": max(0, asset_updates.get("assetCount", 0) - len(successful_deletions))
                        }
                        
                        # Use asyncio to run the async method
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        # Update the session document
                        loop.run_until_complete(
                            convex_sync._update_session_document(entity_id, asset_updates)
                        )
                    
                    logger.info(f"Successfully updated Convex records after cleanup")
                    
                except Exception as e:
                    logger.error(f"Failed to update Convex records after S3 cleanup: {str(e)}")
                    # This is a critical error - S3 objects are deleted but Convex is inconsistent
                    raise Exception(f"Data consistency error: S3 cleanup succeeded but Convex update failed: {str(e)}")
            
            # Log summary
            total_requested = len(keys)
            total_successful = len(successful_deletions)
            total_failed = len(failed_deletions)
            
            logger.info(f"Cleanup summary: {total_successful}/{total_requested} successful, {total_failed} failed")
            
            if failed_deletions:
                logger.warning(f"Some deletions failed: {failed_deletions}")
            
            # Return True only if all operations succeeded
            return total_failed == 0
            
        except Exception as e:
            logger.error(f"Critical error during cleanup with Convex sync: {str(e)}")
            raise

    # ------------------------ Private Helper Methods ------------------------

    def _calculate_file_checksum(self, file_path: str, algorithm: str = "sha256") -> str:
        """
        Calculate checksum for a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Checksum string in format "algorithm:hash"
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return f"{algorithm}:{hash_obj.hexdigest()}"

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
