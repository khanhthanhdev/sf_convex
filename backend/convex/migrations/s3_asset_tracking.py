"""
Migration script to add S3 asset tracking fields to existing records.

This migration:
1. Adds new S3 asset tracking fields to existing Scene records
2. Adds new S3 asset tracking fields to existing VideoSession records
3. Migrates existing s3ChunkKey/s3ChunkUrl data to new videoAsset structure
4. Creates AssetReference records for existing assets
5. Sets default values for new fields

Run this migration after deploying the new schema.
"""

from typing import Dict, List, Optional, Any
from convex import ConvexClient
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3AssetMigration:
    def __init__(self, convex_client: ConvexClient):
        self.client = convex_client
        
    def migrate_scenes(self) -> Dict[str, int]:
        """
        Migrate existing Scene records to include S3 asset tracking fields.
        
        Returns:
            Dictionary with migration statistics
        """
        logger.info("Starting Scene migration...")
        
        # Get all existing scenes
        scenes = self.client.query("listAllDocuments", {"collection": "scenes"})
        
        migrated_count = 0
        error_count = 0
        asset_references_created = 0
        
        for scene in scenes:
            try:
                scene_id = scene["_id"]
                updates = {}
                
                # Set default values for new fields if they don't exist
                if "assetsStatus" not in scene:
                    # Determine status based on existing data
                    if scene.get("s3ChunkUrl"):
                        updates["assetsStatus"] = "ready"
                    elif scene.get("status") == "error":
                        updates["assetsStatus"] = "error"
                        updates["assetsErrorMessage"] = scene.get("errorMessage")
                    else:
                        updates["assetsStatus"] = "pending"
                
                if "assetVersion" not in scene:
                    updates["assetVersion"] = 1
                    
                if "previousAssetVersions" not in scene:
                    updates["previousAssetVersions"] = []
                
                # Migrate existing s3ChunkKey/s3ChunkUrl to videoAsset structure
                if scene.get("s3ChunkKey") and scene.get("s3ChunkUrl") and "videoAsset" not in scene:
                    video_asset = {
                        "s3Key": scene["s3ChunkKey"],
                        "s3Url": scene["s3ChunkUrl"],
                        "contentType": "video/mp4",  # Default assumption
                        "size": 0,  # Will need to be updated separately
                        "checksum": scene.get("checksum", ""),
                        "uploadedAt": scene.get("updatedAt", time.time())
                    }
                    updates["videoAsset"] = video_asset
                    
                    # Create AssetReference record
                    asset_ref = {
                        "entityId": scene_id,
                        "entityType": "scene",
                        "assetType": "video_chunk",
                        "s3Key": scene["s3ChunkKey"],
                        "s3Url": scene["s3ChunkUrl"],
                        "contentType": "video/mp4",
                        "size": 0,
                        "checksum": scene.get("checksum", ""),
                        "version": 1,
                        "createdAt": scene.get("updatedAt", time.time()),
                        "metadata": {
                            "migrated_from_legacy": True,
                            "scene_index": scene.get("index"),
                            "project_id": scene.get("projectId")
                        }
                    }
                    
                    self.client.mutation("createDocument", {
                        "collection": "assetReferences",
                        "data": asset_ref
                    })
                    asset_references_created += 1
                
                # Migrate source code asset if exists
                if scene.get("s3SourceKey") and "sourceCodeAsset" not in scene:
                    source_asset = {
                        "s3Key": scene["s3SourceKey"],
                        "s3Url": f"https://your-bucket.s3.amazonaws.com/{scene['s3SourceKey']}",  # Construct URL
                        "contentType": "text/plain",
                        "size": 0,
                        "checksum": "",
                        "uploadedAt": scene.get("updatedAt", time.time())
                    }
                    updates["sourceCodeAsset"] = source_asset
                    
                    # Create AssetReference record for source code
                    asset_ref = {
                        "entityId": scene_id,
                        "entityType": "scene",
                        "assetType": "source_code",
                        "s3Key": scene["s3SourceKey"],
                        "s3Url": source_asset["s3Url"],
                        "contentType": "text/plain",
                        "size": 0,
                        "checksum": "",
                        "version": 1,
                        "createdAt": scene.get("updatedAt", time.time()),
                        "metadata": {
                            "migrated_from_legacy": True,
                            "scene_index": scene.get("index"),
                            "project_id": scene.get("projectId")
                        }
                    }
                    
                    self.client.mutation("createDocument", {
                        "collection": "assetReferences",
                        "data": asset_ref
                    })
                    asset_references_created += 1
                
                # Apply updates if any
                if updates:
                    self.client.mutation("updateDocument", {
                        "collection": "scenes",
                        "id": scene_id,
                        "updates": updates
                    })
                    migrated_count += 1
                    
            except Exception as e:
                logger.error(f"Error migrating scene {scene.get('_id', 'unknown')}: {str(e)}")
                error_count += 1
        
        logger.info(f"Scene migration completed: {migrated_count} migrated, {error_count} errors, {asset_references_created} asset references created")
        
        return {
            "migrated": migrated_count,
            "errors": error_count,
            "asset_references_created": asset_references_created
        }
    
    def migrate_video_sessions(self) -> Dict[str, int]:
        """
        Migrate existing VideoSession records to include S3 asset tracking fields.
        
        Returns:
            Dictionary with migration statistics
        """
        logger.info("Starting VideoSession migration...")
        
        # Get all existing video sessions
        sessions = self.client.query("listAllDocuments", {"collection": "videoSessions"})
        
        migrated_count = 0
        error_count = 0
        
        for session in sessions:
            try:
                session_id = session["_id"]
                updates = {}
                
                # Set default values for new fields if they don't exist
                if "totalAssetSize" not in session:
                    updates["totalAssetSize"] = 0
                    
                if "assetCount" not in session:
                    updates["assetCount"] = 0
                    
                if "assetsStatus" not in session:
                    # Determine status based on existing session status
                    if session.get("status") == "ready":
                        updates["assetsStatus"] = "ready"
                    elif session.get("status") == "error":
                        updates["assetsStatus"] = "error"
                        updates["assetsErrorMessage"] = session.get("errorMessage")
                    else:
                        updates["assetsStatus"] = "pending"
                
                # Apply updates if any
                if updates:
                    self.client.mutation("updateDocument", {
                        "collection": "videoSessions",
                        "id": session_id,
                        "updates": updates
                    })
                    migrated_count += 1
                    
            except Exception as e:
                logger.error(f"Error migrating video session {session.get('_id', 'unknown')}: {str(e)}")
                error_count += 1
        
        logger.info(f"VideoSession migration completed: {migrated_count} migrated, {error_count} errors")
        
        return {
            "migrated": migrated_count,
            "errors": error_count
        }
    
    def run_migration(self) -> Dict[str, Any]:
        """
        Run the complete migration process.
        
        Returns:
            Dictionary with complete migration statistics
        """
        logger.info("Starting S3 Asset Tracking migration...")
        
        start_time = time.time()
        
        # Run migrations
        scene_results = self.migrate_scenes()
        session_results = self.migrate_video_sessions()
        
        end_time = time.time()
        duration = end_time - start_time
        
        results = {
            "migration_duration_seconds": duration,
            "scenes": scene_results,
            "video_sessions": session_results,
            "total_records_migrated": scene_results["migrated"] + session_results["migrated"],
            "total_errors": scene_results["errors"] + session_results["errors"],
            "asset_references_created": scene_results["asset_references_created"]
        }
        
        logger.info(f"Migration completed in {duration:.2f} seconds")
        logger.info(f"Total records migrated: {results['total_records_migrated']}")
        logger.info(f"Total errors: {results['total_errors']}")
        logger.info(f"Asset references created: {results['asset_references_created']}")
        
        return results

def run_migration(convex_deployment_url: str) -> Dict[str, Any]:
    """
    Main function to run the migration.
    
    Args:
        convex_deployment_url: The Convex deployment URL
        
    Returns:
        Migration results
    """
    client = ConvexClient(convex_deployment_url)
    migration = S3AssetMigration(client)
    return migration.run_migration()

if __name__ == "__main__":
    # Replace with your actual Convex deployment URL
    CONVEX_URL = "your-convex-deployment-url"
    
    try:
        results = run_migration(CONVEX_URL)
        print("Migration completed successfully!")
        print(f"Results: {results}")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise