from typing import Dict, List, Optional, Any, Literal
from convex import ConvexClient
from ..types.schema import S3Asset, AssetType, AssetStatus
from .utils import update_document, get_current_timestamp
import os

CONVEX_URL = os.getenv("CONVEX_URL")
# Initialize Convex client
client = ConvexClient(CONVEX_URL)

def update_scene_assets(
    scene_id: str,
    assets: Dict[str, S3Asset],
    assets_status: AssetStatus = "ready",
    assets_error_message: Optional[str] = None,
    increment_version: bool = True
) -> Dict:
    """
    Update a scene with S3 asset information.
    
    Args:
        scene_id: ID of the scene to update
        assets: Dictionary mapping asset types to S3Asset objects
        assets_status: Overall status of the assets
        assets_error_message: Optional error message if status is "error"
        increment_version: Whether to increment the asset version
        
    Returns:
        The updated scene document
    """
    updates = {
        "assetsStatus": assets_status
    }
    
    # Add asset error message if provided
    if assets_error_message:
        updates["assetsErrorMessage"] = assets_error_message
    
    # Update specific asset fields based on provided assets
    for asset_type, asset_data in assets.items():
        if asset_type == "video":
            updates["videoAsset"] = asset_data.dict()
        elif asset_type == "source_code":
            updates["sourceCodeAsset"] = asset_data.dict()
        elif asset_type == "thumbnail":
            updates["thumbnailAsset"] = asset_data.dict()
        elif asset_type == "subtitle":
            updates["subtitleAsset"] = asset_data.dict()
    
    # Handle version management
    if increment_version:
        # Get current scene to preserve previous versions
        current_scene = client.query("getDocument", {"collection": "scenes", "id": scene_id})
        if current_scene:
            current_version = current_scene.get("assetVersion", 1)
            updates["assetVersion"] = current_version + 1
            
            # Preserve previous asset versions
            previous_versions = current_scene.get("previousAssetVersions", [])
            
            # Add current assets to previous versions if they exist
            current_assets = []
            for field in ["videoAsset", "sourceCodeAsset", "thumbnailAsset", "subtitleAsset"]:
                if current_scene.get(field):
                    current_assets.append(current_scene[field])
            
            if current_assets:
                previous_versions.extend(current_assets)
                # Keep only last 5 versions to prevent unbounded growth
                updates["previousAssetVersions"] = previous_versions[-5:]
    
    return update_document("scenes", scene_id, updates)

def update_session_assets(
    session_id: str,
    combined_video_asset: Optional[S3Asset] = None,
    combined_subtitle_asset: Optional[S3Asset] = None,
    manifest_asset: Optional[S3Asset] = None,
    total_asset_size: Optional[int] = None,
    asset_count: Optional[int] = None,
    assets_status: AssetStatus = "ready",
    assets_error_message: Optional[str] = None
) -> Dict:
    """
    Update a video session with combined asset information.
    
    Args:
        session_id: ID of the video session to update
        combined_video_asset: Combined video asset information
        combined_subtitle_asset: Combined subtitle asset information
        manifest_asset: Manifest file asset information
        total_asset_size: Total size of all assets in bytes
        asset_count: Total number of assets
        assets_status: Overall status of the assets
        assets_error_message: Optional error message if status is "error"
        
    Returns:
        The updated video session document
    """
    updates = {
        "assetsStatus": assets_status
    }
    
    # Add asset error message if provided
    if assets_error_message:
        updates["assetsErrorMessage"] = assets_error_message
    
    # Update asset fields if provided
    if combined_video_asset:
        updates["combinedVideoAsset"] = combined_video_asset.dict()
    
    if combined_subtitle_asset:
        updates["combinedSubtitleAsset"] = combined_subtitle_asset.dict()
    
    if manifest_asset:
        updates["manifestAsset"] = manifest_asset.dict()
    
    if total_asset_size is not None:
        updates["totalAssetSize"] = total_asset_size
    
    if asset_count is not None:
        updates["assetCount"] = asset_count
    
    return update_document("videoSessions", session_id, updates)

def batch_update_assets(
    asset_updates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform batch updates of assets for multiple entities.
    
    Args:
        asset_updates: List of update operations, each containing:
            - entity_id: ID of the entity (scene or session)
            - entity_type: Type of entity ("scene" or "session")
            - assets: Asset data to update
            - assets_status: Status of the assets
            - assets_error_message: Optional error message
            
    Returns:
        Dictionary with successful and failed operations
    """
    successful = []
    failed = []
    
    for update_op in asset_updates:
        try:
            entity_id = update_op["entity_id"]
            entity_type = update_op["entity_type"]
            
            if entity_type == "scene":
                result = update_scene_assets(
                    scene_id=entity_id,
                    assets=update_op.get("assets", {}),
                    assets_status=update_op.get("assets_status", "ready"),
                    assets_error_message=update_op.get("assets_error_message"),
                    increment_version=update_op.get("increment_version", True)
                )
                successful.append(entity_id)
                
            elif entity_type == "session":
                result = update_session_assets(
                    session_id=entity_id,
                    combined_video_asset=update_op.get("combined_video_asset"),
                    combined_subtitle_asset=update_op.get("combined_subtitle_asset"),
                    manifest_asset=update_op.get("manifest_asset"),
                    total_asset_size=update_op.get("total_asset_size"),
                    asset_count=update_op.get("asset_count"),
                    assets_status=update_op.get("assets_status", "ready"),
                    assets_error_message=update_op.get("assets_error_message")
                )
                successful.append(entity_id)
                
            else:
                failed.append((entity_id, f"Unknown entity type: {entity_type}"))
                
        except Exception as e:
            failed.append((update_op.get("entity_id", "unknown"), str(e)))
    
    return {
        "successful": successful,
        "failed": failed,
        "total_processed": len(asset_updates)
    }

def delete_asset_references(
    entity_id: str,
    entity_type: Literal["scene", "session"],
    asset_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Delete asset references for cleanup operations.
    
    Args:
        entity_id: ID of the entity (scene or session)
        entity_type: Type of entity ("scene" or "session")
        asset_types: Optional list of specific asset types to delete.
                    If None, all assets are cleared.
                    
    Returns:
        Dictionary with operation results
    """
    updates = {}
    cleared_assets = []
    
    try:
        if entity_type == "scene":
            # Define all possible scene asset fields
            scene_asset_fields = {
                "video": "videoAsset",
                "source_code": "sourceCodeAsset", 
                "thumbnail": "thumbnailAsset",
                "subtitle": "subtitleAsset"
            }
            
            # Clear specific asset types or all if none specified
            if asset_types:
                for asset_type in asset_types:
                    if asset_type in scene_asset_fields:
                        field_name = scene_asset_fields[asset_type]
                        updates[field_name] = None
                        cleared_assets.append(asset_type)
            else:
                # Clear all asset fields
                for asset_type, field_name in scene_asset_fields.items():
                    updates[field_name] = None
                    cleared_assets.append(asset_type)
            
            # Update asset status if all assets are being cleared
            if not asset_types or len(asset_types) == len(scene_asset_fields):
                updates["assetsStatus"] = "pending"
                updates["assetsErrorMessage"] = None
                updates["assetVersion"] = 1
                updates["previousAssetVersions"] = []
            
        elif entity_type == "session":
            # Define all possible session asset fields
            session_asset_fields = {
                "combined_video": "combinedVideoAsset",
                "combined_subtitle": "combinedSubtitleAsset",
                "manifest": "manifestAsset"
            }
            
            # Clear specific asset types or all if none specified
            if asset_types:
                for asset_type in asset_types:
                    if asset_type in session_asset_fields:
                        field_name = session_asset_fields[asset_type]
                        updates[field_name] = None
                        cleared_assets.append(asset_type)
            else:
                # Clear all asset fields
                for asset_type, field_name in session_asset_fields.items():
                    updates[field_name] = None
                    cleared_assets.append(asset_type)
                
                # Reset asset summary fields
                updates["totalAssetSize"] = 0
                updates["assetCount"] = 0
            
            # Update asset status if all assets are being cleared
            if not asset_types or len(asset_types) == len(session_asset_fields):
                updates["assetsStatus"] = "pending"
                updates["assetsErrorMessage"] = None
        
        else:
            return {
                "success": False,
                "error": f"Unknown entity type: {entity_type}",
                "cleared_assets": []
            }
        
        # Perform the update
        collection = "scenes" if entity_type == "scene" else "videoSessions"
        result = update_document(collection, entity_id, updates)
        
        return {
            "success": True,
            "cleared_assets": cleared_assets,
            "entity_id": entity_id,
            "entity_type": entity_type
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cleared_assets": [],
            "entity_id": entity_id,
            "entity_type": entity_type
        }

def get_assets_by_entity(
    entity_id: str,
    entity_type: Literal["scene", "session"],
    include_previous_versions: bool = False
) -> Dict[str, Any]:
    """
    Retrieve asset information for a specific entity.
    
    Args:
        entity_id: ID of the entity (scene or session)
        entity_type: Type of entity ("scene" or "session")
        include_previous_versions: Whether to include previous asset versions
        
    Returns:
        Dictionary containing asset information
    """
    try:
        collection = "scenes" if entity_type == "scene" else "videoSessions"
        entity = client.query("getDocument", {"collection": collection, "id": entity_id})
        
        if not entity:
            return {
                "success": False,
                "error": f"{entity_type.capitalize()} not found",
                "assets": {}
            }
        
        assets = {}
        
        if entity_type == "scene":
            # Extract scene assets
            asset_fields = {
                "video": "videoAsset",
                "source_code": "sourceCodeAsset",
                "thumbnail": "thumbnailAsset", 
                "subtitle": "subtitleAsset"
            }
            
            for asset_type, field_name in asset_fields.items():
                if entity.get(field_name):
                    assets[asset_type] = entity[field_name]
            
            # Include version information
            assets["_metadata"] = {
                "assets_status": entity.get("assetsStatus", "pending"),
                "assets_error_message": entity.get("assetsErrorMessage"),
                "asset_version": entity.get("assetVersion", 1),
                "updated_at": entity.get("updatedAt")
            }
            
            # Include previous versions if requested
            if include_previous_versions:
                assets["_metadata"]["previous_versions"] = entity.get("previousAssetVersions", [])
        
        elif entity_type == "session":
            # Extract session assets
            asset_fields = {
                "combined_video": "combinedVideoAsset",
                "combined_subtitle": "combinedSubtitleAsset",
                "manifest": "manifestAsset"
            }
            
            for asset_type, field_name in asset_fields.items():
                if entity.get(field_name):
                    assets[asset_type] = entity[field_name]
            
            # Include summary information
            assets["_metadata"] = {
                "assets_status": entity.get("assetsStatus", "pending"),
                "assets_error_message": entity.get("assetsErrorMessage"),
                "total_asset_size": entity.get("totalAssetSize", 0),
                "asset_count": entity.get("assetCount", 0),
                "updated_at": entity.get("updatedAt")
            }
        
        return {
            "success": True,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "assets": assets
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "entity_id": entity_id,
            "entity_type": entity_type,
            "assets": {}
        }