#!/usr/bin/env python3
"""
Validation script to verify S3 asset tracking migration completed successfully.

Usage:
    python validate_migration.py --url <convex-deployment-url>
"""

import argparse
from typing import Dict, List, Any
from convex import ConvexClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationValidator:
    def __init__(self, convex_client: ConvexClient):
        self.client = convex_client
    
    def validate_scenes(self) -> Dict[str, Any]:
        """Validate Scene records have been migrated correctly."""
        logger.info("Validating Scene records...")
        
        scenes = self.client.query("listAllDocuments", {"collection": "scenes"})
        
        total_scenes = len(scenes)
        scenes_with_assets_status = 0
        scenes_with_asset_version = 0
        scenes_with_video_asset = 0
        scenes_with_legacy_data = 0
        
        for scene in scenes:
            if "assetsStatus" in scene:
                scenes_with_assets_status += 1
            if "assetVersion" in scene:
                scenes_with_asset_version += 1
            if "videoAsset" in scene and scene["videoAsset"]:
                scenes_with_video_asset += 1
            if scene.get("s3ChunkKey") or scene.get("s3ChunkUrl"):
                scenes_with_legacy_data += 1
        
        return {
            "total_scenes": total_scenes,
            "scenes_with_assets_status": scenes_with_assets_status,
            "scenes_with_asset_version": scenes_with_asset_version,
            "scenes_with_video_asset": scenes_with_video_asset,
            "scenes_with_legacy_data": scenes_with_legacy_data,
            "migration_coverage": (scenes_with_assets_status / total_scenes * 100) if total_scenes > 0 else 0
        }
    
    def validate_video_sessions(self) -> Dict[str, Any]:
        """Validate VideoSession records have been migrated correctly."""
        logger.info("Validating VideoSession records...")
        
        sessions = self.client.query("listAllDocuments", {"collection": "videoSessions"})
        
        total_sessions = len(sessions)
        sessions_with_assets_status = 0
        sessions_with_asset_count = 0
        sessions_with_total_asset_size = 0
        
        for session in sessions:
            if "assetsStatus" in session:
                sessions_with_assets_status += 1
            if "assetCount" in session:
                sessions_with_asset_count += 1
            if "totalAssetSize" in session:
                sessions_with_total_asset_size += 1
        
        return {
            "total_sessions": total_sessions,
            "sessions_with_assets_status": sessions_with_assets_status,
            "sessions_with_asset_count": sessions_with_asset_count,
            "sessions_with_total_asset_size": sessions_with_total_asset_size,
            "migration_coverage": (sessions_with_assets_status / total_sessions * 100) if total_sessions > 0 else 0
        }
    
    def validate_asset_references(self) -> Dict[str, Any]:
        """Validate AssetReference collection was created and populated."""
        logger.info("Validating AssetReference records...")
        
        try:
            asset_refs = self.client.query("listAllDocuments", {"collection": "assetReferences"})
            
            total_asset_refs = len(asset_refs)
            scene_asset_refs = len([ref for ref in asset_refs if ref.get("entityType") == "scene"])
            session_asset_refs = len([ref for ref in asset_refs if ref.get("entityType") == "session"])
            migrated_refs = len([ref for ref in asset_refs if ref.get("metadata", {}).get("migrated_from_legacy")])
            
            # Check asset types distribution
            asset_types = {}
            for ref in asset_refs:
                asset_type = ref.get("assetType", "unknown")
                asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
            
            return {
                "total_asset_references": total_asset_refs,
                "scene_asset_references": scene_asset_refs,
                "session_asset_references": session_asset_refs,
                "migrated_from_legacy": migrated_refs,
                "asset_types_distribution": asset_types
            }
        except Exception as e:
            return {
                "error": f"Failed to validate asset references: {str(e)}",
                "total_asset_references": 0
            }
    
    def validate_indexes(self) -> Dict[str, Any]:
        """Validate that required indexes exist."""
        logger.info("Validating database indexes...")
        
        # This is a simplified check - in a real implementation,
        # you would query the database schema to verify indexes exist
        try:
            # Try to perform queries that would use the new indexes
            test_queries = [
                ("scenes by assetsStatus", {"collection": "scenes", "filter": {"assetsStatus": "ready"}}),
                ("videoSessions by assetsStatus", {"collection": "videoSessions", "filter": {"assetsStatus": "ready"}}),
                ("assetReferences by entityId", {"collection": "assetReferences", "filter": {"entityId": "test"}})
            ]
            
            successful_queries = 0
            for query_name, query_params in test_queries:
                try:
                    self.client.query("listDocuments", query_params)
                    successful_queries += 1
                except Exception as e:
                    logger.warning(f"Query '{query_name}' failed: {str(e)}")
            
            return {
                "total_test_queries": len(test_queries),
                "successful_queries": successful_queries,
                "indexes_likely_working": successful_queries == len(test_queries)
            }
        except Exception as e:
            return {
                "error": f"Failed to validate indexes: {str(e)}",
                "indexes_likely_working": False
            }
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation and return results."""
        logger.info("Starting migration validation...")
        
        scene_results = self.validate_scenes()
        session_results = self.validate_video_sessions()
        asset_ref_results = self.validate_asset_references()
        index_results = self.validate_indexes()
        
        # Calculate overall migration success
        scene_success = scene_results["migration_coverage"] > 95
        session_success = session_results["migration_coverage"] > 95
        asset_ref_success = asset_ref_results.get("total_asset_references", 0) > 0
        index_success = index_results.get("indexes_likely_working", False)
        
        overall_success = scene_success and session_success and asset_ref_success and index_success
        
        results = {
            "overall_success": overall_success,
            "scenes": scene_results,
            "video_sessions": session_results,
            "asset_references": asset_ref_results,
            "indexes": index_results,
            "summary": {
                "scene_migration_success": scene_success,
                "session_migration_success": session_success,
                "asset_references_created": asset_ref_success,
                "indexes_working": index_success
            }
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Validate S3 asset tracking migration"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Convex deployment URL"
    )
    
    args = parser.parse_args()
    
    print(f"Validating migration for: {args.url}")
    
    try:
        client = ConvexClient(args.url)
        validator = MigrationValidator(client)
        results = validator.run_validation()
        
        print("\n" + "="*50)
        print("MIGRATION VALIDATION RESULTS")
        print("="*50)
        
        if results["overall_success"]:
            print("✅ MIGRATION VALIDATION PASSED")
        else:
            print("❌ MIGRATION VALIDATION FAILED")
        
        print(f"\nScene Migration: {'✅' if results['summary']['scene_migration_success'] else '❌'}")
        print(f"  - Total scenes: {results['scenes']['total_scenes']}")
        print(f"  - Migration coverage: {results['scenes']['migration_coverage']:.1f}%")
        print(f"  - Scenes with video assets: {results['scenes']['scenes_with_video_asset']}")
        
        print(f"\nVideoSession Migration: {'✅' if results['summary']['session_migration_success'] else '❌'}")
        print(f"  - Total sessions: {results['video_sessions']['total_sessions']}")
        print(f"  - Migration coverage: {results['video_sessions']['migration_coverage']:.1f}%")
        
        print(f"\nAsset References: {'✅' if results['summary']['asset_references_created'] else '❌'}")
        print(f"  - Total asset references: {results['asset_references']['total_asset_references']}")
        print(f"  - Scene references: {results['asset_references']['scene_asset_references']}")
        print(f"  - Migrated from legacy: {results['asset_references']['migrated_from_legacy']}")
        
        print(f"\nIndexes: {'✅' if results['summary']['indexes_working'] else '❌'}")
        print(f"  - Test queries successful: {results['indexes']['successful_queries']}/{results['indexes']['total_test_queries']}")
        
        if not results["overall_success"]:
            print("\n⚠️  Some validation checks failed. Review the results above.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\nVALIDATION FAILED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())