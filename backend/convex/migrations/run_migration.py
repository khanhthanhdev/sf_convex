#!/usr/bin/env python3
"""
Utility script to run S3 asset tracking migration.

Usage:
    python run_migration.py --url <convex-deployment-url> [--dry-run]

Options:
    --url: Convex deployment URL (required)
    --dry-run: Run in dry-run mode (preview changes without applying them)
    --help: Show this help message
"""

import argparse
import sys
import os
from typing import Dict, Any

# Add the parent directory to the path so we can import the migration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from migrations.s3_asset_tracking import run_migration

def main():
    parser = argparse.ArgumentParser(
        description="Run S3 asset tracking migration for Convex database"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Convex deployment URL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (preview changes without applying them)"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE: No changes will be applied to the database")
        print("This would run the migration with the following settings:")
        print(f"  Convex URL: {args.url}")
        print("\nTo run the actual migration, remove the --dry-run flag")
        return
    
    print(f"Running S3 asset tracking migration...")
    print(f"Convex URL: {args.url}")
    print("This will modify your database. Make sure you have a backup!")
    
    # Ask for confirmation
    response = input("Do you want to continue? (yes/no): ").lower().strip()
    if response not in ["yes", "y"]:
        print("Migration cancelled.")
        return
    
    try:
        results = run_migration(args.url)
        
        print("\n" + "="*50)
        print("MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Duration: {results['migration_duration_seconds']:.2f} seconds")
        print(f"Total records migrated: {results['total_records_migrated']}")
        print(f"Total errors: {results['total_errors']}")
        print(f"Asset references created: {results['asset_references_created']}")
        print("\nDetailed results:")
        print(f"  Scenes migrated: {results['scenes']['migrated']}")
        print(f"  Scene errors: {results['scenes']['errors']}")
        print(f"  Video sessions migrated: {results['video_sessions']['migrated']}")
        print(f"  Video session errors: {results['video_sessions']['errors']}")
        
        if results['total_errors'] > 0:
            print(f"\nWARNING: {results['total_errors']} errors occurred during migration.")
            print("Check the logs for details.")
        
    except Exception as e:
        print(f"\nMIGRATION FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()