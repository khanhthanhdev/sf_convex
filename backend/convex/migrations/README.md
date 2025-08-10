# Convex Database Migrations

This directory contains database migration scripts for the AI Video Tutor platform.

## Migration 001: S3 Asset Tracking

This migration adds S3 asset tracking capabilities to the existing Convex schema.

### What it does:

1. **Extends Scene model** with new S3 asset tracking fields:
   - `videoAsset`, `sourceCodeAsset`, `thumbnailAsset`, `subtitleAsset`
   - `assetsStatus`, `assetsErrorMessage`
   - `assetVersion`, `previousAssetVersions`

2. **Extends VideoSession model** with asset summary fields:
   - `combinedVideoAsset`, `combinedSubtitleAsset`, `manifestAsset`
   - `totalAssetSize`, `assetCount`
   - `assetsStatus`, `assetsErrorMessage`

3. **Creates AssetReference collection** for detailed asset tracking

4. **Migrates existing data**:
   - Converts existing `s3ChunkKey`/`s3ChunkUrl` to new `videoAsset` structure
   - Creates `AssetReference` records for existing assets
   - Sets appropriate default values for new fields

### Prerequisites:

1. **Backup your database** before running the migration
2. Ensure the new schema has been deployed to Convex
3. Have your Convex deployment URL ready

### Running the migration:

#### Option 1: Using the utility script (Recommended)

```bash
cd backend/convex/migrations
python run_migration.py --url https://your-deployment.convex.cloud

# For dry-run (preview only):
python run_migration.py --url https://your-deployment.convex.cloud --dry-run
```

#### Option 2: Direct execution

```python
from migrations.add_s3_asset_tracking import run_migration

results = run_migration("https://your-deployment.convex.cloud")
print(results)
```

### Expected Results:

The migration will output statistics including:
- Number of Scene records migrated
- Number of VideoSession records migrated
- Number of AssetReference records created
- Any errors encountered
- Total execution time

### Rollback:

This migration is designed to be backward compatible. The old fields (`s3ChunkKey`, `s3ChunkUrl`, etc.) are preserved, so existing code will continue to work.

If you need to rollback:
1. Deploy the old schema
2. The new fields will be ignored by the old code
3. Remove the `assetReferences` collection if desired

### Verification:

After running the migration, verify:

1. **Scene records** have new asset fields populated
2. **VideoSession records** have asset summary fields
3. **AssetReference collection** contains records for existing assets
4. **Existing functionality** still works with legacy fields

### Troubleshooting:

- **Connection errors**: Verify your Convex deployment URL
- **Permission errors**: Ensure your Convex deployment allows mutations
- **Schema errors**: Make sure the new schema is deployed before running migration
- **Partial failures**: The migration logs detailed error information for debugging

### Next Steps:

After successful migration:
1. Update application code to use new asset fields
2. Test the S3-Convex integration functionality
3. Monitor for any issues in production
4. Consider removing legacy fields in a future migration once fully transitioned