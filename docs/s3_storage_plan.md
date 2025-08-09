### S3 Storage Integration Plan

This plan introduces a storage service abstraction for uploading and serving all agent outputs from AWS S3, mirroring the local folder structure per topic.

### Step 1 — Configuration (env-driven) [Done]
- **env vars**: `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (opt), `S3_BUCKET`, `S3_BASE_PREFIX`, `S3_PUBLIC_BASE_URL`, `S3_KMS_KEY_ID`, `S3_UPLOAD_ON_WRITE`, `S3_MAX_CONCURRENCY`, `S3_MULTIPART_THRESHOLD_MB`, `S3_ENDPOINT_URL`, `S3_FORCE_PATH_STYLE`, `S3_PRESIGN_EXPIRATION`.
- **backend**: `backend/app/core/settings.py` exposes the above.
- **agents**: `agents/src/config/config.py` mirrors the same.

### Step 2 — S3 Client Factory [Done]
- **file**: `agents/src/storage/s3_client.py`.
- **get_s3_client()**: region, `signature_version='v4'`, retries `{ total_max_attempts: 10, mode: 'standard' }`, optional `endpoint_url`, path-style toggle, proxy/VPC endpoint support.
- **get_s3_transfer_config()**: tune `multipart_threshold` and `max_concurrency` from env.

### Step 3 — Storage Service Abstraction
- **file**: `agents/src/storage/s3_storage.py`.
- **class**: `S3Storage(bucket: str, base_prefix: str = "")`.
- **upload_file(local_path, key, extra_args=None)**
  - Uses `client.upload_file` with `TransferConfig`.
  - Auto content-type (via `mimetypes.guess_type`).
  - If `S3_KMS_KEY_ID` set, include `ExtraArgs={'ServerSideEncryption':'aws:kms','SSEKMSKeyId':...}`.
  - Optional canned ACL via `extra_args`.
- **upload_bytes(data: bytes, key, extra_args=None)**
  - Uses `client.put_object` or `upload_fileobj` with `io.BytesIO`.
  - Same SSE-KMS and content-type handling.
- **upload_dir(local_root, key_prefix)**
  - Recursively walks `local_root`.
  - Derives keys by joining `key_prefix` with relative paths.
  - Uploads with `ThreadPoolExecutor(max_workers=S3_MAX_CONCURRENCY)`.
- **generate_presigned_url(key, expires=3600)**
  - Uses `client.generate_presigned_url('get_object', ...)`.
- **url_for(key)**
  - If `S3_PUBLIC_BASE_URL` set: return `S3_PUBLIC_BASE_URL + '/' + key`.
  - Else return S3 virtual-host URL (`https://{bucket}.s3.{region}.amazonaws.com/{key}`) or presigned URL (optional behavior flag).
- **write_manifest(topic_prefix, manifest_dict)** (optional)
  - Writes `topic_prefix/manifest.json` including checksums, sizes, and URLs.

### Step 4 — Key and Path Strategy
- **Goal**: Mirror local structure exactly in S3.
- **Root prefix**: `topic-name/` (sanitize to kebab-case; strip spaces/punctuation).
- **Examples**:
  - `topic-name/media/images/chain_rule_scene1_v2/...`
  - `topic-name/media/Tex/<hash>.svg` and `topic-name/media/Tex/<hash>.tex`
  - `topic-name/media/videos/chain_rule_scene1_v3/720p30/Scene1.mp4`
  - `topic-name/scene1/code/chain_rule_scene1_v3.py`
  - `topic-name/scene1/subplans/chain_rule_scene1_animation_narration_plan.txt`
- **Rule**: strip local `output_dir` from absolute path; prepend `topic-name/` to the remaining relative path to form the S3 key.

### Step 5 — Integration Points (minimal invasive)
- **Orchestrator**: `agents/generate_video.py`
  - After topic finishes (or at milestones), call `S3Storage.upload_dir(topic_dir, topic_name + "/")`.
  - Store/return an index of S3 URLs (CDN or presigned) for important artifacts.
- **Renderer**: `agents/src/core/video_renderer.py`
  - After producing combined MP4/SRT, upload those immediately; record S3 URLs.
- **Planner/Generator**: `agents/src/core/video_planner.py`, `agents/src/core/code_generator.py`
  - After writing scene plans/code: if `S3_UPLOAD_ON_WRITE=true`, upload the single file. Otherwise rely on final `upload_dir`.
- **Parser**: `agents/src/core/parse_video.py`
  - After saving frames/text, conditionally upload on write based on `S3_UPLOAD_ON_WRITE`.
- **Backend**: `backend/app/adapters/video_adapters.py`, `backend/app/tasks/video_tasks.py`
  - Where returning `output_path`, upload and return S3 URL(s) instead.
  - Add presigned GET path when `S3_PUBLIC_BASE_URL` is not set (private buckets).

### Step 6 — Security and Access
- Default to private bucket; serve with presigned GETs.
- For public/global delivery, use CloudFront with Origin Access Control (OAC).
- If `S3_KMS_KEY_ID` present, enable SSE-KMS on uploads.
- Configure S3 bucket CORS when direct browser access is required.

### Step 7 — Reliability and Performance
- Retries: standard mode, `total_max_attempts=10`.
- Multipart uploads for large MP4s via `TransferConfig(multipart_threshold=64MB, max_concurrency=S3_MAX_CONCURRENCY)`.
- Optional progress callbacks for CLI mode only.
- Idempotency: overwrite same key; enable S3 versioning on bucket to retain history.

### Step 8 — Observability
- Structured logs per upload (key, size, duration).
- Warn on failures with context; include retry counts if available.
- Optional `manifest.json` per topic listing files, sizes, SHA256, and URLs.

### Step 9 — Dev Workflow
- Toggle S3 off via env; continue writing to local filesystem.
- Optional MinIO for local dev via `S3_ENDPOINT_URL` and `S3_FORCE_PATH_STYLE=true`.

### Step 10 — Data Model and API Surface
- Update `backend/app/schemas/video.py` to include fields like `s3_url`, `source_code_s3_url`, `combined_video_s3_url`.
- Update `backend/app/api/video.py` to return S3 URLs instead of local paths.
- Optional: add endpoint to generate presigned GETs for a given key when using private buckets.

### Step 11 — Migration/Backfill
- Script to bulk upload existing local `output/` directories to S3 using `upload_dir`.
- Backfill datastore (e.g., Convex) with generated S3 URLs.

### Step 12 — Testing
- Unit tests for `S3Storage` using moto or MinIO.
- Integration test for a small topic run: verify expected keys exist and manifest matches.
- Load test multi-scene uploads with configured concurrency.

### Step 13 — Rollout
- Create bucket with versioning and lifecycle (expire `partial_movie_files/` after N days).
- Apply bucket policy and KMS permissions.
- Deploy with `S3_UPLOAD_ON_WRITE=false` first, rely on end-of-run `upload_dir`.
- Flip `S3_UPLOAD_ON_WRITE=true` for near-real-time sync after validation.

### Acceptance Criteria
- All artifacts for a topic are mirrored in S3 under `topic-name/` with the same structure as local output.
- Backend and agents return/record working URLs (CDN or presigned) for key artifacts.
- Large files upload reliably with retries and multipart; uploads are observable in logs.

