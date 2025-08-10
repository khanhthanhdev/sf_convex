# Video Generation API Documentation

This document describes the FastAPI multi-agent system refinement for video generation and scene editing, implementing the requirements from section 1.2 of the project specification.

## Overview

The video generation API provides endpoints for:
- **Video Generation**: Create new videos with multiple scenes
- **Scene Editing**: Edit specific scenes based on user prompts
- **Background Processing**: Asynchronous task handling with progress tracking
- **S3 Integration**: Automatic upload and management of video assets
- **Convex Synchronization**: Database updates with asset metadata

## Architecture

### Components

1. **FastAPI Endpoints** (`backend/app/api/video_generation.py`)
   - RESTful API endpoints for video operations
   - Request validation and error handling
   - Background task coordination

2. **Orchestration Tasks** (`backend/app/tasks/video_generation_tasks.py`)
   - Celery background tasks for video processing
   - Multi-agent coordination
   - S3 upload and Convex sync orchestration

3. **S3 Utilities** (`backend/app/services/s3_utilities.py`)
   - Helper functions for S3 operations
   - Asset upload and management
   - URL generation and presigned URLs

4. **Convex Integration** (`backend/app/services/convex_s3_sync.py`)
   - Database synchronization with S3 assets
   - Asset metadata management
   - Error handling and retry logic

## API Structure

The video generation API is organized into two main groups:

1. **High-Level Workflows** (`/api/v1/video-generation/`): Complete orchestrated workflows for end-users
2. **Low-Level Agents** (`/api/v1/video/agents/`): Individual agent operations for developers

This document focuses on the high-level workflows. For low-level agent operations, see the [API Structure Documentation](api_structure.md).

## High-Level Workflow Endpoints

### 1. Generate Video

**POST** `/api/v1/video-generation/generate`

Creates a new video with multiple scenes through a complete orchestrated workflow.

#### Request Body

```json
{
  "project_id": "string",
  "content_prompt": "string",
  "scene_count": 3,
  "session_id": "string (optional)",
  "model_config": {
    "planner_model": "gpt-4o-mini",
    "scene_model": "gpt-4o-mini (optional)",
    "helper_model": "gpt-4o-mini (optional)",
    "temperature": 0.7,
    "max_tokens": 4000,
    "use_rag": false,
    "use_visual_fix": false
  },
  "render_config": {
    "quality": "medium",
    "resolution": "1080p",
    "fps": 30,
    "max_scene_duration": 30,
    "use_hardware_acceleration": false,
    "max_retries": 3
  },
  "s3_config": {
    "bucket_name": "string (optional)",
    "base_prefix": "string",
    "public_base_url": "string (optional)",
    "enable_versioning": true,
    "storage_class": "STANDARD"
  }
}
```

#### Response

```json
{
  "task_id": "string",
  "project_id": "string",
  "session_id": "string",
  "status": "queued",
  "message": "string",
  "estimated_duration_minutes": 6,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 2. Edit Scene

**POST** `/api/v1/video-generation/edit-scene`

Edits a specific scene based on user prompt.

#### Request Body

```json
{
  "scene_id": "string",
  "project_id": "string",
  "edit_prompt": "string",
  "session_id": "string (optional)",
  "model_config": {
    "planner_model": "gpt-4o-mini",
    "temperature": 0.7
  },
  "render_config": {
    "quality": "medium",
    "max_retries": 3
  },
  "preserve_timing": true,
  "version_increment": true
}
```

#### Response

```json
{
  "task_id": "string",
  "scene_id": "string",
  "project_id": "string",
  "status": "queued",
  "message": "string",
  "estimated_duration_minutes": 3,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 3. Get Task Status

**GET** `/api/v1/video-generation/tasks/{task_id}/status`

Retrieves the current status and progress of a task.

#### Response

```json
{
  "task_id": "string",
  "status": "progress",
  "progress": 45,
  "message": "Rendering scene 2 of 3...",
  "current_step": "string",
  "total_steps": 6,
  "estimated_completion": "2024-01-01T00:10:00Z",
  "project_id": "string",
  "session_id": "string",
  "scene_id": "string (for edit tasks)",
  "result": null,
  "error": null,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:05:00Z"
}
```

### 4. Get Task Result

**GET** `/api/v1/video-generation/tasks/{task_id}/result`

Retrieves the detailed result of a completed task.

#### Response (Video Generation)

```json
{
  "task_id": "string",
  "status": "completed",
  "result": {
    "project_id": "string",
    "session_id": "string",
    "status": "completed",
    "scene_count": 3,
    "scenes": [
      {
        "scene_id": "string",
        "scene_number": 1,
        "title": "Scene 1",
        "duration": 10.0,
        "status": "completed",
        "s3_video_url": "https://bucket.s3.amazonaws.com/...",
        "s3_source_url": "https://bucket.s3.amazonaws.com/...",
        "s3_thumbnail_url": "https://bucket.s3.amazonaws.com/...",
        "version": 1
      }
    ],
    "total_duration": 30.0,
    "s3_combined_video_url": "https://bucket.s3.amazonaws.com/...",
    "s3_manifest_url": "https://bucket.s3.amazonaws.com/...",
    "generation_time_seconds": 120.5,
    "completed_at": "2024-01-01T00:10:00Z"
  },
  "completed_at": "2024-01-01T00:10:00Z"
}
```

### 5. Cancel Task

**DELETE** `/api/v1/video-generation/tasks/{task_id}`

Cancels a running task.

#### Response

```json
{
  "task_id": "string",
  "message": "Task cancellation requested",
  "status": "cancelled"
}
```

### 6. Health Check

**GET** `/api/v1/video-generation/health`

Checks the health of the video generation service.

#### Response

```json
{
  "status": "healthy",
  "service": "video-generation",
  "message": "Video generation service is operational"
}
```

## Workflow

### Video Generation Workflow

1. **Request Processing**
   - Validate request parameters
   - Generate session ID if not provided
   - Queue background task

2. **Scene Planning**
   - Analyze content prompt
   - Generate scene outline using planner model
   - Create implementation plans for each scene

3. **Scene Generation**
   - Generate Manim code for each scene
   - Render scenes concurrently
   - Handle code fixing with visual feedback

4. **Asset Upload**
   - Upload individual scene videos to S3
   - Upload source code files to S3
   - Generate thumbnails and subtitles

5. **Video Combination**
   - Combine individual scenes into final video
   - Upload combined video to S3
   - Create and upload manifest file

6. **Database Sync**
   - Update Convex with S3 URLs
   - Set asset status and metadata
   - Handle error states

### Scene Editing Workflow

1. **Request Processing**
   - Validate scene ID and edit prompt
   - Queue background task

2. **Source Retrieval**
   - Retrieve original Manim source code from S3
   - Load scene metadata from Convex

3. **Code Modification**
   - Use Code Generation Agent with edit prompt
   - Apply MCP context for enhanced editing
   - Generate modified Manim code

4. **Re-rendering**
   - Render modified scene
   - Generate new thumbnails
   - Create version increment

5. **Asset Upload**
   - Upload new scene version to S3
   - Upload updated source code
   - Maintain version history

6. **Database Update**
   - Update Convex with new asset URLs
   - Increment version number
   - Preserve edit history

## Configuration

### Environment Variables

```bash
# S3 Configuration
S3_BUCKET=your-video-bucket
S3_BASE_PREFIX=projects/
S3_PUBLIC_BASE_URL=https://cdn.example.com
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1

# Convex Configuration
CONVEX_DEPLOYMENT_URL=https://your-deployment.convex.cloud
CONVEX_WEBHOOK_SECRET=your-webhook-secret

# Celery Configuration
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379
```

### Model Configuration

The system supports various AI models through the `model_config` parameter:

- **planner_model**: Used for scene planning and high-level decisions
- **scene_model**: Used for individual scene generation (defaults to planner_model)
- **helper_model**: Used for auxiliary tasks like code fixing

Supported models include:
- `gpt-4o-mini`
- `gpt-4o`
- `claude-3-sonnet`
- `claude-3-haiku`

### Render Configuration

Video rendering can be customized through the `render_config` parameter:

- **quality**: `preview`, `low`, `medium`, `high`, `production`
- **resolution**: `720p`, `1080p`, `1440p`, `4k`
- **fps**: Frame rate (15-60)
- **max_scene_duration**: Maximum scene length in seconds
- **use_hardware_acceleration**: Enable GPU acceleration
- **max_retries**: Maximum retry attempts for failed renders

## Error Handling

The API implements comprehensive error handling:

### HTTP Status Codes

- `200`: Success
- `202`: Task accepted (for async operations)
- `400`: Bad request (validation errors)
- `404`: Task or resource not found
- `500`: Internal server error

### Task Status Values

- `queued`: Task is waiting to be processed
- `progress`: Task is currently running
- `completed`: Task finished successfully
- `failed`: Task encountered an error
- `cancelled`: Task was cancelled by user

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Usage Examples

### Python Client Example

```python
import asyncio
import httpx

async def generate_video():
    async with httpx.AsyncClient() as client:
        # Start video generation
        response = await client.post(
            "http://localhost:8000/api/v1/video-generation/generate",
            json={
                "project_id": "my_project",
                "content_prompt": "Explain machine learning basics",
                "scene_count": 3
            }
        )
        task_data = response.json()
        task_id = task_data["task_id"]
        
        # Poll for completion
        while True:
            status_response = await client.get(
                f"http://localhost:8000/api/v1/video-generation/tasks/{task_id}/status"
            )
            status = status_response.json()
            
            print(f"Status: {status['status']} - {status['message']}")
            
            if status["status"] == "completed":
                # Get final result
                result_response = await client.get(
                    f"http://localhost:8000/api/v1/video-generation/tasks/{task_id}/result"
                )
                result = result_response.json()
                print(f"Video URL: {result['result']['s3_combined_video_url']}")
                break
            elif status["status"] == "failed":
                print(f"Generation failed: {status['error']}")
                break
            
            await asyncio.sleep(5)

# Run the example
asyncio.run(generate_video())
```

### cURL Examples

```bash
# Generate video
curl -X POST "http://localhost:8000/api/v1/video-generation/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "example_project",
    "content_prompt": "Create a video about Python programming basics",
    "scene_count": 3
  }'

# Check task status
curl "http://localhost:8000/api/v1/video-generation/tasks/{task_id}/status"

# Edit scene
curl -X POST "http://localhost:8000/api/v1/video-generation/edit-scene" \
  -H "Content-Type: application/json" \
  -d '{
    "scene_id": "session_123_scene_1",
    "project_id": "example_project",
    "edit_prompt": "Add more colorful animations and emphasize key points"
  }'
```

## Performance Considerations

### Concurrency

- Scene generation is processed concurrently (default: 3 scenes)
- S3 uploads use multipart upload for large files
- Celery workers can be scaled horizontally

### Caching

- Generated scenes are cached in S3 with versioning
- Source code is preserved for future edits
- Thumbnails and metadata are cached

### Optimization

- Use hardware acceleration when available
- Implement progressive quality rendering
- Batch S3 operations for efficiency
- Use CDN for asset delivery

## Monitoring and Logging

### Metrics

The system provides metrics for:
- Task completion rates
- Average generation time per scene
- S3 upload success rates
- Error rates by type

### Logging

Structured logging includes:
- Task lifecycle events
- S3 operation results
- Convex sync status
- Error details with context

### Health Checks

Regular health checks monitor:
- Celery worker availability
- S3 connectivity
- Convex database connectivity
- Agent service status

## Security

### Authentication

- API endpoints support JWT authentication
- S3 operations use IAM roles
- Convex operations use secure tokens

### Data Protection

- S3 objects can use server-side encryption
- Presigned URLs have configurable expiration
- Asset access is controlled by project permissions

### Input Validation

- All requests are validated using Pydantic schemas
- Content prompts are sanitized
- File uploads are scanned for malicious content

## Troubleshooting

### Common Issues

1. **Task Stuck in Queue**
   - Check Celery worker status
   - Verify Redis connectivity
   - Review worker logs

2. **S3 Upload Failures**
   - Verify AWS credentials
   - Check bucket permissions
   - Review network connectivity

3. **Convex Sync Errors**
   - Verify deployment URL
   - Check webhook secret
   - Review database schema

4. **Rendering Failures**
   - Check Manim installation
   - Verify system dependencies
   - Review generated code for errors

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

This provides verbose output for troubleshooting issues.

## Future Enhancements

### Planned Features

- Real-time progress streaming via WebSockets
- Advanced scene templates and presets
- Collaborative editing capabilities
- Integration with external video services
- Advanced analytics and reporting

### Performance Improvements

- GPU-accelerated rendering
- Distributed scene processing
- Intelligent caching strategies
- Progressive video delivery

### API Extensions

- Webhook notifications for task completion
- Bulk operations for multiple videos
- Advanced filtering and search
- Export to various video formats