# API Structure Documentation

This document explains the organization of the video generation API endpoints and their intended use cases.

## API Organization

The video generation functionality is organized into two main API groups:

### 1. High-Level Workflows (`/api/v1/video-generation/`)

**Purpose**: Complete, orchestrated video generation and editing workflows
**Target Users**: End-users, frontend applications, external integrations
**Features**: 
- Full pipeline automation
- S3 asset management
- Convex database synchronization
- Advanced progress tracking
- Error recovery and retry logic

#### Endpoints:

- **POST** `/api/v1/video-generation/generate`
  - Complete video generation from prompt to final video
  - Handles planning, implementation, rendering, and asset upload
  - Returns session with all scene assets and combined video

- **POST** `/api/v1/video-generation/edit-scene`
  - Edit specific scenes with natural language prompts
  - Retrieves original code, modifies, re-renders, and updates assets
  - Maintains version history

- **GET** `/api/v1/video-generation/tasks/{task_id}/status`
  - Advanced progress tracking with detailed steps
  - Estimated completion times
  - Rich metadata about current operations

### 2. Low-Level Agent Operations (`/api/v1/video/agents/`)

**Purpose**: Direct access to individual video generation components
**Target Users**: Developers, debugging, custom workflows, testing
**Features**:
- Fine-grained control over each step
- Direct agent function access
- Minimal orchestration
- Basic job tracking

#### Endpoints:

- **POST** `/api/v1/video/agents/plan`
  - Generate scene outline using planning agent
  - Returns structured scene plan

- **POST** `/api/v1/video/agents/implement`
  - Generate detailed implementation plans for scenes
  - Takes scene outline, returns implementation details

- **POST** `/api/v1/video/agents/code`
  - Generate Manim code for a specific scene
  - Takes implementation plan, returns executable code

- **POST** `/api/v1/video/agents/render/scene`
  - Render a single scene from Manim code
  - Direct rendering without asset management

- **POST** `/api/v1/video/agents/render/combine`
  - Combine multiple scene videos
  - Basic video concatenation

- **GET** `/api/v1/video/agents/jobs/{job_id}`
  - Basic job status for agent operations
  - Simple status and result retrieval

## Usage Patterns

### For End Users (Recommended)

Use the high-level workflows for most use cases:

```python
# Generate complete video
response = await client.post("/api/v1/video-generation/generate", json={
    "project_id": "my_project",
    "content_prompt": "Explain machine learning basics",
    "scene_count": 3
})

# Edit a scene
edit_response = await client.post("/api/v1/video-generation/edit-scene", json={
    "scene_id": "scene_123",
    "project_id": "my_project", 
    "edit_prompt": "Make it more colorful and engaging"
})
```

### For Developers/Custom Workflows

Use the low-level agents for custom control:

```python
# Step-by-step custom workflow
plan_response = await client.post("/api/v1/video/agents/plan", json={
    "topic": "Machine Learning",
    "description": "Educational video",
    "session_id": "custom_session"
})

# Get the plan result
plan_result = await client.get(f"/api/v1/video/agents/jobs/{plan_response['job_id']}")

# Generate implementation for specific scenes
implement_response = await client.post("/api/v1/video/agents/implement", json={
    "topic": "Machine Learning",
    "description": "Educational video",
    "plan_xml": plan_result["result"]["plan"],
    "session_id": "custom_session"
})
```

## Key Differences

| Aspect | High-Level Workflows | Low-Level Agents |
|--------|---------------------|------------------|
| **Complexity** | Simple, one-call solutions | Multi-step, manual orchestration |
| **Asset Management** | Automatic S3 upload/management | Manual file handling |
| **Database Sync** | Automatic Convex synchronization | No database integration |
| **Error Handling** | Comprehensive retry logic | Basic error reporting |
| **Progress Tracking** | Detailed progress with estimates | Simple job status |
| **Use Case** | Production applications | Development, testing, custom workflows |
| **Learning Curve** | Easy to use | Requires understanding of internals |

## Migration Guide

### From Legacy `/tasks/video/*` Endpoints

The old task endpoints in main.py have been replaced:

- `POST /tasks/video/generate` → `POST /api/v1/video-generation/generate`
- `POST /tasks/video/edit` → `POST /api/v1/video-generation/edit-scene`

### Benefits of New Structure

1. **Clear Separation**: High-level vs low-level operations
2. **Better Organization**: Related endpoints grouped together
3. **Improved Documentation**: Each group has specific documentation
4. **Enhanced Features**: High-level workflows include S3 and Convex integration
5. **Backward Compatibility**: Low-level agents preserve existing functionality

## Error Handling

### High-Level Workflows
- Comprehensive error recovery
- Automatic retries with exponential backoff
- Detailed error messages with context
- Graceful degradation

### Low-Level Agents
- Basic error reporting
- Manual retry handling
- Simple error messages
- Direct failure propagation

## Authentication & Authorization

Both API groups support the same authentication mechanisms:
- JWT tokens
- API keys
- Project-based access control

## Rate Limiting

- High-level workflows: Limited by resource-intensive operations
- Low-level agents: Higher rate limits for development use

## Monitoring & Logging

- High-level workflows: Comprehensive metrics and logging
- Low-level agents: Basic operation logging

## Future Considerations

### Planned Enhancements

1. **WebSocket Support**: Real-time progress updates for high-level workflows
2. **Batch Operations**: Multiple video generation in single request
3. **Template System**: Pre-defined video templates
4. **Advanced Analytics**: Usage metrics and performance insights

### Deprecation Policy

- Low-level agents will remain stable for development use
- High-level workflows will receive new features and improvements
- Legacy endpoints in main.py are deprecated and will be removed in v2.0