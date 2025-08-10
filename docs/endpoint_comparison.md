# API Endpoint Comparison

## Quick Reference: Which API Should I Use?

### 🎯 **For Most Users: High-Level Workflows**

**Use `/api/v1/video-generation/*` when you want:**
- ✅ Complete video generation in one call
- ✅ Automatic S3 upload and asset management  
- ✅ Database synchronization with Convex
- ✅ Advanced progress tracking with estimates
- ✅ Built-in error recovery and retries
- ✅ Scene editing with version management

**Example**: "I want to generate a 3-scene educational video about Python programming"

```bash
curl -X POST "/api/v1/video-generation/generate" \
  -d '{"project_id": "edu_001", "content_prompt": "Python basics", "scene_count": 3}'
```

### 🔧 **For Developers: Low-Level Agents**

**Use `/api/v1/video/agents/*` when you want:**
- ✅ Fine-grained control over each step
- ✅ Custom workflows and orchestration
- ✅ Testing individual components
- ✅ Debugging specific operations
- ✅ Building custom integrations

**Example**: "I want to generate scene plans, then modify them before rendering"

```bash
# Step 1: Generate plan
curl -X POST "/api/v1/video/agents/plan" \
  -d '{"topic": "Python", "description": "Educational video"}'

# Step 2: Generate implementation (after reviewing plan)
curl -X POST "/api/v1/video/agents/implement" \
  -d '{"topic": "Python", "plan_xml": "...", "session_id": "..."}'

# Step 3: Generate code for specific scene
curl -X POST "/api/v1/video/agents/code" \
  -d '{"scene_outline": "...", "implementation": "...", "scene_number": 1}'
```

## Detailed Comparison

| Feature | High-Level Workflows<br>`/video-generation/*` | Low-Level Agents<br>`/video/agents/*` |
|---------|-----------------------------------------------|--------------------------------------|
| **Complexity** | 🟢 Simple - One call does everything | 🟡 Complex - Multiple coordinated calls |
| **S3 Integration** | 🟢 Automatic upload & management | 🔴 Manual file handling required |
| **Database Sync** | 🟢 Automatic Convex synchronization | 🔴 No database integration |
| **Progress Tracking** | 🟢 Detailed with time estimates | 🟡 Basic job status only |
| **Error Handling** | 🟢 Comprehensive retry logic | 🟡 Basic error reporting |
| **Asset Management** | 🟢 Full lifecycle management | 🔴 No asset management |
| **Version Control** | 🟢 Automatic scene versioning | 🔴 No version management |
| **Learning Curve** | 🟢 Easy to use | 🟡 Requires system knowledge |
| **Use Case** | 🟢 Production applications | 🟢 Development & testing |
| **Performance** | 🟡 Higher overhead | 🟢 Minimal overhead |
| **Flexibility** | 🟡 Predefined workflows | 🟢 Complete customization |

## Endpoint Mapping

### High-Level Workflows → Low-Level Agents

| High-Level Endpoint | Equivalent Low-Level Sequence |
|-------------------|------------------------------|
| `POST /video-generation/generate` | 1. `POST /video/agents/plan`<br>2. `POST /video/agents/implement`<br>3. `POST /video/agents/code` (per scene)<br>4. `POST /video/agents/render/scene` (per scene)<br>5. `POST /video/agents/render/combine`<br>6. Manual S3 upload<br>7. Manual Convex sync |
| `POST /video-generation/edit-scene` | 1. Retrieve source from S3<br>2. `POST /video/agents/code` (with modifications)<br>3. `POST /video/agents/render/scene`<br>4. Manual S3 upload<br>5. Manual Convex sync |

### Status & Results

| High-Level | Low-Level | Notes |
|-----------|-----------|-------|
| `GET /video-generation/tasks/{id}/status` | `GET /video/agents/jobs/{id}` | High-level has richer progress info |
| `GET /video-generation/tasks/{id}/result` | `GET /video/agents/jobs/{id}` | High-level includes S3 URLs & metadata |

## Migration Examples

### From Legacy Main.py Endpoints

**Old (Deprecated)**:
```bash
POST /tasks/video/generate
POST /tasks/video/edit
GET /tasks/{id}/status
```

**New (Recommended)**:
```bash
POST /api/v1/video-generation/generate
POST /api/v1/video-generation/edit-scene  
GET /api/v1/video-generation/tasks/{id}/status
```

### From Low-Level to High-Level

**Before (Multiple calls)**:
```python
# Generate plan
plan_response = await client.post("/api/v1/video/agents/plan", json=plan_data)
plan_result = await wait_for_completion(plan_response["job_id"])

# Generate implementation  
impl_response = await client.post("/api/v1/video/agents/implement", json=impl_data)
impl_result = await wait_for_completion(impl_response["job_id"])

# Generate and render each scene
for scene in scenes:
    code_response = await client.post("/api/v1/video/agents/code", json=scene_data)
    code_result = await wait_for_completion(code_response["job_id"])
    
    render_response = await client.post("/api/v1/video/agents/render/scene", json=render_data)
    render_result = await wait_for_completion(render_response["job_id"])
    
    # Manual S3 upload
    s3_url = upload_to_s3(render_result["output_file"])
    
    # Manual Convex sync
    await sync_to_convex(scene_id, s3_url)

# Combine videos
combine_response = await client.post("/api/v1/video/agents/render/combine", json=combine_data)
final_result = await wait_for_completion(combine_response["job_id"])
```

**After (Single call)**:
```python
# One call does everything
response = await client.post("/api/v1/video-generation/generate", json={
    "project_id": "my_project",
    "content_prompt": "Educational video about Python",
    "scene_count": 3
})

# Simple progress monitoring
result = await wait_for_completion(response["task_id"])
# Result includes all S3 URLs, Convex sync is automatic
```

## When to Use Each

### ✅ Use High-Level Workflows When:
- Building user-facing applications
- You want the complete video generation pipeline
- You need S3 asset management
- You want database synchronization
- You prefer simple, reliable operations
- You're building MVPs or prototypes

### ✅ Use Low-Level Agents When:
- Building custom video generation tools
- You need fine-grained control over each step
- You're implementing custom asset management
- You're debugging or testing specific components
- You're integrating with existing systems
- You need maximum performance and minimal overhead

## Best Practices

### For High-Level Workflows:
1. Always monitor task progress using the status endpoint
2. Handle task failures gracefully with retry logic
3. Use the health check endpoint to verify service availability
4. Configure appropriate timeouts for long-running operations

### For Low-Level Agents:
1. Implement your own orchestration logic
2. Handle file management and cleanup
3. Implement custom error recovery
4. Monitor individual job statuses
5. Coordinate between multiple agent calls

## Task Organization

The backend tasks are organized to match the API structure:

| API Group | Task File | Task Examples |
|-----------|-----------|---------------|
| **High-Level Workflows**<br>`/video-generation/*` | `video_generation_tasks.py` | `generate_video_orchestration_task`<br>`edit_scene_orchestration_task` |
| **Low-Level Agents**<br>`/video/agents/*` | `video_tasks.py` | `plan_scenes_task`<br>`codegen_task`<br>`render_scene_task` |

For detailed task organization, see [Task Organization Documentation](task_organization.md).

## Support & Documentation

- **High-Level Workflows**: [Video Generation API Documentation](video_generation_api.md)
- **Low-Level Agents**: [Agent Operations Documentation](agent_operations_api.md)
- **Architecture Overview**: [API Structure Documentation](api_structure.md)
- **Task Organization**: [Task Organization Documentation](task_organization.md)
- **Examples**: [Usage Examples](../examples/)