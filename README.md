# AI Video Tutor Backend

This is the backend infrastructure for the AI Video Tutor platform, implementing a microservices architecture with FastAPI, Convex.dev, Redis, and Celery.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Convex.dev    │    │   FastAPI       │
│   (Next.js)     │◄──►│   (Database)    │◄──►│   (API Gateway) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Celery        │
                                              │   (Workers)     │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Redis         │
                                              │   (Message      │
                                              │    Broker)      │
                                              └─────────────────┘
```

## Services

### 1. Convex.dev (Database & Real-time Sync)
- **Location**: `convex/`
- **Purpose**: Database schema, CRUD operations, real-time subscriptions
- **Schema**: Users, Projects, Video Sessions, Scenes
- **Functions**: Queries, mutations, and actions for data management

### 2. FastAPI (API Gateway)
- **Location**: `fastapi/`
- **Purpose**: REST API endpoints, webhook handling, business logic
- **Endpoints**: 
  - `POST /jobs/generate` - Trigger video generation
  - `POST /jobs/edit` - Trigger scene editing
  - `GET /health` - Health check
  - `POST /test/hello` - Hello world demo

### 3. Celery (Background Processing)
- **Location**: `fastapi/tasks.py`
- **Purpose**: Asynchronous task processing for video generation
- **Tasks**: Video generation, scene editing, rendering

### 4. Redis (Message Broker)
- **Purpose**: Celery message broker and result backend
- **Port**: 6379

## Quick Start

### Prerequisites
- Python 3.11+
- Redis server
- Convex.dev account and deployment

### 1. Environment Setup

Copy the example environment file and configure your settings:

```bash
cd backend/fastapi
cp env.example .env
```

Edit `.env` with your actual values:

```bash
# Convex Configuration
CONVEX_URL=https://your-deployment.convex.cloud
CONVEX_ADMIN_KEY=your-admin-key-here

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Webhook Security
WEBHOOK_SECRET=your-secret-here
```

### 2. Start Redis

**Option A: Local Redis**
```bash
redis-server
```

**Option B: Docker Redis**
```bash
docker run -d -p 6379:6379 redis:alpine
```

### 3. Deploy Convex Schema

```bash
cd backend/convex
npx convex dev
```

This will deploy your schema and functions to Convex.

### 4. Start Backend Services

**Option A: Using the startup script**
```bash
cd backend
./start-backend.sh
```

**Option B: Using Docker Compose**
```bash
cd backend
docker-compose up
```

**Option C: Manual startup**
```bash
cd backend/fastapi

# Install dependencies
pip install -r requirements.txt

# Start Celery worker (in background)
celery -A celery_app worker --loglevel=info &

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Hello World Demo
```bash
# Queue a hello task
curl -X POST "http://localhost:8000/test/hello?name=World"

# Check task status (replace TASK_ID with actual ID)
curl http://localhost:8000/test/task/TASK_ID
```

### Video Generation
```bash
curl -X POST http://localhost:8000/jobs/generate \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Signature: sha256=..." \
  -d '{
    "projectId": "project_id",
    "sessionId": "session_id", 
    "sceneCount": 3,
    "prompt": "Create a video about calculus",
    "timestamp": 1234567890
  }'
```

## Development

### Project Structure
```
backend/
├── convex/                 # Convex.dev functions and schema
│   ├── schema.ts          # Database schema
│   ├── users.ts           # User management
│   ├── projects.ts        # Project management
│   ├── videoSessions.ts   # Video session management
│   ├── scenes.ts          # Scene management
│   └── jobs.ts            # Job triggering actions
├── fastapi/               # FastAPI application
│   ├── main.py            # FastAPI app and endpoints
│   ├── celery_app.py      # Celery configuration
│   ├── tasks.py           # Celery tasks
│   ├── convex_client.py   # Convex Python SDK client
│   └── requirements.txt   # Python dependencies
├── docker-compose.yml     # Docker services
├── start-backend.sh       # Startup script
└── README.md             # This file
```

### Adding New Tasks

1. Add task function to `fastapi/tasks.py`:
```python
@celery_app.task(bind=True)
def my_new_task(self, arg1, arg2):
    # Task implementation
    pass
```

2. Add endpoint to `fastapi/main.py`:
```python
@app.post("/jobs/my-task")
async def trigger_my_task(request: Request, body: MyTaskRequest):
    # Queue the task
    task = my_new_task.delay(body.arg1, body.arg2)
    return {"jobId": task.id}
```

3. Add Convex action to `convex/jobs.ts`:
```typescript
export const triggerMyTask = action({
  args: { arg1: v.string(), arg2: v.string() },
  handler: async (ctx, args) => {
    // Call FastAPI endpoint
  }
});
```

### Monitoring

- **Celery Flower** (optional): `celery -A celery_app flower`
- **Redis CLI**: `redis-cli monitor`
- **FastAPI Docs**: http://localhost:8000/docs

## Testing

### Unit Tests
```bash
cd backend/fastapi
python -m pytest tests/
```

### Integration Tests
```bash
# Test Convex connection
curl http://localhost:8000/health

# Test Celery integration
curl -X POST "http://localhost:8000/test/hello?name=Test"
```

## Deployment

### Production Considerations

1. **Environment Variables**: Set all required environment variables
2. **Redis**: Use managed Redis service (AWS ElastiCache, etc.)
3. **Convex**: Use production deployment
4. **Security**: Set proper `WEBHOOK_SECRET`
5. **Monitoring**: Add application monitoring and logging
6. **Scaling**: Use multiple Celery workers for high load

### Docker Deployment
```bash
cd backend
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running: `redis-cli ping`
   - Check `REDIS_URL` environment variable

2. **Convex Connection Error**
   - Verify `CONVEX_URL` and `CONVEX_ADMIN_KEY`
   - Check Convex deployment status

3. **Celery Worker Not Starting**
   - Check Redis connection
   - Verify task imports in `celery_app.py`

4. **Webhook Signature Errors**
   - Set `WEBHOOK_SECRET` environment variable
   - Check signature generation in Convex actions

### Logs

- **FastAPI**: Check console output or logs
- **Celery**: `celery -A celery_app worker --loglevel=debug`
- **Redis**: `redis-cli monitor`

## Next Steps

This backend infrastructure provides the foundation for:

1. **Video Generation Pipeline** (Task 2)
2. **Scene-Based Editing** (Task 3) 
3. **Real-time Video Preview** (Task 4)
4. **Authentication Integration** (Task 6)

Each subsequent task will build upon this infrastructure to add specific functionality.
