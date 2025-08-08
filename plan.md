# Detailed MVP Implementation Plan (Refined)

This plan tightens contracts, state, security, and delivery for a 1-month MVP. It locks key decisions, details schemas, APIs, jobs, and E2E tests, and maps to a week-by-week schedule.

## 0) Locked Decisions

- Job queue: Celery + Redis (retry with backoff, visibility timeout).
- Rendering isolation: Dockerized Manim workers with resource limits.
- Storage: S3 with versioned keys; CloudFront distribution for delivery.
- Auth: NextAuth (GitHub/Google) + Convex auth adapter for a stable userId.
- Webhooks: HMAC-signed with timestamp, both directions (Convex → FastAPI and FastAPI → Convex).
- Scene model: Frame-accurate scene boundaries and status state machine.

## 1) Data Model and State

Define a minimal, explicit state machine to avoid race conditions and drive UI.

- Session status: idle → queued → generating → rendering → ready | error
- Scene status: queued → generating_code → rendering → uploading → ready | error

Convex schema (skeleton):
```typescript
// filepath: d:\CODE\sf_convex\convex\schema.ts
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  users: defineTable({
    externalId: v.string(),           // from NextAuth
    name: v.optional(v.string()),
    email: v.optional(v.string()),
    createdAt: v.number(),
  }).index("by_externalId", ["externalId"]),

  projects: defineTable({
    ownerId: v.id("users"),
    title: v.string(),
    description: v.optional(v.string()),
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_owner", ["ownerId"]),

  videoSessions: defineTable({
    projectId: v.id("projects"),
    status: v.string(), // "idle" | "queued" | "generating" | "rendering" | "ready" | "error"
    targetFps: v.number(),         // 30
    width: v.number(),             // 1920
    height: v.number(),            // 1080
    codec: v.string(),             // "h264"
    audioHz: v.number(),           // 48000
    durationInFrames: v.optional(v.number()),
    errorMessage: v.optional(v.string()),
    jobId: v.optional(v.string()),
    version: v.number(),           // monotonically increasing
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_project", ["projectId"]),

  scenes: defineTable({
    projectId: v.id("projects"),
    sessionId: v.id("videoSessions"),
    index: v.number(),                  // 0-based ordering
    startFrame: v.number(),             // cumulative based on previous scenes
    endFrame: v.number(),               // inclusive end
    durationInFrames: v.number(),
    title: v.optional(v.string()),

    status: v.string(), // "queued" | "generating_code" | "rendering" | "uploading" | "ready" | "error"
    errorMessage: v.optional(v.string()),

    s3ChunkKey: v.optional(v.string()),     // versioned key: projects/{projectId}/sessions/{sessionId}/scenes/{index}/v{N}.mp4
    s3ChunkUrl: v.optional(v.string()),     // CloudFront URL
    s3SourceKey: v.optional(v.string()),    // Manim code .py
    checksum: v.optional(v.string()),       // SHA256 of chunk for cache bust
    jobId: v.optional(v.string()),
    version: v.number(),                    // scene-specific version
    updatedAt: v.number(),
  }).index("by_session", ["sessionId"]).index("by_project", ["projectId"]),
});
```

## 2) Contracts: Convex ↔ FastAPI

Use Convex Actions for outbound HTTP and a shared secret with HMAC.

Headers:
- X-Signature: hex(HMAC_SHA256(secret, timestamp + "." + body))
- X-Timestamp: epoch millis
- X-Correlation-Id: UUID v4

Endpoints (FastAPI):
- POST /jobs/generate
- POST /jobs/edit
- POST /callbacks/scene-updated (FastAPI → Convex action)
- POST /callbacks/session-updated (FastAPI → Convex action)

Convex Actions (server-side only):
- api.jobs.triggerGenerate(projectId, sessionId, sceneCount, prompt)
- api.jobs.triggerEdit(sceneId, prompt)
- api.callbacks.sceneUpdated(...)
- api.callbacks.sessionUpdated(...)

## 3) Backend: FastAPI + Celery + Redis + Docker

- Celery workers run Manim rendering inside Docker to sandbox execution.
- Retries: exponential backoff, max 3 attempts; mark scene/session error after max.
- Correlation IDs propagated to logs (uvicorn, Celery, app logs).

FastAPI skeleton:
```python
// filepath: d:\CODE\sf_convex\backend\main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import hmac, hashlib, time, os, uuid
from celery import Celery

app = FastAPI(title="AI Video Tutor API", version="1.0.0")

celery = Celery(__name__, broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
celery.conf.update(result_backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"))

SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret")

def verify(req_body: bytes, ts: str, sig: str):
  mac = hmac.new(SECRET.encode(), f"{ts}.{req_body.decode()}".encode(), hashlib.sha256).hexdigest()
  return hmac.compare_digest(mac, sig)

class GenerateReq(BaseModel):
  projectId: str
  sessionId: str
  sceneCount: int
  prompt: str

class EditReq(BaseModel):
  sceneId: str
  projectId: str
  prompt: str

@app.post("/jobs/generate")
async def generate(req: Request, body: GenerateReq):
  ts = req.headers.get("x-timestamp")
  sig = req.headers.get("x-signature")
  if not (ts and sig and verify(await req.body(), ts, sig)):
    raise HTTPException(status_code=401, detail="invalid signature")
  job_id = str(uuid.uuid4())
  celery.send_task("tasks.generate_video", args=[body.dict(), job_id])
  return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/edit")
async def edit(req: Request, body: EditReq):
  ts = req.headers.get("x-timestamp")
  sig = req.headers.get("x-signature")
  if not (ts and sig and verify(await req.body(), ts, sig)):
    raise HTTPException(status_code=401, detail="invalid signature")
  job_id = str(uuid.uuid4())
  celery.send_task("tasks.edit_scene", args=[body.dict(), job_id])
  return {"job_id": job_id, "status": "queued"}
```

Celery tasks outline:
```python
// filepath: d:\CODE\sf_convex\backend\tasks.py
from celery import Celery
import os, subprocess, json, tempfile, time
import boto3, hashlib, requests

celery = Celery(__name__, broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
celery.conf.update(result_backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"))

CF_BASE = os.getenv("CLOUDFRONT_BASE_URL")
S3_BUCKET_CHUNKS = os.getenv("S3_BUCKET_CHUNKS")
S3_BUCKET_SOURCES = os.getenv("S3_BUCKET_SOURCES")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
CONVEX_CALLBACK_URL = os.getenv("CONVEX_CALLBACK_URL")

def sign(body: dict):
  import hmac, hashlib, time
  ts = str(int(time.time() * 1000))
  payload = json.dumps(body, separators=(",", ":"))
  sig = hmac.new(WEBHOOK_SECRET.encode(), f"{ts}.{payload}".encode(), hashlib.sha256).hexdigest()
  return {"X-Timestamp": ts, "X-Signature": sig, "Content-Type": "application/json"}

@celery.task(name="tasks.generate_video", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def generate_video(self, req: dict, job_id: str):
  # 1) Split prompt -> scenes, generate code per scene (via CodeGenerator + MCP)
  # 2) For each scene: render in Dockerized Manim, upload to S3 (versioned key), compute checksum
  # 3) Call Convex callbacks to update scene and session status/URLs
  pass

@celery.task(name="tasks.edit_scene", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def edit_scene(self, req: dict, job_id: str):
  # 1) Fetch original source code from S3
  # 2) Use CodeGenerator with MCP to modify code based on prompt
  # 3) Re-render, upload new versioned chunk, callback Convex
  pass
```

Dockerized rendering (example command):
- docker run --rm -m 2g --cpus="2" -v <tmp_dir>:/work manimcommunity/manim:stable python render_scene.py

## 4) MCP Server

- FastAPI app serving curated Manim snippets/examples.
- Endpoints: GET /context/{topic}, GET /search?q=
- In-memory index with simple TF-IDF or keyword scoring for MVP.
- License check for included docs/snippets.

```python
// filepath: d:\CODE\sf_convex\mcp_server\main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import json, os

app = FastAPI(title="MCP Server", version="1.0.0")
DOCS_DIR = os.getenv("MCP_DOCS_DIR", "docs/manim")
CACHE: Dict[str, List[Dict]] = {}

@app.on_event("startup")
def load_docs():
  # Load JSON files into CACHE by topic
  pass

@app.get("/context/{topic}")
def context(topic: str) -> Dict:
  return {"topic": topic, "snippets": CACHE.get(topic, [])[:8]}

@app.get("/search")
def search(q: str) -> List[Dict]:
  # naive scoring over titles + body
  return []
```

## 5) Frontend: Next.js + Remotion + Convex

- Remotion Player to stitch remote MP4 chunks; use exact frame counts.
- Scene markers clickable; seek to startFrame.
- Optimistic UI updates for “edit requested”.

Remotion composition skeleton:
```typescript
// filepath: d:\CODE\sf_convex\remotion\VideoSequence.tsx
import { AbsoluteFill, Video, useCurrentFrame } from 'remotion';

type Scene = { url: string; startFrame: number; endFrame: number; };

export const VideoSequence: React.FC<{ scenes: Scene[] }> = ({ scenes }) => {
  const frame = useCurrentFrame();
  const current = scenes.find(s => frame >= s.startFrame && frame <= s.endFrame);
  if (!current) return null;
  const offset = frame - current.startFrame;
  return (
    <AbsoluteFill>
      <Video src={current.url} startFrom={offset} />
    </AbsoluteFill>
  );
};
```

Player wrapper:
```typescript
// filepath: d:\CODE\sf_convex\components\VideoPlayer.tsx
import { Player } from '@remotion/player';
import { VideoSequence } from '../remotion/VideoSequence';

export function VideoPlayer({ scenes, fps, width, height }:{
  scenes: { url: string; startFrame: number; endFrame: number; }[];
  fps: number; width: number; height: number;
}) {
  const durationInFrames = scenes.length ? scenes[scenes.length-1].endFrame + 1 : 1;
  return (
    <Player
      component={VideoSequence}
      inputProps={{ scenes }}
      durationInFrames={durationInFrames}
      compositionWidth={width}
      compositionHeight={height}
      fps={fps}
      controls
    />
  );
}
```

Convex client setup and hooks for realtime scenes/session remain as planned, but include status fields and frame metrics.

## 6) S3/CloudFront, CORS, and Headers

- Buckets:
  - ai-video-tutor-chunks (public via CloudFront)
  - ai-video-tutor-sources (private)
- Keys:
  - chunks: projects/{projectId}/sessions/{sessionId}/scenes/{index}/v{sceneVersion}.mp4
  - sources: projects/{projectId}/sessions/{sessionId}/scenes/{index}/v{sceneVersion}.py
- Headers:
  - Content-Type: video/mp4
  - Cache-Control: public, max-age=31536000, immutable (versioned URLs)
  - Accept-Ranges: bytes (ensure Range requests)
  - CORS: Allow GET, HEAD from frontend origin
- CloudFront:
  - Origin: S3 chunks
  - Compress disabled for MP4
  - Signed URLs not required for MVP

## 7) Security and Secrets

- HMAC signing on all webhooks with timestamp; reject stale (>5 min) timestamps.
- No S3 write creds in frontend; server-only uploads.
- Environment consolidation via .env files per service; do not commit.
- Basic RBAC: users can only access projects they own.

## 8) Observability

- Structured logs (JSON) including correlationId, projectId, sessionId, sceneId, jobId.
- Metrics (if time): render time per scene, queue latency, failure rate.
- Error reporting: capture and surface errorMessage in Convex, show in UI.

## 9) E2E Tests (MVP)

- Generate flow: given prompt with 3–5 scenes → ready within T minutes; all scenes ready.
- Playback: Remotion can play and seek to each scene boundary without stutter.
- Edit flow: request edit on scene 2 → scene status transitions correctly → new version URL propagates and plays.
- Failure: induce render error → surfaces error in UI and allows retry.

## 10) Week-by-Week Schedule (30 Days)

Week 1: Backend foundation
- D1: Convex project init; schema tables and indexes; auth scaffold (NextAuth + Convex).
- D2: CRUD for projects/sessions/scenes; status transitions; actions for callbacks.
- D3: FastAPI skeleton; HMAC verify; /jobs endpoints; Celery wiring; Redis.
- D4: Dockerized render container; S3 clients; versioned key helpers; callback actions to Convex.
- D5: MCP server skeleton; doc ingest; basic search; CodeGenerator stub.
- D6–7: Integrate CodeGenerator with MCP; stub Manim render; end-to-end dry run with fake chunks.

Week 2: Frontend core
- D8: Next.js + Tailwind + Convex client; Remotion config; base composition.
- D9: Player with scene markers; frame-accurate seek; loading and error states.
- D10: Realtime hooks for session/scenes; map status to UI; basic project/session pages.
- D11: Trigger generate from UI → Convex action → FastAPI; show queued/progress.
- D12–14: Polish playback; CORS/CloudFront headers; preflight checks; UX tweaks.

Week 3: Editing & feedback loop
- D15: SceneEditor panel; prompt input; Submit button (fixes prior typo).
- D16–17: Convex function triggerEdit; FastAPI edit endpoint; re-render pipeline.
- D18–19: S3 versioning end-to-end; optimistic UI; revert-to-previous (if time).
- D20–21: Error handling, retries, and manual retry from UI.

Week 4: Hardening, tests, and demo
- D22–24: E2E tests (generate, playback, edit); fix integration bugs.
- D25–26: Perf guardrails; concurrency; S3 lifecycle policy for old versions.
- D27–28: UX polish; loading indicators; clear error messages; correlationId in logs.
- D29–30: Docs and demo script; sample prompt and walkthrough.

## 11) Environment Variables (MVP)

- Frontend: NEXT_PUBLIC_CONVEX_URL, NEXTAUTH_URL, NEXTAUTH_SECRET, NEXTAUTH_PROVIDERS (GitHub/Google)
- Convex: CONVEX_SITE_URL, WEBHOOK_SECRET, FASTAPI_URL
- FastAPI: WEBHOOK_SECRET, REDIS_URL, S3_BUCKET_CHUNKS, S3_BUCKET_SOURCES, AWS creds, CLOUDFRONT_BASE_URL, CONVEX_CALLBACK_URL, OPENAI/Anthropic keys, MCP_SERVER_URL
- MCP: MCP_DOCS_DIR

## 12) Quick Open Issues

- Choose minimal Manim subset for MVP; pin Manim version.
- Define timeout per scene render (e.g., 3–5 minutes).
- Decide on storing source code diffs vs full copies (MVP: full copies).

This plan keeps scope tight while solidifying contracts, state, and delivery to de-risk the demo.