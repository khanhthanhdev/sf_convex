Below is a step-by-step setup plan to get your backend running with FastAPI, Convex.dev, Celery and Redis—and to verify everything by running a tiny demo “hello-world” task.

---

## 🛠️ 1. Prerequisites

* **Python ≥ 3.9**, Node.js ≥ 16
* **Redis** installed locally or via Docker
* **Convex CLI** (`npm install -g convex`)
* A Convex account/project created at [https://app.convex.dev](https://app.convex.dev)

---

## 📁 2. Project Layout

```
backend/
├── convex/              # Convex schema & functions
│   ├── schema.ts
│   └── functions/
├── fastapi/             # FastAPI app
│   ├── main.py
│   └── celery_tasks.py
├── .env
├── requirements.txt
└── package.json         # for convex CLI
```

---

## 🔧 3. Environment & Dependencies

1. **Create a virtualenv**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Python deps**

   ```bash
   pip install fastapi uvicorn celery redis boto3 convex-py
   ```

3. **Init Convex**

   ```bash
   cd convex
   npx convex init      # follow prompts to link your project
   ```

4. **Create `.env`** at root:

   ```ini
   REDIS_URL=redis://localhost:6379/0
   CONVEX_URL=https://your-project.convex.cloud
   CONVEX_ADMIN_KEY=your-secret-key
   ```

---

## 🚀 4. Convex.dev Quickstart

1. **Define a minimal schema** in `convex/schema.ts`:

   ```ts
   import { defineSchema, defineTable } from "convex/server";
   import { v } from "convex/values";

   export default defineSchema({
     messages: defineTable({
       text: v.string(),
       createdAt: v.number(),
     }),
   });
   ```

2. **Deploy your schema**

   ```bash
   cd convex
   npx convex deploy
   ```

3. **Write a simple mutation** in `convex/functions/addMessage.ts`:

   ```ts
   import { mutation } from "./_generated/server";
   export const addMessage = mutation({
     args: { text: v.string() },
     handler: async (ctx, { text }) => {
       return await ctx.db.insert("messages", {
         text,
         createdAt: Date.now(),
       });
     },
   });
   ```

---

## 🔨 5. FastAPI + Celery + Redis Setup

### 5.1. `requirements.txt`

```
fastapi
uvicorn
celery
redis
boto3
convex-py
python-dotenv
```

### 5.2. FastAPI App (`fastapi/main.py`)

```python
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery import Celery
from convex import ConvexClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
celery = Celery(
    "worker",
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_URL")
)
convex = ConvexClient(os.getenv("CONVEX_URL"), os.getenv("CONVEX_ADMIN_KEY"))

class HelloRequest(BaseModel):
    name: str

@app.post("/say-hello")
async def say_hello(req: HelloRequest):
    # enqueue a Celery task
    task = celery.send_task("celery_tasks.hello_task", args=[req.name])
    return {"task_id": task.id, "status": "queued"}

@app.get("/messages")
async def get_messages():
    # fetch from Convex
    result = await convex.run_query("functions.listMessages", {})
    return result
```

### 5.3. Celery Tasks (`fastapi/celery_tasks.py`)

```python
from celery import Celery
import time

celery = Celery(
    "worker",
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_URL")
)

@celery.task(name="celery_tasks.hello_task")
def hello_task(name: str):
    # simulate work
    time.sleep(2)
    return f"👋 Hello, {name}!"
```

---

## ✅ 6. Run & Verify

1. **Start Redis**

   ```bash
   redis-server
   # or: docker run -p 6379:6379 redis
   ```

2. **Start Celery worker**

   ```bash
   cd fastapi
   celery -A celery_tasks worker --loglevel=info
   ```

3. **Start FastAPI server**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Smoke test**

   * **POST** `http://localhost:8000/say-hello` with JSON

     ```json
     { "name": "Alice" }
     ```

     → returns `{"task_id":"…","status":"queued"}`
   * Check Celery logs to see `hello_task` run and complete.

5. **Convex check**

   * Write and deploy `listMessages` query in Convex:

     ```ts
     export const listMessages = query({
       handler: async (ctx) => ctx.db.query("messages").collect()
     });
     ```
   * **GET** `http://localhost:8000/messages` → should return an empty array.

---

### Next Steps

* Wire Celery tasks to call your core CLI generator and push results into Convex (via `convex.run_mutation("functions.addMessage", { text })`).
* Expand schema for `projects`, `videoSessions`, `scenes`.
* Add authentication middleware and secure your endpoints.

With this in place, you’ll have a robust foundation for orchestrating your AI-powered video generation pipeline!
