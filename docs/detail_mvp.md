# Detailed Implementation Plan: AI Video Tutor MVP (1 Month)

## Week 1: Backend Foundation & Core AI Integration

### Day 1-2: Convex.dev Backend Setup

#### Task 1.1: Initialize Convex Project
```bash
# Install Convex CLI
npm install -g convex

# Create new project
npx create-convex@latest ai-video-tutor-backend
cd ai-video-tutor-backend

# Initialize Convex
npx convex dev
```

#### Task 1.2: Define MVP Schema
Create `convex/schema.ts`:
```typescript
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  projects: defineTable({
    name: v.string(),
    description: v.optional(v.string()),
    userId: v.id("users"),
    status: v.union(v.literal("creating"), v.literal("ready"), v.literal("error")),
    createdAt: v.number(),
  }),
  
  videoSessions: defineTable({
    projectId: v.id("projects"),
    totalDuration: v.optional(v.number()),
    sceneCount: v.number(),
    status: v.union(v.literal("generating"), v.literal("ready"), v.literal("error")),
    s3BaseUrl: v.optional(v.string()),
    createdAt: v.number(),
  }),
  
  sceneMetadata: defineTable({
    videoSessionId: v.id("videoSessions"),
    sceneIndex: v.number(),
    sceneId: v.string(), // unique identifier for scene
    title: v.string(),
    duration: v.number(),
    s3ChunkUrl: v.optional(v.string()),
    sourceCodeS3Url: v.optional(v.string()),
    status: v.union(v.literal("generating"), v.literal("ready"), v.literal("error")),
    lastModified: v.number(),
  }),
  
  users: defineTable({
    email: v.string(),
    name: v.string(),
    createdAt: v.number(),
  }),
});
```

#### Task 1.3: Implement Basic CRUD Operations
Create `convex/projects.ts`:
```typescript
import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const create = mutation({
  args: {
    name: v.string(),
    description: v.optional(v.string()),
    userId: v.id("users"),
  },
  handler: async (ctx, args) => {
    const projectId = await ctx.db.insert("projects", {
      ...args,
      status: "creating",
      createdAt: Date.now(),
    });
    return projectId;
  },
});

export const list = query({
  args: { userId: v.id("users") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("projects")
      .filter((q) => q.eq(q.field("userId"), args.userId))
      .order("desc")
      .collect();
  },
});

export const get = query({
  args: { projectId: v.id("projects") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.projectId);
  },
});
```

### Day 3-4: FastAPI Multi-Agent System Setup

#### Task 1.4: Create FastAPI Project Structure
```bash
mkdir ai-video-tutor-api
cd ai-video-tutor-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn celery redis boto3 openai anthropic
```

#### Task 1.5: FastAPI Core Setup
Create `main.py`:
```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List, Optional
import uuid

app = FastAPI(title="AI Video Tutor API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoGenerationRequest(BaseModel):
    project_id: str
    content_prompt: str
    scene_count: int = 5

class SceneEditRequest(BaseModel):
    scene_id: str
    edit_prompt: str
    project_id: str

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Trigger video generation process"""
    job_id = str(uuid.uuid4())
    
    # Add to background task queue
    background_tasks.add_task(process_video_generation, request, job_id)
    
    return {"job_id": job_id, "status": "started"}

@app.post("/edit-scene")
async def edit_scene(request: SceneEditRequest, background_tasks: BackgroundTasks):
    """Trigger scene editing process"""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(process_scene_edit, request, job_id)
    
    return {"job_id": job_id, "status": "started"}

async def process_video_generation(request: VideoGenerationRequest, job_id: str):
    """Background task for video generation"""
    # This will integrate with your existing core function
    pass

async def process_scene_edit(request: SceneEditRequest, job_id: str):
    """Background task for scene editing"""
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Task 1.6: Integrate Existing Core Function
Create `agents/video_generator.py`:
```python
import subprocess
import os
import boto3
from typing import List, Dict
import tempfile
import json

class VideoGeneratorAgent:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'ai-video-tutor-chunks')
    
    def generate_video_chunks(self, content_prompt: str, scene_count: int, project_id: str) -> List[Dict]:
        """
        Integrate your existing CLI command here
        Break the output into chunks and upload to S3
        """
        scenes = []
        
        # Call your existing core function
        # This is where you'll integrate your CLI command
        result = self._call_existing_core_function(content_prompt, scene_count)
        
        # Process the result and create chunks
        for i, scene_data in enumerate(result['scenes']):
            scene_id = f"{project_id}_scene_{i}"
            
            # Upload chunk to S3
            s3_url = self._upload_chunk_to_s3(scene_data['video_path'], scene_id)
            code_s3_url = self._upload_code_to_s3(scene_data['code'], scene_id)
            
            scenes.append({
                'scene_id': scene_id,
                'scene_index': i,
                'title': scene_data.get('title', f'Scene {i+1}'),
                'duration': scene_data.get('duration', 30),
                's3_chunk_url': s3_url,
                'source_code_s3_url': code_s3_url,
                'status': 'ready'
            })
        
        return scenes
    
    def _call_existing_core_function(self, content_prompt: str, scene_count: int) -> Dict:
        """
        Replace this with your actual core function call
        """
        # Placeholder - integrate your existing CLI command here
        # Example: subprocess.run(['your-cli-command', content_prompt])
        
        # Mock response for now
        return {
            'scenes': [
                {
                    'title': f'Introduction to {content_prompt}',
                    'video_path': '/tmp/scene_0.mp4',
                    'code': 'from manim import *\n\nclass Scene0(Scene):\n    def construct(self):\n        text = Text("Introduction")\n        self.play(Write(text))',
                    'duration': 30
                }
                # Add more scenes based on scene_count
            ]
        }
    
    def _upload_chunk_to_s3(self, video_path: str, scene_id: str) -> str:
        """Upload video chunk to S3"""
        key = f"video-chunks/{scene_id}.mp4"
        
        try:
            self.s3_client.upload_file(video_path, self.bucket_name, key)
            return f"https://{self.bucket_name}.s3.amazonaws.com/{key}"
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return None
    
    def _upload_code_to_s3(self, code: str, scene_id: str) -> str:
        """Upload source code to S3"""
        key = f"source-code/{scene_id}.py"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            self.s3_client.upload_file(temp_path, self.bucket_name, key)
            os.unlink(temp_path)
            
            return f"https://{self.bucket_name}.s3.amazonaws.com/{key}"
        except Exception as e:
            print(f"Error uploading code to S3: {e}")
            return None
```

### Day 5-7: MCP Server Initial Integration

#### Task 1.7: Use context7 MCP server



## Week 2: Frontend Core & Video Preview

### Day 8-9: Remotion.dev + Next.js Project Setup

#### Task 2.1: Initialize Next.js with Remotion
```bash
# Create Next.js project
npx create-next-app@latest ai-video-tutor-frontend --typescript --tailwind --eslint
cd ai-video-tutor-frontend

# Install Remotion
npm install remotion @remotion/cli @remotion/player

# Install Convex client
npm install convex

# Install additional dependencies
npm install @radix-ui/react-slider @radix-ui/react-button lucide-react
```

#### Task 2.2: Setup Remotion Configuration
Create `remotion.config.ts`:
```typescript
import { Config } from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
Config.setConcurrency(2);
Config.setPixelFormat('yuv420p');
Config.setCodec('h264');
```

#### Task 2.3: Create Basic Remotion Composition
Create `remotion/VideoComposition.tsx`:
```typescript
import { Composition } from 'remotion';
import { VideoSequence } from './VideoSequence';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="VideoSequence"
        component={VideoSequence}
        durationInFrames={1800} // 60 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          videoChunks: [],
          sceneMetadata: [],
        }}
      />
    </>
  );
};
```

Create `remotion/VideoSequence.tsx`:
```typescript
import { Video, useCurrentFrame, useVideoConfig } from 'remotion';
import { useMemo } from 'react';

interface SceneMetadata {
  sceneId: string;
  sceneIndex: number;
  title: string;
  duration: number;
  s3ChunkUrl: string;
  startFrame: number;
  endFrame: number;
}

interface VideoSequenceProps {
  videoChunks: string[];
  sceneMetadata: SceneMetadata[];
}

export const VideoSequence: React.FC<VideoSequenceProps> = ({
  videoChunks,
  sceneMetadata,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Calculate which scene should be playing at current frame
  const currentScene = useMemo(() => {
    return sceneMetadata.find(
      (scene) => frame >= scene.startFrame && frame < scene.endFrame
    );
  }, [frame, sceneMetadata]);

  if (!currentScene) {
    return null;
  }

  // Calculate the local frame within the current scene
  const sceneFrame = frame - currentScene.startFrame;

  return (
    <div style={{ width: '100%', height: '100%', backgroundColor: 'black' }}>
      {currentScene.s3ChunkUrl && (
        <Video
          src={currentScene.s3ChunkUrl}
          startFrom={0}
          endAt={currentScene.duration * fps}
          style={{ width: '100%', height: '100%' }}
        />
      )}
      
      {/* Scene title overlay */}
      <div
        style={{
          position: 'absolute',
          bottom: 50,
          left: 50,
          color: 'white',
          fontSize: 24,
          fontWeight: 'bold',
          backgroundColor: 'rgba(0,0,0,0.7)',
          padding: '10px 20px',
          borderRadius: 8,
        }}
      >
        {currentScene.title}
      </div>
    </div>
  );
};
```

### Day 10-11: Convex.dev Frontend Integration

#### Task 2.4: Setup Convex Client
Create `convex/_generated/api.d.ts` (this will be auto-generated):
```typescript
// This file is auto-generated by Convex
export interface Api {
  projects: {
    create: any;
    list: any;
    get: any;
  };
  videoSessions: {
    create: any;
    get: any;
    list: any;
  };
  sceneMetadata: {
    list: any;
    update: any;
  };
}
```

Create `lib/convex.ts`:
```typescript
import { ConvexProvider, ConvexReactClient } from 'convex/react';

const convex = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL!);

export default convex;
```

#### Task 2.5: Create Convex Provider
Update `pages/_app.tsx`:
```typescript
import type { AppProps } from 'next/app';
import { ConvexProvider, ConvexReactClient } from 'convex/react';
import '../styles/globals.css';

const convex = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL!);

export default function App({ Component, pageProps }: AppProps) {
  return (
    <ConvexProvider client={convex}>
      <Component {...pageProps} />
    </ConvexProvider>
  );
}
```

#### Task 2.6: Create Video Session Hook
Create `hooks/useVideoSession.ts`:
```typescript
import { useQuery, useMutation } from 'convex/react';
import { api } from '../convex/_generated/api';
import { Id } from '../convex/_generated/dataModel';

export const useVideoSession = (projectId: Id<'projects'>) => {
  const videoSessions = useQuery(api.videoSessions.list, { projectId });
  const sceneMetadata = useQuery(
    api.sceneMetadata.list,
    videoSessions?.[0] ? { videoSessionId: videoSessions[0]._id } : 'skip'
  );

  const createVideoSession = useMutation(api.videoSessions.create);
  const updateSceneMetadata = useMutation(api.sceneMetadata.update);

  return {
    videoSession: videoSessions?.[0],
    sceneMetadata: sceneMetadata || [],
    createVideoSession,
    updateSceneMetadata,
  };
};
```

### Day 12-14: Scene-based Video Player

#### Task 2.7: Create Video Player Component
Create `components/VideoPlayer.tsx`:
```typescript
import { Player } from '@remotion/player';
import { VideoSequence } from '../remotion/VideoSequence';
import { useMemo, useState } from 'react';
import { Button } from './ui/Button';
import { Slider } from './ui/Slider';

interface SceneMetadata {
  sceneId: string;
  sceneIndex: number;
  title: string;
  duration: number;
  s3ChunkUrl: string;
}

interface VideoPlayerProps {
  sceneMetadata: SceneMetadata[];
  onSceneSelect: (sceneId: string) => void;
  selectedSceneId?: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  sceneMetadata,
  onSceneSelect,
  selectedSceneId,
}) => {
  const [currentFrame, setCurrentFrame] = useState(0);

  // Calculate total duration and scene boundaries
  const { totalFrames, sceneBoundaries } = useMemo(() => {
    let totalFrames = 0;
    const boundaries: Array<{
      sceneId: string;
      startFrame: number;
      endFrame: number;
      title: string;
    }> = [];

    sceneMetadata.forEach((scene) => {
      const startFrame = totalFrames;
      const durationFrames = scene.duration * 30; // 30 fps
      const endFrame = startFrame + durationFrames;

      boundaries.push({
        sceneId: scene.sceneId,
        startFrame,
        endFrame,
        title: scene.title,
      });

      totalFrames = endFrame;
    });

    return { totalFrames, sceneBoundaries: boundaries };
  }, [sceneMetadata]);

  // Prepare scene metadata with frame boundaries
  const enhancedSceneMetadata = useMemo(() => {
    return sceneMetadata.map((scene, index) => {
      const boundary = sceneBoundaries[index];
      return {
        ...scene,
        startFrame: boundary.startFrame,
        endFrame: boundary.endFrame,
      };
    });
  }, [sceneMetadata, sceneBoundaries]);

  const handleSceneMarkerClick = (sceneId: string, startFrame: number) => {
    setCurrentFrame(startFrame);
    onSceneSelect(sceneId);
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Video Player */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <Player
          component={VideoSequence}
          durationInFrames={totalFrames}
          compositionWidth={1920}
          compositionHeight={1080}
          fps={30}
          style={{
            width: '100%',
            height: 'auto',
          }}
          inputProps={{
            videoChunks: sceneMetadata.map(s => s.s3ChunkUrl),
            sceneMetadata: enhancedSceneMetadata,
          }}
          controls
          loop
        />
      </div>

      {/* Scene Timeline */}
      <div className="mt-4">
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-sm font-medium">Scenes:</span>
        </div>
        
        {/* Scene Markers */}
        <div className="relative h-12 bg-gray-200 rounded">
          {sceneBoundaries.map((boundary) => {
            const leftPercent = (boundary.startFrame / totalFrames) * 100;
            const widthPercent = totalFrames
              ? ((boundary.endFrame - boundary.startFrame) / totalFrames) * 100
              : 0;
            
            return (
              <div
                key={boundary.sceneId}
                className={`absolute h-full cursor-pointer transition-colors ${
                  selectedSceneId === boundary.sceneId
                    ? 'bg-blue-500'
                    : 'bg-gray-400 hover:bg-gray-500'
                }`}
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                }}
                onClick={() => handleSceneMarkerClick(boundary.sceneId, boundary.startFrame)}
                title={boundary.title}
              >
                <div className="p-1 text-xs text-white truncate">
                  {boundary.title}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Scene List */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
        {sceneMetadata.map((scene) => (
          <Button
            key={scene.sceneId}
            variant={selectedSceneId === scene.sceneId ? 'default' : 'outline'}
            className="text-left justify-start"
            onClick={() => onSceneSelect(scene.sceneId)}
          >
            <div>
              <div className="font-medium">{scene.title}</div>
              <div className="text-sm text-gray-500">{scene.duration}s</div>
            </div>
          </Button>
        ))}
      </div>
    </div>
  );
};
```

#### Task 2.8: Create Main Video Page
Create `pages/video/[projectId].tsx`:
```typescript
import { useRouter } from 'next/router';
import { useState } from 'react';
import { VideoPlayer } from '../../components/VideoPlayer';
import { SceneEditor } from '../../components/SceneEditor';
import { useVideoSession } from '../../hooks/useVideoSession';
import { Id } from '../../convex/_generated/dataModel';

export default function VideoPage() {
  const router = useRouter();
  const { projectId } = router.query;
  const [selectedSceneId, setSelectedSceneId] = useState<string>();

  const { videoSession, sceneMetadata } = useVideoSession(
    projectId as Id<'projects'>
  );

  const selectedScene = sceneMetadata.find(
    (scene) => scene.sceneId === selectedSceneId
  );

  if (!videoSession) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Loading video...</h2>
          <p className="text-gray-600">Please wait while we prepare your video.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Video Player */}
        <div className="lg:col-span-2">
          <VideoPlayer
            sceneMetadata={sceneMetadata}
            onSceneSelect={setSelectedSceneId}
            selectedSceneId={selectedSceneId}
          />
        </div>

        {/* Scene Editor */}
        <div className="lg:col-span-1">
          {selectedScene ? (
            <SceneEditor
              scene={selectedScene}
              onEditSubmit={(editPrompt) => {
                // This will be implemented in Week 3
                console.log('Edit request:', editPrompt);
              }}
            />
          ) : (
            <div className="bg-gray-50 rounded-lg p-6 text-center">
              <h3 className="text-lg font-medium mb-2">Select a Scene</h3>
              <p className="text-gray-600">
                Click on a scene in the timeline or scene list to edit it.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```


## Week 3: Basic Editing & Feedback Loop

### Day 15-17: Prompt-based Scene Editor UI

#### Task 3.1: Create Scene Editor Component
Create `components/SceneEditor.tsx`:
```typescript
import { useState } from 'react';
import { Button } from './ui/Button';
import { Textarea } from './ui/Textarea';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Loader2, Play, Code } from 'lucide-react';

interface Scene {
  sceneId: string;
  sceneIndex: number;
  title: string;
  duration: number;
  s3ChunkUrl: string;
  sourceCodeS3Url?: string;
  status: 'generating' | 'ready' | 'error';
}

interface SceneEditorProps {
  scene: Scene;
  onEditSubmit: (editPrompt: string) => Promise<void>;
}

export const SceneEditor: React.FC<SceneEditorProps> = ({
  scene,
  onEditSubmit,
}) => {
  const [editPrompt, setEditPrompt] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [sourceCode, setSourceCode] = useState('');

  const handleSubmitEdit = async () => {
    if (!editPrompt.trim()) return;

    setIsEditing(true);
    try {
      await onEditSubmit(editPrompt);
      setEditPrompt('');
    } catch (error) {
      console.error('Error submitting edit:', error);
    } finally {
      setIsEditing(false);
    }
  };

  const handleViewCode = async () => {
    if (!scene.sourceCodeS3Url) return;

    try {
      const response = await fetch(scene.sourceCodeS3Url);
      const code = await response.text();
      setSourceCode(code);
      setShowCode(true);
    } catch (error) {
      console.error('Error fetching source code:', error);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Edit Scene: {scene.title}</span>
          <div className="flex space-x-2">
            {scene.sourceCodeS3Url && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleViewCode}
              >
                <Code className="w-4 h-4 mr-1" />
                View Code
              </Button>
            )}
            <div className={`px-2 py-1 rounded text-xs ${
              scene.status === 'ready' ? 'bg-green-100 text-green-800' :
              scene.status === 'generating' ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              {scene.status}
            </div>
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Scene Info */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium">Duration:</span> {scene.duration}s
          </div>
          <div>
            <span className="font-medium">Scene:</span> {scene.sceneIndex + 1}
          </div>
        </div>

        {/* Edit Prompt */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Describe your changes:
          </label>
          <Textarea
            value={editPrompt}
            onChange={(e) => setEditPrompt(e.target.value)}
            placeholder="e.g., 'Make the text larger and add a blue background' or 'Change the animation to slide in from the left'"
            rows={4}
            disabled={isEditing}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-2">
          <Button
            onClick={handleSubmitEdit}
            disabled={!editPrompt.trim() || isEditing}
            className="flex-1"
          >
            {isEditing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              'Apply Changes'
            )}
          </Button>
        </div>

        {/* Example Prompts */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-2">Example prompts:</h4>
          <div className="space-y-1 text-xs text-gray-600">
            <div>• "Make the text animation faster"</div>
            <div>• "Add a red circle that grows in size"</div>
            <div>• "Change the background color to dark blue"</div>
            <div>• "Add mathematical equations"</div>
          </div>
        </div>
      </CardContent>

      {/* Code Modal */}
      {showCode && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl max-h-[80vh] overflow-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Source Code</h3>
              <Button variant="outline" onClick={() => setShowCode(false)}>
                Close
              </Button>
            </div>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
              <code>{sourceCode}</code>
            </pre>
          </div>
        </div>
      )}
    </Card>
  );
};
```

### Day 18-19: Editing Request to Backend

#### Task 3.2: Create Convex Functions for Editing
Create `convex/sceneEditing.ts`:
```typescript
import { mutation } from "./_generated/server";
import { v } from "convex/values";

export const requestSceneEdit = mutation({
  args: {
    sceneId: v.string(),
    editPrompt: v.string(),
    projectId: v.id("projects"),
  },
  handler: async (ctx, args) => {
    // Find the scene metadata
    const scene = await ctx.db
      .query("sceneMetadata")
      .filter((q) => q.eq(q.field("sceneId"), args.sceneId))
      .first();

    if (!scene) {
      throw new Error("Scene not found");
    }

    // Update scene status to generating
    await ctx.db.patch(scene._id, {
      status: "generating",
      lastModified: Date.now(),
    });

    // Trigger FastAPI webhook
    const webhookUrl = process.env.FASTAPI_WEBHOOK_URL;
    if (webhookUrl) {
      try {
        await fetch(`${webhookUrl}/edit-scene`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            scene_id: args.sceneId,
            edit_prompt: args.editPrompt,
            project_id: args.projectId,
            convex_scene_id: scene._id,
          }),
        });
      } catch (error) {
        console.error('Error triggering FastAPI webhook:', error);
        // Revert status on error
        await ctx.db.patch(scene._id, {
          status: "error",
          lastModified: Date.now(),
        });
        throw new Error("Failed to trigger scene editing");
      }
    }

    return { success: true, sceneId: args.sceneId };
  },
});

export const updateSceneAfterEdit = mutation({
  args: {
    sceneId: v.string(),
    newS3ChunkUrl: v.string(),
    newSourceCodeS3Url: v.string(),
    status: v.union(v.literal("ready"), v.literal("error")),
  },
  handler: async (ctx, args) => {
    const scene = await ctx.db
      .query("sceneMetadata")
      .filter((q) => q.eq(q.field("sceneId"), args.sceneId))
      .first();

    if (!scene) {
      throw new Error("Scene not found");
    }

    await ctx.db.patch(scene._id, {
      s3ChunkUrl: args.newS3ChunkUrl,
      sourceCodeS3Url: args.newSourceCodeS3Url,
      status: args.status,
      lastModified: Date.now(),
    });

    return { success: true };
  },
});
```

#### Task 3.3: Update Frontend to Use Editing Function
Update `hooks/useVideoSession.ts`:
```typescript
import { useQuery, useMutation } from 'convex/react';
import { api } from '../convex/_generated/api';
import { Id } from '../convex/_generated/dataModel';

export const useVideoSession = (projectId: Id<'projects'>) => {
  const videoSessions = useQuery(api.videoSessions.list, { projectId });
  const sceneMetadata = useQuery(
    api.sceneMetadata.list,
    videoSessions?.[0] ? { videoSessionId: videoSessions[0]._id } : 'skip'
  );

  const createVideoSession = useMutation(api.videoSessions.create);
  const requestSceneEdit = useMutation(api.sceneEditing.requestSceneEdit);

  const editScene = async (sceneId: string, editPrompt: string) => {
    if (!projectId) throw new Error('Project ID is required');
    
    return await requestSceneEdit({
      sceneId,
      editPrompt,
      projectId,
    });
  };

  return {
    videoSession: videoSessions?.[0],
    sceneMetadata: sceneMetadata || [],
    createVideoSession,
    editScene,
  };
};
```

### Day 20-21: Scene Re-rendering & S3 Update

#### Task 3.4: Update FastAPI to Handle Scene Editing
Update `main.py` in FastAPI:
```python
from agents.code_generator import CodeGeneratorAgent
from agents.video_generator import VideoGeneratorAgent
import boto3
import requests

# Initialize agents
code_generator = CodeGeneratorAgent()
video_generator = VideoGeneratorAgent()

@app.post("/edit-scene")
async def edit_scene(request: SceneEditRequest, background_tasks: BackgroundTasks):
    """Handle scene editing request from Convex"""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(process_scene_edit, request, job_id)
    
    return {"job_id": job_id, "status": "started"}

async def process_scene_edit(request: SceneEditRequest, job_id: str):
    """Process scene editing in background"""
    try:
        # 1. Retrieve original code from S3
        original_code = await retrieve_scene_code_from_s3(request.scene_id)
        
        # 2. Generate edited code using AI agent
        edited_code = code_generator.edit_scene_code(original_code, request.edit_prompt)
        
        # 3. Render the new scene
        new_video_path = await render_scene_code(edited_code, request.scene_id)
        
        # 4. Upload new video chunk to S3
        new_s3_chunk_url = video_generator._upload_chunk_to_s3(
            new_video_path, f"{request.scene_id}_edited"
        )
        
        # 5. Upload new source code to S3
        new_source_code_s3_url = video_generator._upload_code_to_s3(
            edited_code, f"{request.scene_id}_edited"
        )
        
        # 6. Update Convex with new URLs
        await update_convex_scene(
            request.scene_id,
            new_s3_chunk_url,
            new_source_code_s3_url,
            "ready"
        )
        
    except Exception as e:
        print(f"Error processing scene edit: {e}")
        # Update Convex with error status
        await update_convex_scene(request.scene_id, "", "", "error")

async def retrieve_scene_code_from_s3(scene_id: str) -> str:
    """Retrieve original scene code from S3"""
    s3_client = boto3.client('s3')
    bucket_name = os.getenv('S3_BUCKET_NAME')
    
    try:
        key = f"source-code/{scene_id}.py"
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Error retrieving code from S3: {e}")
        raise

async def render_scene_code(code: str, scene_id: str) -> str:
    """Render Manim code to video file"""
    import tempfile
    import subprocess
    
    # Create temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_code_path = f.name
    
    # Create output directory
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, f"{scene_id}_edited.mp4")
    
    try:
        # Run Manim command
        cmd = [
            'manim', temp_code_path, 
            '--format', 'mp4',
            '--output_file', output_path,
            '--resolution', '1080p',
            '--fps', '30'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Manim rendering failed: {result.stderr}")
        
        return output_path
        
    finally:
        # Clean up temporary code file
        os.unlink(temp_code_path)

async def update_convex_scene(scene_id: str, s3_chunk_url: str, source_code_s3_url: str, status: str):
    """Update Convex with new scene data"""
    convex_webhook_url = os.getenv('CONVEX_WEBHOOK_URL')
    
    if convex_webhook_url:
        try:
            await requests.post(f"{convex_webhook_url}/update-scene", json={
                'scene_id': scene_id,
                'new_s3_chunk_url': s3_chunk_url,
                'new_source_code_s3_url': source_code_s3_url,
                'status': status
            })
        except Exception as e:
            print(f"Error updating Convex: {e}")
```

## Week 4: Integration, Testing & Polish

### Day 22-24: End-to-End Workflow Testing

#### Task 4.1: Create Test Suite
Create `tests/test_workflow.py`:
```python
import pytest
import requests
import time
from convex import ConvexClient

class TestVideoWorkflow:
    def setup_method(self):
        self.convex_client = ConvexClient(url=os.getenv('CONVEX_URL'))
        self.fastapi_url = os.getenv('FASTAPI_URL', 'http://localhost:8000')
    
    def test_video_generation_workflow(self):
        """Test complete video generation workflow"""
        # 1. Create project
        project_id = self.convex_client.mutation('projects:create', {
            'name': 'Test Project',
            'description': 'Test video generation',
            'userId': 'test-user-id'
        })
        
        # 2. Trigger video generation
        response = requests.post(f"{self.fastapi_url}/generate-video", json={
            'project_id': project_id,
            'content_prompt': 'Explain basic algebra',
            'scene_count': 3
        })
        
        assert response.status_code == 200
        job_id = response.json()['job_id']
        
        # 3. Wait for completion and verify scenes
        self._wait_for_video_completion(project_id)
        
        # 4. Verify scenes are created
        scenes = self.convex_client.query('sceneMetadata:list', {
            'videoSessionId': self._get_video_session_id(project_id)
        })
        
        assert len(scenes) == 3
        assert all(scene['status'] == 'ready' for scene in scenes)
    
    def test_scene_editing_workflow(self):
        """Test scene editing workflow"""
        # Setup: Create a project with scenes
        project_id = self._create_test_project_with_scenes()
        scenes = self._get_project_scenes(project_id)
        
        # 1. Request scene edit
        edit_response = self.convex_client.mutation('sceneEditing:requestSceneEdit', {
            'sceneId': scenes[0]['sceneId'],
            'editPrompt': 'Make the text bigger and blue',
            'projectId': project_id
        })
        
        assert edit_response['success'] == True
        
        # 2. Wait for edit completion
        self._wait_for_scene_edit_completion(scenes[0]['sceneId'])
        
        # 3. Verify scene was updated
        updated_scene = self._get_scene_by_id(scenes[0]['sceneId'])
        assert updated_scene['status'] == 'ready'
        assert updated_scene['lastModified'] > scenes[0]['lastModified']
```

#### Task 4.2: Create Integration Test Script
Create `scripts/test_integration.sh`:
```bash
#!/bin/bash

echo "Starting integration tests..."

# Start services
echo "Starting MCP server..."
cd mcp_server && python main.py &
MCP_PID=$!

echo "Starting FastAPI server..."
cd ../ai-video-tutor-api && python main.py &
FASTAPI_PID=$!

echo "Starting Convex dev server..."
cd ../ai-video-tutor-backend && npx convex dev &
CONVEX_PID=$!

echo "Starting Next.js frontend..."
cd ../ai-video-tutor-frontend && npm run dev &
NEXTJS_PID=$!

# Wait for services to start
sleep 10

# Run tests
echo "Running integration tests..."
cd ../tests && python -m pytest test_workflow.py -v

# Cleanup
echo "Cleaning up..."
kill $MCP_PID $FASTAPI_PID $CONVEX_PID $NEXTJS_PID

echo "Integration tests completed!"
```

### Day 25-26: Performance & Optimization

#### Task 4.3: Add Performance Monitoring
Create `utils/performance.py`:
```python
import time
import logging
from functools import wraps

def monitor_performance(func_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logging.info(f"{func_name} completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logging.error(f"{func_name} failed after {execution_time:.2f}s: {e}")
                raise
        return wrapper
    return decorator

# Usage in agents
@monitor_performance("scene_code_generation")
async def generate_scene_code(self, scene_prompt: str, scene_index: int) -> str:
    # ... existing code
```

#### Task 4.4: Implement Basic Caching
Create `utils/cache.py`:
```python
import redis
import json
import os
from typing import Any, Optional

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            return self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

# Usage in code generator
cache = CacheManager()

def get_cached_mcp_context(self, prompt: str) -> str:
    cache_key = f"mcp_context:{hash(prompt)}"
    cached_context = cache.get(cache_key)
    
    if cached_context:
        return cached_context
    
    context = self._get_mcp_context(prompt)
    cache.set(cache_key, context, ttl=1800)  # 30 minutes
    return context
```

### Day 27-28: User Experience Polish

#### Task 4.5: Add Loading States and Error Handling
Update `components/VideoPlayer.tsx`:
```typescript
// Add loading and error states
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState<string | null>(null);

// Add error boundary
if (error) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
      <h3 className="text-lg font-semibold text-red-800 mb-2">Error Loading Video</h3>
      <p className="text-red-600 mb-4">{error}</p>
      <Button onClick={() => window.location.reload()}>
        Reload Page
      </Button>
    </div>
  );
}

// Add loading state
if (isLoading || !sceneMetadata.length) {
  return (
    <div className="bg-gray-50 rounded-lg p-8 text-center">
      <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
      <h3 className="text-lg font-semibold mb-2">Loading Video...</h3>
      <p className="text-gray-600">Please wait while we prepare your video scenes.</p>
    </div>
  );
}
```

### Day 29-30: Documentation & Demo Preparation

#### Task 4.6: Create User Documentation
Create `docs/user-guide.md`:
```markdown
# AI Video Tutor - User Guide

## Getting Started

### Creating Your First Video
1. Navigate to the dashboard
2. Click "Create New Project"
3. Enter your content prompt (e.g., "Explain quadratic equations")
4. Wait for video generation (typically 2-3 minutes)

### Editing Scenes
1. Click on any scene in the timeline
2. Enter your editing prompt in the sidebar
3. Click "Apply Changes"
4. Wait for the scene to re-render

### Best Practices for Prompts
- Be specific about visual elements
- Mention colors, sizes, and animations
- Use educational language
- Keep prompts concise but descriptive

## Troubleshooting
- If a scene fails to generate, try a simpler prompt
- Check your internet connection for video playback issues
- Contact support if problems persist
```

This comprehensive implementation plan provides specific, actionable tasks for each day of the month-long development cycle, with actual code examples and detailed technical specifications for building the MVP.

