# Requirements Document

## Introduction

The AI Video Tutor platform is a full-stack application that enables users to generate educational videos using AI-powered Manim animations. Users can input text prompts to create multi-scene videos, edit individual scenes through natural language prompts, and preview their content in real-time. The system breaks down video generation into manageable chunks, allowing for granular editing and efficient processing.

## Requirements

### Requirement 1: Backend Infrastructure Setup

**User Story:** As a developer, I want a robust backend infrastructure so that I can build scalable video generation services.

#### Acceptance Criteria

1. WHEN the system is initialized THEN it SHALL have FastAPI serving REST endpoints
2. WHEN a request is made THEN Convex.dev SHALL handle data persistence and real-time updates
3. WHEN background tasks are needed THEN Celery workers SHALL process them asynchronously
4. WHEN task queuing is required THEN Redis SHALL serve as the message broker and result store
5. WHEN the system starts THEN all services SHALL be properly connected and verified through a hello-world demo

### Requirement 2: Video Generation Pipeline

**User Story:** As a user, I want to generate educational videos from text prompts so that I can create animated content without manual coding.

#### Acceptance Criteria

1. WHEN a user submits a content prompt THEN the system SHALL break it into multiple scenes
2. WHEN scenes are identified THEN each scene SHALL be processed into Manim code
3. WHEN Manim code is generated THEN it SHALL be rendered into video chunks
4. WHEN video chunks are created THEN they SHALL be uploaded to S3 with versioned keys
5. WHEN processing is complete THEN the system SHALL update Convex with scene metadata and URLs

### Requirement 3: Scene-Based Video Editing

**User Story:** As a user, I want to edit individual scenes using natural language prompts so that I can refine my video content without starting over.

#### Acceptance Criteria

1. WHEN a user selects a scene THEN the system SHALL display the current scene content
2. WHEN a user submits an edit prompt THEN the system SHALL modify the Manim code accordingly
3. WHEN code is modified THEN the scene SHALL be re-rendered and uploaded to S3
4. WHEN re-rendering is complete THEN the frontend SHALL update with the new video chunk
5. WHEN editing fails THEN the system SHALL preserve the original scene and show error messages

### Requirement 4: Real-time Video Preview

**User Story:** As a user, I want to preview my video with scene-accurate playback so that I can see the complete video experience.

#### Acceptance Criteria

1. WHEN video chunks are available THEN Remotion SHALL stitch them into a seamless playback
2. WHEN a user seeks in the timeline THEN the player SHALL jump to the correct scene and frame
3. WHEN scenes have different durations THEN the timeline SHALL show accurate scene boundaries
4. WHEN a scene is being edited THEN the UI SHALL show loading states and progress
5. WHEN playback occurs THEN video chunks SHALL load efficiently from CloudFront CDN

### Requirement 5: Project and Session Management

**User Story:** As a user, I want to organize my videos into projects so that I can manage multiple educational topics.

#### Acceptance Criteria

1. WHEN a user creates a project THEN it SHALL be stored with metadata in Convex
2. WHEN a user generates a video THEN it SHALL create a new video session linked to the project
3. WHEN scenes are created THEN they SHALL be associated with the correct session and project
4. WHEN a user accesses a project THEN they SHALL see all associated video sessions
5. WHEN projects are listed THEN users SHALL only see projects they own

### Requirement 6: Authentication and Security

**User Story:** As a user, I want secure access to my projects so that my content remains private and protected.

#### Acceptance Criteria

1. WHEN a user accesses the system THEN they SHALL authenticate via Convex Auth
2. WHEN API calls are made THEN webhooks SHALL be HMAC-signed with timestamps
3. WHEN timestamps are stale (>5 minutes) THEN requests SHALL be rejected
4. WHEN S3 operations occur THEN write credentials SHALL only be available server-side
5. WHEN users access data THEN they SHALL only see projects they own

### Requirement 7: Error Handling and Monitoring

**User Story:** As a user, I want clear feedback when things go wrong so that I can understand and resolve issues.

#### Acceptance Criteria

1. WHEN errors occur THEN they SHALL be logged with correlation IDs
2. WHEN video generation fails THEN users SHALL see descriptive error messages
3. WHEN scene rendering fails THEN the system SHALL retry with exponential backoff
4. WHEN maximum retries are reached THEN scenes SHALL be marked as error state
5. WHEN errors are displayed THEN users SHALL have options to retry or get help