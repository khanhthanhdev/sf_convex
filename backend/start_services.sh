#!/bin/bash

# AI Video Tutor Backend Services Startup Script
# This script starts Redis, Celery worker, and FastAPI server

set -e

echo "🚀 Starting AI Video Tutor Backend Services..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Please run:"
    echo "   source .venv/bin/activate"
    exit 1
fi

# Check if Redis is available
echo "📡 Checking Redis connection..."
if ! redis-cli ping >/dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   Option 1: redis-server"
    echo "   Option 2: docker run -d -p 6379:6379 redis:alpine"
    exit 1
fi
echo "✅ Redis is running"

# Create logs directory
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "✅ All services stopped"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start Celery worker in background
echo "🔄 Starting Celery worker..."
python celery_worker.py > logs/celery_worker.log 2>&1 &
CELERY_PID=$!
echo "✅ Celery worker started (PID: $CELERY_PID)"

# Wait a moment for worker to start
sleep 2

# Start FastAPI server
echo "🌐 Starting FastAPI server..."
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🏥 Health check available at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start FastAPI server (this will run in foreground)
python run_dev.py

echo "🏁 Startup script completed"
