"""
Simple hello world and example tasks for testing Celery
"""
import time
from celery import current_task
from app.core.celery_app import celery_app
from app.core.redis_client import redis_client

@celery_app.task(bind=True, name="hello_task")
def hello_task(self, name: str = "World") -> dict:
    """
    Simple hello world task for testing Celery
    
    Args:
        name: Name to greet
        
    Returns:
        dict: Task result with greeting message
    """
    task_id = self.request.id
    
    # Update task state
    self.update_state(
        state='PROGRESS',
        meta={'current': 1, 'total': 3, 'status': 'Starting...'}
    )
    
    time.sleep(1)  # Simulate work
    
    # Update task state
    self.update_state(
        state='PROGRESS', 
        meta={'current': 2, 'total': 3, 'status': 'Processing...'}
    )
    
    time.sleep(1)  # Simulate more work
    
    # Cache result in Redis
    result = {
        'task_id': task_id,
        'message': f'Hello, {name}!',
        'timestamp': time.time(),
        'worker': self.request.hostname
    }
    
    # Store in Redis for 1 hour
    redis_client.set_json(f"task_result:{task_id}", result, ex=3600)
    
    return result

@celery_app.task(bind=True, name="add_numbers")
def add_numbers(self, x: int, y: int) -> dict:
    """
    Simple addition task for testing
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        dict: Task result with sum
    """
    task_id = self.request.id
    
    # Simulate some processing time
    time.sleep(2)
    
    result = {
        'task_id': task_id,
        'x': x,
        'y': y,
        'sum': x + y,
        'timestamp': time.time()
    }
    
    return result

@celery_app.task(bind=True, name="long_running_task")
def long_running_task(self, duration: int = 10) -> dict:
    """
    Long running task for testing progress updates
    
    Args:
        duration: How long to run (seconds)
        
    Returns:
        dict: Task completion result
    """
    task_id = self.request.id
    
    for i in range(duration):
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i + 1,
                'total': duration,
                'status': f'Step {i + 1} of {duration}',
                'percentage': int((i + 1) / duration * 100)
            }
        )
        time.sleep(1)
    
    result = {
        'task_id': task_id,
        'duration': duration,
        'status': 'completed',
        'timestamp': time.time()
    }
    
    return result
