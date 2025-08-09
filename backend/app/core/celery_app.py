import os
from celery import Celery
from .settings import settings

# Celery configuration
broker_url = settings.get_celery_broker_url()
result_backend = settings.get_celery_result_backend()

# Create Celery app
celery_app = Celery(
    "ai_video_backend",
    broker=broker_url,
    backend=result_backend,
    include=['app.tasks.video_tasks', 'app.tasks.hello_tasks']  # Import task modules
)

# Configure Celery
celery_app.conf.update(
    # Task routing
    task_default_queue="default",
    task_routes={
        'app.tasks.video_tasks.*': {'queue': 'render'},
        'app.tasks.hello_tasks.*': {'queue': 'default'},
    },
    
    # Queue configuration
    task_queues={
        "default": {
            "exchange": "default",
            "routing_key": "default",
        },
        "render": {
            "exchange": "render", 
            "routing_key": "render",
        },
        "high": {
            "exchange": "high",
            "routing_key": "high",
        },
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Task configuration
    task_time_limit=60*60,           # 1 hour hard timeout
    task_soft_time_limit=60*55,      # 55 minutes soft timeout
    task_acks_late=True,             # Acknowledge tasks after completion
    task_reject_on_worker_lost=True, # Reject tasks if worker dies
    
    # Result configuration
    result_expires=3600,             # 1 hour
    result_persistent=True,
    
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Celery signal for auto-discovery
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks if needed"""
    # Example: sender.add_periodic_task(30.0, test_task.s(), name='test every 30s')
    pass

# For backwards compatibility
celery = celery_app
