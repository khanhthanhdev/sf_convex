"""
Celery tasks for AI Video Tutor Backend
"""

from .hello_tasks import hello_task, add_numbers
from .video_tasks import generate_video_task, edit_scene_task

__all__ = [
    'hello_task',
    'add_numbers', 
    'generate_video_task',
    'edit_scene_task'
]
