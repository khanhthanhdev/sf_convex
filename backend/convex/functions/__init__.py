# This file makes the functions directory a Python package
# Import all functions to make them available at the package level
from .video_sessions import (
    create_video_session,
    update_video_session_status,
    get_video_session,
    list_video_sessions,
)

from .scenes import (
    create_scene,
    update_scene_status,
    get_scene,
    list_scenes,
    get_next_queued_scene,
)

from .assets import (
    update_scene_assets,
    update_session_assets,
    batch_update_assets,
    delete_asset_references,
    get_assets_by_entity,
)
