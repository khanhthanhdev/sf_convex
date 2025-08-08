"""
File watcher service for configuration hot-reloading.

This module provides file watching capabilities for .env files during development,
enabling automatic configuration reloading when files change.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

logger = logging.getLogger(__name__)


class ConfigFileEventHandler(FileSystemEventHandler):
    """Event handler for configuration file changes."""
    
    def __init__(self, callback: Callable[[str], None], watched_files: List[str]):
        """Initialize the event handler.
        
        Args:
            callback: Function to call when a watched file changes
            watched_files: List of file names to watch (e.g., ['.env', '.env.local'])
        """
        super().__init__()
        self.callback = callback
        self.watched_files = set(watched_files)
        self.last_modified: Dict[str, float] = {}
        self.debounce_delay = 0.5  # 500ms debounce to avoid multiple rapid events
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        self._handle_file_event(event.src_path, "modified")
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        self._handle_file_event(event.src_path, "created")
    
    def _handle_file_event(self, file_path: str, event_type: str):
        """Handle file system events with debouncing."""
        file_name = Path(file_path).name
        
        # Check if this is a file we're watching
        if file_name not in self.watched_files:
            return
        
        # Debounce rapid events
        current_time = time.time()
        last_time = self.last_modified.get(file_path, 0)
        
        if current_time - last_time < self.debounce_delay:
            return
        
        self.last_modified[file_path] = current_time
        
        logger.info(f"Configuration file {event_type}: {file_path}")
        
        try:
            # Call the callback function
            self.callback(file_path)
        except Exception as e:
            logger.error(f"Error handling file change event for {file_path}: {e}")


class ConfigFileWatcher:
    """File watcher for configuration files with hot-reloading support."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the file watcher.
        
        Args:
            base_path: Base directory to watch. Defaults to current directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[ConfigFileEventHandler] = None
        self.is_watching = False
        self.watched_files: List[str] = []
        self.change_callbacks: List[Callable[[str], None]] = []
        self._lock = threading.RLock()
        
        # Development mode check
        self.development_mode = os.getenv('ENVIRONMENT', 'development') == 'development'
        self.hot_reload_enabled = self.development_mode and os.getenv('ENABLE_HOT_RELOAD', 'true').lower() == 'true'
        
        logger.info(f"ConfigFileWatcher initialized (development_mode={self.development_mode}, hot_reload={self.hot_reload_enabled})")
    
    def add_callback(self, callback: Callable[[str], None]):
        """Add a callback function to be called when files change.
        
        Args:
            callback: Function that takes a file path as argument
        """
        with self._lock:
            if callback not in self.change_callbacks:
                self.change_callbacks.append(callback)
                logger.debug(f"Added file change callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable[[str], None]):
        """Remove a callback function.
        
        Args:
            callback: Function to remove
        """
        with self._lock:
            if callback in self.change_callbacks:
                self.change_callbacks.remove(callback)
                logger.debug(f"Removed file change callback: {callback.__name__}")
    
    def start_watching(self, files_to_watch: Optional[List[str]] = None) -> bool:
        """Start watching configuration files.
        
        Args:
            files_to_watch: List of file names to watch. Defaults to common .env files.
            
        Returns:
            True if watching started successfully, False otherwise
        """
        if not self.hot_reload_enabled:
            logger.info("Hot-reload is disabled - file watching not started")
            return False
        
        if self.is_watching:
            logger.warning("File watcher is already running")
            return True
        
        if files_to_watch is None:
            files_to_watch = ['.env', '.env.local', '.env.development', '.env.development.local']
        
        with self._lock:
            try:
                # Create event handler
                self.event_handler = ConfigFileEventHandler(
                    callback=self._on_file_changed,
                    watched_files=files_to_watch
                )
                
                # Create and start observer
                self.observer = Observer()
                self.observer.schedule(
                    self.event_handler,
                    path=str(self.base_path),
                    recursive=False
                )
                
                self.observer.start()
                self.is_watching = True
                self.watched_files = files_to_watch.copy()
                
                logger.info(f"Started watching configuration files: {files_to_watch}")
                logger.info(f"Watching directory: {self.base_path}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to start file watcher: {e}")
                self._cleanup_observer()
                return False
    
    def stop_watching(self):
        """Stop watching configuration files."""
        with self._lock:
            if not self.is_watching:
                return
            
            try:
                self._cleanup_observer()
                logger.info("Stopped watching configuration files")
                
            except Exception as e:
                logger.error(f"Error stopping file watcher: {e}")
    
    def _cleanup_observer(self):
        """Clean up the observer and related resources."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5.0)  # Wait up to 5 seconds
            except Exception as e:
                logger.warning(f"Error stopping observer: {e}")
            finally:
                self.observer = None
        
        self.event_handler = None
        self.is_watching = False
        self.watched_files.clear()
    
    def _on_file_changed(self, file_path: str):
        """Handle file change events by notifying all callbacks.
        
        Args:
            file_path: Path to the changed file
        """
        logger.info(f"Configuration file changed: {file_path}")
        
        # Notify all registered callbacks
        with self._lock:
            for callback in self.change_callbacks.copy():  # Copy to avoid modification during iteration
                try:
                    callback(file_path)
                except Exception as e:
                    logger.error(f"Error in file change callback {callback.__name__}: {e}")
    
    def is_file_watched(self, file_name: str) -> bool:
        """Check if a file is being watched.
        
        Args:
            file_name: Name of the file to check
            
        Returns:
            True if the file is being watched, False otherwise
        """
        return file_name in self.watched_files
    
    def get_watched_files(self) -> List[str]:
        """Get list of files currently being watched.
        
        Returns:
            List of watched file names
        """
        return self.watched_files.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the file watcher.
        
        Returns:
            Dictionary with watcher status information
        """
        return {
            'is_watching': self.is_watching,
            'development_mode': self.development_mode,
            'hot_reload_enabled': self.hot_reload_enabled,
            'watched_files': self.watched_files.copy(),
            'callback_count': len(self.change_callbacks),
            'base_path': str(self.base_path)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure watcher is stopped."""
        self.stop_watching()