"""
Progress tracking module for SubWhisper.

This module provides utilities for tracking and reporting progress of
long-running operations.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from tqdm import tqdm

from src.utils.errors import SubWhisperError

logger = logging.getLogger("subwhisper")

# Global registry of progress trackers
_active_trackers: Dict[str, 'ProgressTracker'] = {}


class ProgressTracker:
    """
    Progress tracker for long-running operations.
    
    This class provides a way to track and report progress of long-running
    operations like audio processing, transcription, etc.
    """
    
    def __init__(self, 
                name: str, 
                total: int, 
                description: str = "", 
                unit: str = "it", 
                auto_refresh: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            name: Unique name for the tracker
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress (e.g., "files", "MB", etc.)
            auto_refresh: Whether to automatically refresh the progress bar
        """
        self.name = name
        self.total = total
        self.description = description
        self.n = 0
        self.unit = unit
        self.auto_refresh = auto_refresh
        self.started_at = time.time()
        self.completed = False
        self.failed = False
        self.error_message = ""
        self.data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Create progress bar
        self.progress_bar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            leave=True
        )
        
        # Register the tracker
        with self._lock:
            _active_trackers[name] = self
    
    def update(self, n: int = 1) -> None:
        """
        Update progress by the specified amount.
        
        Args:
            n: Number of steps to increment
        """
        with self._lock:
            self.n += n
            if self.n > self.total:
                self.n = self.total
            
            # Update the progress bar
            self.progress_bar.update(n)
    
    def set_description(self, description: str) -> None:
        """
        Set the description of the progress bar.
        
        Args:
            description: New description
        """
        with self._lock:
            self.description = description
            self.progress_bar.set_description(description)
    
    def set_postfix(self, **kwargs) -> None:
        """
        Set the postfix of the progress bar.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        with self._lock:
            self.progress_bar.set_postfix(**kwargs)
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark the operation as completed.
        
        Args:
            message: Optional completion message
        """
        with self._lock:
            if not self.completed and not self.failed:
                self.n = self.total
                self.completed = True
                
                if message:
                    self.set_description(f"{self.description} - {message}")
                else:
                    self.set_description(f"{self.description} - Completed")
                
                self.progress_bar.update(self.total - self.progress_bar.n)
                self.progress_bar.close()
                
                logger.info(f"Operation '{self.name}' completed in {time.time() - self.started_at:.2f}s")
                
                # Unregister the tracker
                _active_trackers.pop(self.name, None)
    
    def fail(self, error_message: str) -> None:
        """
        Mark the operation as failed.
        
        Args:
            error_message: Error message to display
        """
        with self._lock:
            if not self.completed and not self.failed:
                self.failed = True
                self.error_message = error_message
                
                self.set_description(f"{self.description} - Failed: {error_message}")
                self.progress_bar.close()
                
                logger.error(f"Operation '{self.name}' failed: {error_message}")
                
                # Unregister the tracker
                _active_trackers.pop(self.name, None)
    
    def __enter__(self) -> 'ProgressTracker':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.fail(str(exc_val))
        elif not self.completed and not self.failed:
            self.complete()


def track_progress(name: str, 
                  total: int, 
                  description: str = "", 
                  unit: str = "it",
                  auto_refresh: bool = True) -> ProgressTracker:
    """
    Create and return a new progress tracker.
    
    Args:
        name: Unique name for the tracker
        total: Total number of steps
        description: Description of the operation
        unit: Unit of progress (e.g., "files", "MB", etc.)
        auto_refresh: Whether to automatically refresh the progress bar
    
    Returns:
        ProgressTracker instance
    
    Raises:
        SubWhisperError: If a tracker with the same name already exists
    """
    if name in _active_trackers:
        raise SubWhisperError(
            f"Progress tracker '{name}' already exists",
            code="DUPLICATE_TRACKER"
        )
    
    return ProgressTracker(name, total, description, unit, auto_refresh)


def get_tracker(name: str) -> Optional[ProgressTracker]:
    """
    Get an existing progress tracker by name.
    
    Args:
        name: Name of the tracker
    
    Returns:
        ProgressTracker instance or None if not found
    """
    return _active_trackers.get(name)


def track_function(name: str, description: str = "", unit: str = "it") -> Callable:
    """
    Decorator to track progress of a function.
    
    The decorated function should either:
    1. Accept a 'progress_tracker' keyword argument, or
    2. Accept **kwargs to receive the tracker
    
    Args:
        name: Base name for the tracker (will be made unique)
        description: Description of the operation
        unit: Unit of progress
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create unique name based on function
            unique_name = f"{name}_{id(func)}"
            
            # Create tracker only if not already provided
            if "progress_tracker" not in kwargs:
                tracker = track_progress(unique_name, 100, description, unit)
                kwargs["progress_tracker"] = tracker
            else:
                tracker = kwargs["progress_tracker"]
            
            try:
                with tracker:
                    return func(*args, **kwargs)
            except Exception as e:
                tracker.fail(str(e))
                raise
        
        return wrapper
    
    return decorator 