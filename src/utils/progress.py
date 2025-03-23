"""
Progress tracking utilities for SubWhisper.
"""

import sys
import time
import logging
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ProgressStage(Enum):
    """Enum for different stages of the SubWhisper pipeline."""
    INITIALIZE = "initialize"
    VALIDATE_VIDEO = "validate_video"
    EXTRACT_AUDIO = "extract_audio"
    PROCESS_AUDIO = "process_audio"
    DETECT_LANGUAGE = "detect_language"
    TRANSCRIBE = "transcribe"
    GENERATE_SUBTITLES = "generate_subtitles"
    FORMAT_SUBTITLES = "format_subtitles"
    COMPLETE = "complete"

class ProgressCallback:
    """Callback for progress updates."""
    
    def __init__(self, callback_fn: Callable[[str, float, Optional[Dict[str, Any]]], None]):
        """
        Initialize progress callback.
        
        Args:
            callback_fn: Function to call with progress updates
                         Arguments: (stage, progress, metadata)
        """
        self.callback_fn = callback_fn
    
    def update(self, stage: Union[str, ProgressStage], progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress.
        
        Args:
            stage: Current stage
            progress: Progress value (0.0 to 1.0)
            metadata: Additional metadata
        """
        if isinstance(stage, ProgressStage):
            stage = stage.value
        
        self.callback_fn(stage, progress, metadata)

class ProgressManager:
    """Manager for tracking progress of the SubWhisper pipeline."""
    
    def __init__(self, verbose: bool = True, callback: Optional[ProgressCallback] = None):
        """
        Initialize progress manager.
        
        Args:
            verbose: Whether to print progress to console
            callback: Optional callback for progress updates
        """
        self.verbose = verbose
        self.callback = callback
        self.stages = list(ProgressStage)
        self.current_stage_index = 0
        self.current_stage = self.stages[self.current_stage_index]
        self.progress = 0.0
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.stage_durations = {}
        self.metadata = {}
        self.tqdm_instances = {}
    
    def start_stage(self, stage: Union[str, ProgressStage], desc: Optional[str] = None) -> None:
        """
        Start a new stage.
        
        Args:
            stage: Stage to start
            desc: Optional description for the stage
        """
        if isinstance(stage, str):
            try:
                stage = ProgressStage(stage)
            except ValueError:
                logger.warning(f"Unknown progress stage: {stage}")
                return
        
        self.current_stage = stage
        try:
            self.current_stage_index = self.stages.index(stage)
        except ValueError:
            self.current_stage_index = 0
        
        self.progress = 0.0
        self.stage_start_time = time.time()
        
        if desc is None:
            desc = f"Stage {self.current_stage_index + 1}/{len(self.stages)}: {stage.value.replace('_', ' ').title()}"
        
        if self.verbose:
            # Close previous tqdm instance if it exists
            if self.current_stage.value in self.tqdm_instances:
                self.tqdm_instances[self.current_stage.value].close()
            
            # Create new tqdm instance
            self.tqdm_instances[self.current_stage.value] = tqdm(
                total=100,
                desc=desc,
                unit="%",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}"
            )
        
        # Call callback
        if self.callback:
            self.callback.update(stage, 0.0, {
                "desc": desc,
                "stage_index": self.current_stage_index,
                "total_stages": len(self.stages),
                "start_time": self.stage_start_time,
            })
    
    def update(self, progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress for the current stage.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            metadata: Additional metadata
        """
        # Ensure progress is between 0 and 1
        progress = max(0.0, min(1.0, progress))
        
        # Update progress
        self.progress = progress
        
        # Update metadata
        if metadata:
            self.metadata.update(metadata)
        
        # Update tqdm
        if self.verbose and self.current_stage.value in self.tqdm_instances:
            self.tqdm_instances[self.current_stage.value].n = int(progress * 100)
            self.tqdm_instances[self.current_stage.value].refresh()
        
        # Call callback
        if self.callback:
            self.callback.update(self.current_stage, progress, metadata)
    
    def complete_stage(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Complete the current stage.
        
        Args:
            metadata: Additional metadata
        """
        # Calculate stage duration
        stage_duration = time.time() - self.stage_start_time
        self.stage_durations[self.current_stage.value] = stage_duration
        
        # Update progress to 100%
        self.update(1.0, metadata)
        
        # Close tqdm instance
        if self.verbose and self.current_stage.value in self.tqdm_instances:
            self.tqdm_instances[self.current_stage.value].close()
        
        # Move to next stage if not the last stage
        if self.current_stage != self.stages[-1]:
            next_stage_index = self.current_stage_index + 1
            if next_stage_index < len(self.stages):
                next_stage = self.stages[next_stage_index]
                self.start_stage(next_stage)
    
    def complete(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete all stages.
        
        Args:
            metadata: Additional metadata
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate total duration
        total_duration = time.time() - self.start_time
        
        # Complete current stage
        self.complete_stage(metadata)
        
        # If not already at the last stage, move to the last stage
        if self.current_stage != self.stages[-1]:
            self.start_stage(self.stages[-1])
            self.complete_stage()
        
        # Close all tqdm instances
        for tqdm_instance in self.tqdm_instances.values():
            tqdm_instance.close()
        
        # Print summary
        if self.verbose:
            print("\nProcessing completed in {:.2f} seconds".format(total_duration))
            for stage, duration in self.stage_durations.items():
                print("  {}: {:.2f} seconds ({:.1f}%)".format(
                    stage.replace("_", " ").title(),
                    duration,
                    (duration / total_duration) * 100
                ))
        
        # Collect metrics
        metrics = {
            "total_duration": total_duration,
            "stage_durations": self.stage_durations,
            "start_time": self.start_time,
            "end_time": time.time(),
        }
        
        # Call callback
        if self.callback:
            combined_metadata = {}
            if metadata:
                combined_metadata.update(metadata)
            combined_metadata.update(metrics)
            self.callback.update(ProgressStage.COMPLETE, 1.0, combined_metadata)
        
        return metrics
    
    def overall_progress(self) -> float:
        """
        Get the overall progress of the pipeline.
        
        Returns:
            Overall progress as a value between 0.0 and 1.0
        """
        # Calculate progress as (completed_stages + current_stage_progress) / total_stages
        completed_stages = self.current_stage_index
        current_progress = self.progress
        total_stages = len(self.stages)
        
        return (completed_stages + current_progress) / total_stages

def create_console_callback() -> ProgressCallback:
    """
    Create a progress callback that prints to the console.
    
    Returns:
        ProgressCallback instance
    """
    def console_callback(stage: str, progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Print progress to the console."""
        metadata_str = ""
        if metadata:
            metadata_str = " " + ", ".join(f"{k}: {v}" for k, v in metadata.items())
        
        print(f"\r{stage}: {progress:.1%}{metadata_str}", end="")
        sys.stdout.flush()
    
    return ProgressCallback(console_callback)

def create_logger_callback(logger: logging.Logger) -> ProgressCallback:
    """
    Create a progress callback that logs to a logger.
    
    Args:
        logger: Logger to log to
        
    Returns:
        ProgressCallback instance
    """
    def logger_callback(stage: str, progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log progress to the logger."""
        metadata_str = ""
        if metadata:
            metadata_str = " " + ", ".join(f"{k}: {v}" for k, v in metadata.items())
        
        logger.debug(f"{stage}: {progress:.1%}{metadata_str}")
    
    return ProgressCallback(logger_callback) 