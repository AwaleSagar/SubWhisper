"""
Timestamp synchronization module for SubWhisper.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.utils.config import Config
from src.subtitles.generator import Subtitle

logger = logging.getLogger("subwhisper")

class TimestampSyncError(Exception):
    """Exception raised for errors in timestamp synchronization."""
    pass

class SyncEngine:
    """Timestamp synchronization engine."""
    
    def __init__(self, config: Config):
        """
        Initialize synchronization engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def adjust_timing(self, subtitles: List[Subtitle], 
                      offset: float = 0.0, 
                      scale_factor: float = 1.0) -> List[Subtitle]:
        """
        Adjust subtitle timing with offset and scale factor.
        
        Args:
            subtitles: List of Subtitle objects
            offset: Time offset in seconds (positive or negative)
            scale_factor: Scale factor for timing (default: 1.0)
            
        Returns:
            List of Subtitle objects with adjusted timing
        """
        try:
            logger.info(f"Adjusting subtitle timing (offset: {offset:.2f}s, scale: {scale_factor:.2f})")
            
            adjusted_subtitles = []
            
            for subtitle in subtitles:
                # Apply offset and scale factor
                start_time = max(0, subtitle.start * scale_factor + offset)
                end_time = max(start_time + 0.1, subtitle.end * scale_factor + offset)
                
                # Create adjusted subtitle
                adjusted_subtitle = Subtitle(
                    index=subtitle.index,
                    start=start_time,
                    end=end_time,
                    text=subtitle.text
                )
                
                adjusted_subtitles.append(adjusted_subtitle)
            
            logger.info(f"Adjusted timing for {len(adjusted_subtitles)} subtitles")
            return adjusted_subtitles
            
        except Exception as e:
            error_message = f"Timestamp adjustment failed: {str(e)}"
            logger.error(error_message)
            raise TimestampSyncError(error_message)
    
    def find_sync_parameters(self, reference_subtitles: List[Subtitle], 
                           target_subtitles: List[Subtitle]) -> Tuple[float, float]:
        """
        Find optimal synchronization parameters between two subtitle sets.
        
        Useful for aligning subtitles from different sources.
        
        Args:
            reference_subtitles: Reference subtitle set
            target_subtitles: Target subtitle set to synchronize
            
        Returns:
            Tuple of (offset, scale_factor)
        """
        try:
            logger.info("Finding optimal synchronization parameters")
            
            if not reference_subtitles or not target_subtitles:
                logger.warning("Empty subtitle set, using default sync parameters")
                return (0.0, 1.0)
            
            # Extract timing information
            ref_starts = np.array([s.start for s in reference_subtitles])
            ref_ends = np.array([s.end for s in reference_subtitles])
            target_starts = np.array([s.start for s in target_subtitles])
            target_ends = np.array([s.end for s in target_subtitles])
            
            # Calculate duration ratios
            ref_durations = ref_ends - ref_starts
            target_durations = target_ends - target_starts
            
            # Calculate scale factor based on average duration ratio
            scale_factor = np.mean(ref_durations) / np.mean(target_durations)
            
            # Apply scale factor to target times
            scaled_starts = target_starts * scale_factor
            
            # Calculate offset based on average start time difference
            offset = np.mean(ref_starts) - np.mean(scaled_starts)
            
            logger.info(f"Calculated sync parameters: offset={offset:.2f}s, scale={scale_factor:.2f}")
            return (offset, scale_factor)
            
        except Exception as e:
            error_message = f"Failed to find sync parameters: {str(e)}"
            logger.error(error_message)
            # Return default values
            return (0.0, 1.0)
    
    def detect_silence_boundaries(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect silence boundaries in audio data.
        
        Useful for fine-tuning subtitle timing.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            List of (start_time, end_time) tuples for silent sections
        """
        try:
            import librosa
            
            # Calculate RMS energy
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            rms = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Convert to dB
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Define silence threshold (adjust as needed)
            silence_threshold_db = -40.0
            
            # Identify silent frames
            silent_frames = (rms_db < silence_threshold_db)
            
            # Find silence boundaries
            silence_boundaries = []
            is_silent = False
            silent_start = 0
            
            for i, silent in enumerate(silent_frames):
                if silent and not is_silent:
                    # Start of silence
                    silent_start = i * hop_length / sample_rate
                    is_silent = True
                elif not silent and is_silent:
                    # End of silence
                    silent_end = i * hop_length / sample_rate
                    # Only include if silence is long enough (adjust threshold as needed)
                    if silent_end - silent_start > 0.2:
                        silence_boundaries.append((silent_start, silent_end))
                    is_silent = False
            
            # Handle case where audio ends during silence
            if is_silent:
                silent_end = len(audio_data) / sample_rate
                if silent_end - silent_start > 0.2:
                    silence_boundaries.append((silent_start, silent_end))
            
            logger.info(f"Detected {len(silence_boundaries)} silence boundaries")
            return silence_boundaries
            
        except Exception as e:
            logger.warning(f"Failed to detect silence boundaries: {str(e)}")
            return []
    
    def refine_subtitle_timing(self, subtitles: List[Subtitle], 
                             silence_boundaries: List[Tuple[float, float]]) -> List[Subtitle]:
        """
        Refine subtitle timing based on silence boundaries.
        
        Args:
            subtitles: List of Subtitle objects
            silence_boundaries: List of silence boundaries as (start, end) tuples
            
        Returns:
            List of Subtitle objects with refined timing
        """
        try:
            logger.info("Refining subtitle timing based on silence boundaries")
            
            if not silence_boundaries:
                logger.warning("No silence boundaries provided for timing refinement")
                return subtitles
            
            refined_subtitles = []
            
            for subtitle in subtitles:
                best_start = subtitle.start
                best_end = subtitle.end
                
                # Try to align subtitle start with end of silence
                for silence_start, silence_end in silence_boundaries:
                    # If a silence ends just before subtitle start, align start to silence end
                    if abs(silence_end - subtitle.start) < 0.3:  # 300ms threshold
                        best_start = silence_end
                
                # Try to align subtitle end with start of silence
                for silence_start, silence_end in silence_boundaries:
                    # If a silence starts just after subtitle end, align end to silence start
                    if abs(silence_start - subtitle.end) < 0.3:  # 300ms threshold
                        best_end = silence_start
                
                # Create refined subtitle
                refined_subtitle = Subtitle(
                    index=subtitle.index,
                    start=best_start,
                    end=best_end,
                    text=subtitle.text
                )
                
                refined_subtitles.append(refined_subtitle)
            
            logger.info(f"Refined timing for {len(refined_subtitles)} subtitles")
            return refined_subtitles
            
        except Exception as e:
            error_message = f"Subtitle timing refinement failed: {str(e)}"
            logger.error(error_message)
            # Return original subtitles on error
            return subtitles 