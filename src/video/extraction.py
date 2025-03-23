"""
Audio extraction module for SubWhisper.
"""

import os
import subprocess
import logging
from typing import Optional

from src.utils.config import Config

logger = logging.getLogger("subwhisper")

class AudioExtractionError(Exception):
    """Exception raised for errors in audio extraction."""
    pass

class AudioExtractor:
    """Audio extraction class."""
    
    def __init__(self, config: Config):
        """
        Initialize audio extractor.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def extract(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the extracted audio file (optional)
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            AudioExtractionError: If audio extraction fails
        """
        # If output path is not provided, create a temporary file
        if output_path is None:
            output_path = self.config.get_temp_file_path(
                "audio", f".{self.config.audio_format}"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Log extraction process
        logger.info(f"Extracting audio from '{video_path}' to '{output_path}'")
        
        try:
            # Run ffmpeg command to extract audio
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le" if self.config.audio_format == "wav" else "libmp3lame",
                "-ar", str(self.config.audio_sample_rate),
                "-ac", "1",  # Mono
                "-y",  # Overwrite output files
                output_path
            ]
            
            logger.debug(f"Running command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if output file exists and has a non-zero size
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise AudioExtractionError("Audio extraction failed: Output file is empty or does not exist")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_message = f"Audio extraction failed: {e.stderr}"
            logger.error(error_message)
            raise AudioExtractionError(error_message)
    
    def extract_segment(self, video_path: str, start_time: float, 
                        duration: Optional[float] = None, 
                        output_path: Optional[str] = None) -> str:
        """
        Extract a segment of audio from a video file.
        
        Args:
            video_path: Path to the input video file
            start_time: Start time in seconds
            duration: Duration in seconds (optional)
            output_path: Path to save the extracted audio file (optional)
            
        Returns:
            Path to the extracted audio segment
            
        Raises:
            AudioExtractionError: If audio extraction fails
        """
        # If output path is not provided, create a temporary file
        if output_path is None:
            output_path = self.config.get_temp_file_path(
                f"audio_segment_{start_time}", f".{self.config.audio_format}"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Log extraction process
        time_info = f"from {start_time}s"
        if duration is not None:
            time_info += f" for {duration}s"
        
        logger.info(f"Extracting audio segment {time_info} from '{video_path}' to '{output_path}'")
        
        try:
            # Build ffmpeg command
            command = [
                "ffmpeg",
                "-i", video_path,
                "-ss", str(start_time),  # Start time
            ]
            
            # Add duration if provided
            if duration is not None:
                command.extend(["-t", str(duration)])
            
            # Add output options
            command.extend([
                "-vn",  # No video
                "-acodec", "pcm_s16le" if self.config.audio_format == "wav" else "libmp3lame",
                "-ar", str(self.config.audio_sample_rate),
                "-ac", "1",  # Mono
                "-y",  # Overwrite output files
                output_path
            ])
            
            logger.debug(f"Running command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if output file exists and has a non-zero size
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise AudioExtractionError("Audio extraction failed: Output file is empty or does not exist")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_message = f"Audio segment extraction failed: {e.stderr}"
            logger.error(error_message)
            raise AudioExtractionError(error_message) 