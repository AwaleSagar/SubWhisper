"""
Video input and validation module for SubWhisper.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

class VideoInputError(Exception):
    """Exception raised for errors in video input."""
    pass

class VideoInput:
    """Video input and validation class."""
    
    def __init__(self, video_path: str):
        """
        Initialize video input and validate file.
        
        Args:
            video_path: Path to the input video file
            
        Raises:
            VideoInputError: If the video file is invalid or not supported
        """
        self.video_path = os.path.abspath(video_path)
        
        # Check if file exists
        if not os.path.isfile(self.video_path):
            raise VideoInputError(f"Video file not found: {self.video_path}")
        
        # Check if file has a valid video extension
        _, ext = os.path.splitext(self.video_path)
        if not ext.lower() in [".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"]:
            raise VideoInputError(f"Unsupported video format: {ext}")
        
        # Validate video file using ffprobe
        try:
            self._validate()
        except Exception as e:
            raise VideoInputError(f"Invalid video file: {str(e)}")
    
    def _validate(self) -> None:
        """
        Validate video file using ffprobe.
        
        Raises:
            VideoInputError: If the video file is invalid or not supported
        """
        try:
            # Run ffprobe command
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "stream=codec_type",
                    "-of", "json",
                    self.video_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse JSON output
            data = json.loads(result.stdout)
            
            # Check if there's at least one video stream
            has_video = False
            has_audio = False
            
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    has_video = True
                elif stream.get("codec_type") == "audio":
                    has_audio = True
            
            if not has_video:
                raise VideoInputError("No video stream found in the file")
                
            if not has_audio:
                raise VideoInputError("No audio stream found in the file")
                
        except subprocess.CalledProcessError as e:
            raise VideoInputError(f"Error validating video: {e.stderr}")
        except json.JSONDecodeError:
            raise VideoInputError("Error parsing video information")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the video file.
        
        Returns:
            Dictionary containing video information
        """
        try:
            # Run ffprobe command to get detailed information
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,sample_rate,channels",
                    "-of", "json",
                    self.video_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse JSON output
            data = json.loads(result.stdout)
            
            # Extract relevant information
            info = {
                "path": self.video_path,
                "filename": os.path.basename(self.video_path),
                "format": os.path.splitext(self.video_path)[1][1:],
                "duration": float(data.get("format", {}).get("duration", 0)),
                "size": int(data.get("format", {}).get("size", 0)),
                "bit_rate": int(data.get("format", {}).get("bit_rate", 0)),
                "streams": []
            }
            
            for stream in data.get("streams", []):
                stream_info = {
                    "codec_type": stream.get("codec_type"),
                    "codec_name": stream.get("codec_name")
                }
                
                # Add video-specific information
                if stream.get("codec_type") == "video":
                    stream_info.update({
                        "width": int(stream.get("width", 0)),
                        "height": int(stream.get("height", 0)),
                    })
                
                # Add audio-specific information
                elif stream.get("codec_type") == "audio":
                    stream_info.update({
                        "sample_rate": int(stream.get("sample_rate", 0)),
                        "channels": int(stream.get("channels", 0)),
                    })
                
                info["streams"].append(stream_info)
            
            return info
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
            # If we can't get detailed info, return basic information
            return {
                "path": self.video_path,
                "filename": os.path.basename(self.video_path),
                "format": os.path.splitext(self.video_path)[1][1:],
                "error": str(e)
            }
    
    def get_duration(self) -> float:
        """
        Get the duration of the video in seconds.
        
        Returns:
            Duration in seconds or 0 if duration cannot be determined
        """
        info = self.get_info()
        return info.get("duration", 0) 