"""
Subtitle generation module for SubWhisper.
"""

import os
import logging
import re
import datetime
from typing import Dict, Any, List, Optional, Union, NamedTuple
import html
from dataclasses import dataclass

from src.utils.config import Config
from src.audio.speech import Transcription

logger = logging.getLogger("subwhisper")

@dataclass
class Subtitle:
    """Subtitle entry with text and timing information."""
    index: int
    start: datetime.timedelta
    end: datetime.timedelta
    text: str

class SubtitleGenerationError(Exception):
    """Exception raised for errors in subtitle generation."""
    pass

class SubtitleGenerator:
    """Subtitle generation class for creating subtitles from transcripts."""
    
    def __init__(self, config: Config):
        """
        Initialize subtitle generator.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def generate(self, transcript) -> List[Subtitle]:
        """
        Generate subtitles from transcript.
        
        Args:
            transcript: Transcript object with segments
            
        Returns:
            List of Subtitle objects
            
        Raises:
            SubtitleGenerationError: If subtitle generation fails
        """
        try:
            subtitles = []
            
            # Process each segment in the transcript
            for idx, segment in enumerate(transcript.segments):
                # Convert start and end times from seconds to timedelta
                start_time = datetime.timedelta(seconds=segment["start"])
                end_time = datetime.timedelta(seconds=segment["end"])
                
                # Create subtitle
                subtitle = Subtitle(
                    index=idx + 1,
                    start=start_time,
                    end=end_time,
                    text=segment["text"]
                )
                
                subtitles.append(subtitle)
            
            logger.info(f"Generated {len(subtitles)} subtitles")
            return subtitles
            
        except Exception as e:
            error_message = f"Subtitle generation failed: {str(e)}"
            logger.error(error_message)
            raise SubtitleGenerationError(error_message)
    
    def _process_segments(self, segments: List[Dict[str, Any]], 
                        max_chars: int = 42, 
                        max_lines: int = 2) -> List[Subtitle]:
        """
        Process transcription segments into subtitles.
        
        Args:
            segments: List of transcription segments
            max_chars: Maximum characters per line
            max_lines: Maximum number of lines per subtitle
            
        Returns:
            List of Subtitle objects
        """
        subtitles = []
        current_index = 1
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            # Skip empty segments
            if not text:
                continue
            
            # Format text for subtitles (line breaks, etc.)
            formatted_text = self._format_text(text, max_chars, max_lines)
            
            # Create subtitle
            subtitle = Subtitle(
                index=current_index,
                start=start_time,
                end=end_time,
                text=formatted_text
            )
            
            subtitles.append(subtitle)
            current_index += 1
        
        return subtitles
    
    def _format_text(self, text: str, max_chars: int = 42, max_lines: int = 2) -> str:
        """
        Format text for subtitles with appropriate line breaks.
        
        Args:
            text: Text to format
            max_chars: Maximum characters per line
            max_lines: Maximum number of lines per subtitle
            
        Returns:
            Formatted text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # If text is short enough, return as is
        if len(text) <= max_chars:
            return text
        
        # Split text into words
        words = text.split()
        
        # Format text with line breaks
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed max_chars
            if len(current_line) + len(word) + 1 <= max_chars:
                # Add word to current line
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                # Start a new line
                lines.append(current_line)
                current_line = word
                
                # Check if we've reached max_lines
                if len(lines) >= max_lines - 1:  # -1 because we still have current_line
                    break
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # If we have more words but reached max_lines, add ellipsis to last line
        if len(lines) == max_lines and words[-1] not in lines[-1].split():
            lines[-1] = lines[-1].strip() + "..."
        
        # Join lines with line breaks
        return "\n".join(lines)
    
    def _adjust_timing(self, subtitles: List[Subtitle], 
                      min_duration: float = 0.5,
                      max_duration: float = 7.0) -> List[Subtitle]:
        """
        Adjust subtitle timing for better readability.
        
        Args:
            subtitles: List of Subtitle objects
            min_duration: Minimum duration for subtitle in seconds
            max_duration: Maximum duration for subtitle in seconds
            
        Returns:
            List of Subtitle objects with adjusted timing
        """
        adjusted_subtitles = []
        
        for i, subtitle in enumerate(subtitles):
            start_time = subtitle.start
            end_time = subtitle.end
            duration = end_time - start_time
            
            # Ensure minimum duration
            if duration < min_duration:
                # Try to extend end time
                if i < len(subtitles) - 1 and end_time + (min_duration - duration) <= subtitles[i+1].start:
                    end_time += (min_duration - duration)
                else:
                    # If we can't extend end time, try to move start time earlier
                    start_time = max(0, start_time - (min_duration - duration))
            
            # Ensure maximum duration
            if end_time - start_time > max_duration:
                end_time = start_time + max_duration
            
            # Create adjusted subtitle
            adjusted_subtitle = Subtitle(
                index=subtitle.index,
                start=start_time,
                end=end_time,
                text=subtitle.text
            )
            
            adjusted_subtitles.append(adjusted_subtitle)
        
        # Ensure no overlap between subtitles
        for i in range(len(adjusted_subtitles) - 1):
            if adjusted_subtitles[i].end > adjusted_subtitles[i+1].start:
                # Adjust end time of current subtitle
                adjusted_subtitles[i].end = adjusted_subtitles[i+1].start - 0.01
        
        return adjusted_subtitles
    
    def _clean_subtitles(self, subtitles: List[Subtitle]) -> List[Subtitle]:
        """
        Clean subtitle text (HTML entities, special characters, etc.)
        
        Args:
            subtitles: List of Subtitle objects
            
        Returns:
            List of Subtitle objects with cleaned text
        """
        cleaned_subtitles = []
        
        for subtitle in subtitles:
            # Unescape HTML entities
            text = html.unescape(subtitle.text)
            
            # Remove special characters
            text = re.sub(r'[\x00-\x1F\x7F]', '', text)
            
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text)
            
            # Trim spaces from start and end of each line
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
            
            # Create cleaned subtitle
            cleaned_subtitle = Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                text=text
            )
            
            cleaned_subtitles.append(cleaned_subtitle)
        
        return cleaned_subtitles 