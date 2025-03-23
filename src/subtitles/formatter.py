"""
Subtitle formatting module for SubWhisper.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
import datetime

from src.utils.config import Config
from src.subtitles.generator import Subtitle

logger = logging.getLogger("subwhisper")

class SubtitleFormattingError(Exception):
    """Exception raised for errors in subtitle formatting."""
    pass

class SubtitleFormatter:
    """Subtitle formatting class for different output formats."""
    
    def __init__(self, config: Config):
        """
        Initialize subtitle formatter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Register formatters for different subtitle formats
        self.formatters = {
            "srt": self._format_srt,
            "vtt": self._format_vtt,
            "ass": self._format_ass
        }
    
    def format_and_save(self, subtitles: List[Subtitle], output_path: str, 
                       format_type: Optional[str] = None) -> str:
        """
        Format subtitles and save to file.
        
        Args:
            subtitles: List of Subtitle objects
            output_path: Path to save the formatted subtitles
            format_type: Subtitle format type (srt, vtt, ass)
            
        Returns:
            Path to the saved subtitle file
            
        Raises:
            SubtitleFormattingError: If subtitle formatting fails
        """
        try:
            # Determine format type from file extension if not specified
            if format_type is None:
                _, ext = os.path.splitext(output_path)
                format_type = ext.lstrip('.').lower()
            
            # Ensure format is supported
            if format_type not in self.formatters:
                raise SubtitleFormattingError(f"Unsupported subtitle format: {format_type}")
            
            # Format subtitles
            logger.info(f"Formatting subtitles as {format_type}")
            formatted_content = self.formatters[format_type](subtitles)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            logger.info(f"Subtitles saved to: {output_path}")
            return output_path
            
        except Exception as e:
            error_message = f"Subtitle formatting failed: {str(e)}"
            logger.error(error_message)
            raise SubtitleFormattingError(error_message)
    
    def _format_srt(self, subtitles: List[Subtitle]) -> str:
        """
        Format subtitles in SRT format.
        
        Args:
            subtitles: List of Subtitle objects
            
        Returns:
            Formatted subtitles in SRT format
        """
        lines = []
        
        for subtitle in subtitles:
            # Format index
            lines.append(str(subtitle.index))
            
            # Format timestamp
            start_time = self._format_srt_timestamp(subtitle.start)
            end_time = self._format_srt_timestamp(subtitle.end)
            lines.append(f"{start_time} --> {end_time}")
            
            # Add subtitle text
            lines.append(subtitle.text)
            
            # Add empty line
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for SRT format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp
        """
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _format_vtt(self, subtitles: List[Subtitle]) -> str:
        """
        Format subtitles in WebVTT format.
        
        Args:
            subtitles: List of Subtitle objects
            
        Returns:
            Formatted subtitles in WebVTT format
        """
        lines = ["WEBVTT", ""]
        
        for subtitle in subtitles:
            # Format index (optional in VTT)
            lines.append(f"NOTE {subtitle.index}")
            
            # Format timestamp
            start_time = self._format_vtt_timestamp(subtitle.start)
            end_time = self._format_vtt_timestamp(subtitle.end)
            lines.append(f"{start_time} --> {end_time}")
            
            # Add subtitle text
            lines.append(subtitle.text)
            
            # Add empty line
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for WebVTT format (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp
        """
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"
    
    def _format_ass(self, subtitles: List[Subtitle]) -> str:
        """
        Format subtitles in Advanced SubStation Alpha (ASS) format.
        
        Args:
            subtitles: List of Subtitle objects
            
        Returns:
            Formatted subtitles in ASS format
        """
        # Add script info
        lines = [
            "[Script Info]",
            "Title: Generated by AI Video TTS",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        # Add dialogue lines
        for subtitle in subtitles:
            start_time = self._format_ass_timestamp(subtitle.start)
            end_time = self._format_ass_timestamp(subtitle.end)
            
            # Replace newlines with ASS line breaks
            text = subtitle.text.replace("\n", "\\N")
            
            # Add dialogue line
            lines.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}")
        
        return "\n".join(lines)
    
    def _format_ass_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for ASS format (H:MM:SS.cc).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp
        """
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        centiseconds = int((seconds - int(seconds)) * 100)
        
        return f"{hours}:{minutes:02d}:{int(seconds):02d}.{centiseconds:02d}" 