"""
Configuration handling for SubWhisper.
"""

import os
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """Configuration class for SubWhisper."""
    
    def __init__(self, args):
        """
        Initialize configuration from command line arguments.
        
        Args:
            args: Command line arguments
        """
        self.input_file = args.input
        self.output_file = args.output
        self.subtitle_format = args.format
        self.language = args.language
        self.model_size = args.whisper_model
        self.use_gpu = args.gpu and self._is_gpu_available()
        self.verbose = args.verbose
        self.temp_dir = args.temp_dir or tempfile.gettempdir()
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Model paths
        self.models_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "models"
        ))
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Additional config
        self._load_default_config()
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_default_config(self):
        """Load default configuration values."""
        # Default values
        self.audio_sample_rate = 16000
        self.audio_format = "wav"
        self.supported_video_formats = [
            ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"
        ]
        self.supported_audio_formats = [
            ".wav", ".mp3", ".aac", ".flac", ".ogg"
        ]
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ar": "Arabic",
            "hi": "Hindi"
        }
        
        # Model configurations based on size
        self.model_configs = {
            "tiny": {
                "params": "tiny",
                "beam_size": 2,
                "memory_usage": "low"
            },
            "base": {
                "params": "base",
                "beam_size": 3,
                "memory_usage": "medium"
            },
            "small": {
                "params": "small",
                "beam_size": 3, 
                "memory_usage": "medium"
            },
            "medium": {
                "params": "medium",
                "beam_size": 4,
                "memory_usage": "high"
            },
            "large": {
                "params": "large",
                "beam_size": 5,
                "memory_usage": "very_high"
            }
        }
        
        # Subtitle formatting settings
        self.subtitle_settings = {
            "max_chars_per_line": 42,
            "min_duration": 0.5,  # Minimum duration for subtitle in seconds
            "max_duration": 7.0,  # Maximum duration for subtitle in seconds
            "line_count": 2,      # Maximum number of lines per subtitle
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for the selected model."""
        return self.model_configs.get(self.model_size, self.model_configs["base"])
    
    def get_model_path(self, model_type: str = "whisper") -> str:
        """
        Get the path for a specific model type.
        
        Args:
            model_type: Type of model ("whisper", "langid", etc.)
            
        Returns:
            Path to the model directory
        """
        model_dir = os.path.join(self.models_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def get_temp_file_path(self, prefix: str, suffix: str) -> str:
        """
        Get a temporary file path.
        
        Args:
            prefix: Prefix for the temp file
            suffix: Suffix/extension for the temp file
            
        Returns:
            Path to a temporary file
        """
        return os.path.join(
            self.temp_dir,
            f"{prefix}_{os.urandom(4).hex()}{suffix}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "subtitle_format": self.subtitle_format,
            "language": self.language,
            "model_size": self.model_size,
            "use_gpu": self.use_gpu,
            "verbose": self.verbose,
            "temp_dir": self.temp_dir,
            "models_dir": self.models_dir,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_format": self.audio_format,
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.to_dict(), indent=2) 