"""
Speech recognition module for SubWhisper.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
import whisper

from src.utils.config import Config

logger = logging.getLogger("subwhisper")

class SpeechRecognitionError(Exception):
    """Exception raised for errors in speech recognition."""
    pass

class Transcription:
    """Class representing a transcription with timestamps."""
    
    def __init__(self, segments: List[Dict[str, Any]]):
        """
        Initialize transcription with segments.
        
        Args:
            segments: List of segment dictionaries with text and timestamps
        """
        self.segments = segments
    
    def __repr__(self) -> str:
        """String representation of the transcription."""
        return f"Transcription with {len(self.segments)} segments"

class SpeechRecognizer:
    """Speech recognition class using Whisper models."""
    
    def __init__(self, config: Config):
        """
        Initialize speech recognizer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model = None
        self.device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} for speech recognition")
    
    def _model_exists_locally(self, model_size: str) -> bool:
        """
        Check if the specified Whisper model exists locally.
        
        Args:
            model_size: Size of the Whisper model to check
            
        Returns:
            True if the model exists locally, False otherwise
        """
        try:
            # Get the expected model path
            model_dir = self.config.get_model_path("whisper")
            
            # Whisper models are stored in the format "model_size.pt"
            model_path = os.path.join(model_dir, f"{model_size}.pt")
            
            # Check if the model file exists
            return os.path.exists(model_path)
        except Exception as e:
            logger.debug(f"Error checking for local model: {str(e)}")
            return False
    
    def _load_model(self) -> None:
        """
        Load the Whisper model based on configuration.
        
        Raises:
            SpeechRecognitionError: If model loading fails
        """
        try:
            model_size = self.config.model_size
            
            # Check if the model exists locally
            if not self._model_exists_locally(model_size):
                logger.info(f"Whisper model '{model_size}' not found locally")
                
                # Prompt the user for download confirmation
                download_prompt = f"Whisper model '{model_size}' is not installed. Do you want to download it now? (y/n): "
                user_response = input(download_prompt).strip().lower()
                
                if user_response != 'y' and user_response != 'yes':
                    error_message = f"Model download cancelled. Please choose a different model size or download manually."
                    logger.error(error_message)
                    raise SpeechRecognitionError(error_message)
                
                logger.info(f"Downloading Whisper model: {model_size}")
            else:
                logger.info(f"Loading Whisper model: {model_size}")
            
            # Get the model path
            model_dir = self.config.get_model_path("whisper")
            
            # Load or download the model
            self.model = whisper.load_model(
                model_size, 
                device=self.device, 
                download_root=model_dir
            )
            
            logger.info(f"Whisper model {model_size} loaded successfully")
        
        except Exception as e:
            error_message = f"Failed to load Whisper model: {str(e)}"
            logger.error(error_message)
            raise SpeechRecognitionError(error_message)
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Transcription:
        """
        Transcribe audio file to text with timestamps.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (if known)
            
        Returns:
            Transcription object with segments
            
        Raises:
            SpeechRecognitionError: If transcription fails
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()
            
            # Set transcription options
            options = {
                "beam_size": self.config.get_model_config().get("beam_size", 3),
                "best_of": 5,
                "temperature": 0,  # Use greedy decoding
                "fp16": self.device == "cuda",  # Use fp16 on GPU
                "verbose": self.config.verbose
            }
            
            # Add language if specified
            if language:
                options["language"] = language
            
            # Perform transcription
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                **options
            )
            
            # Create Transcription object
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "id": segment.get("id"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text").strip(),
                    "confidence": segment.get("confidence", 0.0)
                })
            
            logger.info(f"Transcription completed with {len(segments)} segments")
            return Transcription(segments)
        
        except Exception as e:
            error_message = f"Transcription failed: {str(e)}"
            logger.error(error_message)
            raise SpeechRecognitionError(error_message)
    
    def transcribe_segments(self, segment_paths: List[str], language: Optional[str] = None) -> Transcription:
        """
        Transcribe multiple audio segments and combine the results.
        
        Args:
            segment_paths: List of paths to audio segment files
            language: Language code (if known)
            
        Returns:
            Combined Transcription object with all segments
            
        Raises:
            SpeechRecognitionError: If transcription fails
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()
            
            # Process each segment
            all_segments = []
            base_id = 0
            
            for i, segment_path in enumerate(segment_paths):
                logger.info(f"Transcribing segment {i+1}/{len(segment_paths)}: {segment_path}")
                
                # Extract start time from filename
                # Format is segment_{i}_{start}_{end}.{format}
                filename = os.path.basename(segment_path)
                parts = filename.split("_")
                if len(parts) >= 4:
                    segment_start_time = float(parts[2])
                else:
                    segment_start_time = 0.0
                
                # Transcribe segment
                segment_transcription = self.transcribe(segment_path, language)
                
                # Adjust timestamps and append segments
                for segment in segment_transcription.segments:
                    segment["id"] = base_id + segment.get("id", 0)
                    segment["start"] += segment_start_time
                    segment["end"] += segment_start_time
                    all_segments.append(segment)
                
                # Update base ID for next segment
                base_id = len(all_segments)
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x.get("start", 0))
            
            logger.info(f"Combined transcription completed with {len(all_segments)} segments")
            return Transcription(all_segments)
            
        except Exception as e:
            error_message = f"Segment transcription failed: {str(e)}"
            logger.error(error_message)
            raise SpeechRecognitionError(error_message) 