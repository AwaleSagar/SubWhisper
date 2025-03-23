"""
Speech recognition module for SubWhisper.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

import whisper

from src.utils.config import Config
from src.utils.errors import SpeechRecognitionError, ModelNotFoundError, handle_exception

logger = logging.getLogger("subwhisper")

class Transcription:
    """Class representing a speech transcription."""
    
    def __init__(self, segments: List[Dict[str, Any]], language: str, text: str):
        """
        Initialize a transcription.
        
        Args:
            segments: List of transcription segments
            language: Detected/used language code
            text: Full transcription text
        """
        self.segments = segments
        self.language = language
        self.text = text
    
    def __str__(self) -> str:
        """Return string representation of the transcription."""
        return self.text


class SpeechRecognizer:
    """Speech recognition class using Whisper models."""
    
    def __init__(self, config: Config):
        """
        Initialize speech recognizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        
        # Set device (GPU if available and enabled)
        self.device = "cuda" if torch.cuda.is_available() and self.config.gpu else "cpu"
        logger.info(f"Using device: {self.device}")
    
    @handle_exception
    def _load_model(self, model_name: Optional[str] = None, no_prompt: bool = False) -> bool:
        """
        Load Whisper model.
        
        Args:
            model_name: Model name (tiny, base, small, medium, large)
            no_prompt: If True, skip user confirmation for downloading
            
        Returns:
            True if model was loaded successfully, False otherwise
            
        Raises:
            SpeechRecognitionError: If model loading fails
            ModelNotFoundError: If model is not found and auto-download is disabled or fails
        """
        try:
            # Determine model size from config or argument
            model_size = model_name or self.config.whisper_model
            if not model_size:
                model_size = "base"  # Default model size
            
            logger.info(f"Loading Whisper model: {model_size}")
            
            # Check if we have the model locally
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "whisper")
            model_exists = os.path.exists(os.path.join(model_dir, f"{model_size}.pt")) or \
                          os.path.exists(os.path.join(model_dir, f"{model_size}.en.pt"))
            
            # If model doesn't exist locally and not in no_prompt mode, confirm download
            if not model_exists and not no_prompt:
                print(f"\nWhisper model '{model_size}' is not installed.")
                if model_size == "tiny":
                    print("Size: ~75 MB")
                elif model_size == "base":
                    print("Size: ~150 MB")
                elif model_size == "small":
                    print("Size: ~500 MB")
                elif model_size == "medium":
                    print("Size: ~1.5 GB")
                elif model_size == "large":
                    print("Size: ~3 GB")
                
                user_input = input("Do you want to download it now? (y/n): ")
                
                if user_input.lower() not in ["y", "yes"]:
                    raise ModelNotFoundError(
                        f"Model '{model_size}' not found and download was cancelled",
                        code="MODEL_DOWNLOAD_CANCELLED"
                    )
            
            # Load or download the model
            self.model = whisper.load_model(
                model_size, 
                device=self.device, 
                download_root=model_dir
            )
            
            logger.info(f"Whisper model {model_size} loaded successfully")
            return True
        
        except Exception as e:
            error_message = f"Failed to load Whisper model: {str(e)}"
            logger.error(error_message)
            if no_prompt:
                return False
            raise SpeechRecognitionError(
                error_message,
                code="MODEL_LOAD_FAILED",
                details={"model_name": model_name or self.config.whisper_model}
            )
    
    def load_model(self, model_name: Optional[str] = None, no_prompt: bool = False) -> bool:
        """
        Public wrapper for _load_model.
        
        Args:
            model_name: Optional model name to override config
            no_prompt: If True, skip the user confirmation prompt
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        return self._load_model(model_name, no_prompt)
    
    @handle_exception
    def transcribe(self, audio_path: str, language: Optional[str] = None, no_prompt: bool = False) -> Transcription:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'es')
            no_prompt: If True, skip user confirmation for model download
            
        Returns:
            Transcription object with segments and text
            
        Raises:
            SpeechRecognitionError: If transcription fails
            FileError: If audio file is not found
        """
        try:
            # Load model if not loaded
            if self.model is None:
                success = self._load_model(no_prompt=no_prompt)
                if not success:
                    raise SpeechRecognitionError(
                        "Transcription failed: Could not load model",
                        code="MODEL_LOAD_FAILED"
                    )
            
            # Load audio
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with Whisper
            transcribe_options = {}
            
            # Add language if specified
            if language:
                transcribe_options["language"] = language
            
            # Add additional options from config
            if hasattr(self.config, "transcription_options"):
                transcribe_options.update(self.config.transcription_options)
            
            # Perform transcription
            result = self.model.transcribe(audio_path, **transcribe_options)
            
            # Extract segments and full text
            segments = result.get("segments", [])
            text = result.get("text", "")
            detected_language = result.get("language", language or "")
            
            logger.info(f"Transcription completed: {len(segments)} segments")
            logger.debug(f"Detected language: {detected_language}")
            
            # Create and return Transcription object
            return Transcription(segments, detected_language, text)
            
        except Exception as e:
            if isinstance(e, SpeechRecognitionError):
                # Re-raise existing SpeechRecognitionError
                raise
            
            error_message = f"Transcription failed: {str(e)}"
            logger.error(error_message)
            raise SpeechRecognitionError(
                error_message, 
                code="TRANSCRIPTION_FAILED",
                details={"audio_path": audio_path, "language": language}
            )
    
    @handle_exception
    def transcribe_segments(self, 
                           segments: List[Dict[str, Any]], 
                           language: Optional[str] = None,
                           no_prompt: bool = False) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio segments.
        
        Args:
            segments: List of segment dictionaries with 'path', 'start', 'end'
            language: Language code (e.g., 'en', 'es')
            no_prompt: If True, skip user confirmation for model download
            
        Returns:
            List of transcription segments with adjusted timestamps
            
        Raises:
            SpeechRecognitionError: If segment transcription fails
            FileError: If audio file is not found
        """
        try:
            # Load model if not loaded
            if self.model is None:
                success = self._load_model(no_prompt=no_prompt)
                if not success:
                    raise SpeechRecognitionError(
                        "Segment transcription failed: Could not load model",
                        code="MODEL_LOAD_FAILED"
                    )
            
            results = []
            
            # Process each segment
            for segment in segments:
                # Extract segment info
                segment_path = segment["path"]
                segment_start = segment["start"]
                
                # Transcribe segment
                transcription = self.transcribe(segment_path, language, no_prompt=no_prompt)
                
                # Adjust timestamps for the segment
                for trans_segment in transcription.segments:
                    # Adjust start and end times
                    trans_segment["start"] += segment_start
                    trans_segment["end"] += segment_start
                    
                    # Add to results
                    results.append(trans_segment)
            
            logger.info(f"Transcribed {len(segments)} segments, total {len(results)} transcription segments")
            return results
            
        except Exception as e:
            if isinstance(e, SpeechRecognitionError):
                # Re-raise existing SpeechRecognitionError
                raise
                
            error_message = f"Segment transcription failed: {str(e)}"
            logger.error(error_message)
            raise SpeechRecognitionError(
                error_message,
                code="SEGMENT_TRANSCRIPTION_FAILED"
            ) 