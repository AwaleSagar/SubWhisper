"""
Language detection module for SubWhisper.
"""

import os
import logging
import langid
import torch
import whisper
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import librosa

from src.utils.config import Config

logger = logging.getLogger("subwhisper")

class LanguageDetectionError(Exception):
    """Exception raised for errors in language detection."""
    pass

class LanguageDetector:
    """Language detection class."""
    
    def __init__(self, config: Config):
        """
        Initialize language detector.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.whisper_model = None
        self.device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    
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
    
    def _load_whisper_model(self) -> None:
        """
        Load a small Whisper model for language identification.
        
        Raises:
            LanguageDetectionError: If model loading fails
        """
        try:
            # Use tiny model for language detection to save resources
            model_size = "tiny"
            
            # Check if the model exists locally
            if not self._model_exists_locally(model_size):
                logger.info(f"Whisper model '{model_size}' for language detection not found locally")
                
                # Prompt the user for download confirmation
                download_prompt = f"Whisper model '{model_size}' for language detection is not installed. Do you want to download it now? (y/n): "
                user_response = input(download_prompt).strip().lower()
                
                if user_response != 'y' and user_response != 'yes':
                    error_message = f"Model download cancelled. Language detection will be limited."
                    logger.error(error_message)
                    raise LanguageDetectionError(error_message)
                
                logger.info(f"Downloading Whisper model for language detection: {model_size}")
            else:
                logger.info(f"Loading Whisper model for language detection: {model_size}")
            
            # Get the model path
            model_dir = self.config.get_model_path("whisper")
            
            # Load or download the model
            self.whisper_model = whisper.load_model(
                model_size, 
                device=self.device, 
                download_root=model_dir
            )
            
            logger.info(f"Whisper model {model_size} loaded successfully for language detection")
        
        except Exception as e:
            error_message = f"Failed to load Whisper model for language detection: {str(e)}"
            logger.error(error_message)
            raise LanguageDetectionError(error_message)
    
    def detect(self, audio_path: str) -> str:
        """
        Detect language from audio file.
        
        Uses a combination of approaches:
        1. Whisper model for audio-based language detection
        2. langid for text-based validation (after initial transcription)
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Detected language code (ISO 639-1)
            
        Raises:
            LanguageDetectionError: If language detection fails
        """
        try:
            # Load audio
            logger.info(f"Detecting language from audio: {audio_path}")
            
            # First approach: Whisper model (more accurate for speech audio)
            whisper_lang = self._detect_with_whisper(audio_path)
            logger.info(f"Whisper detected language: {whisper_lang}")
            
            # Second approach: langid (text-based validation)
            # Transcribe a short segment and detect language from text
            langid_results = self._detect_with_langid(audio_path)
            logger.info(f"langid detected language: {langid_results[0]} (confidence: {langid_results[1]:.2f})")
            
            # Decision logic:
            # - If whisper is confident and langid agrees, use whisper result
            # - If they disagree, use whisper (it's generally more accurate for speech)
            # - If whisper fails, fall back to langid
            
            if whisper_lang:
                detected_lang = whisper_lang
            else:
                detected_lang = langid_results[0]
            
            # Validate detected language is supported
            if detected_lang not in self.config.supported_languages:
                logger.warning(f"Detected language {detected_lang} is not in supported languages")
                # Fall back to English if not supported
                detected_lang = "en"
            
            language_name = self.config.supported_languages.get(detected_lang, "Unknown")
            logger.info(f"Final detected language: {detected_lang} ({language_name})")
            
            return detected_lang
            
        except Exception as e:
            error_message = f"Language detection failed: {str(e)}"
            logger.error(error_message)
            raise LanguageDetectionError(error_message)
    
    def _detect_with_whisper(self, audio_path: str) -> Optional[str]:
        """
        Detect language using Whisper model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Detected language code or None if detection fails
        """
        try:
            # Load model if not loaded
            if self.whisper_model is None:
                self._load_whisper_model()
            
            # Load audio
            audio = whisper.load_audio(audio_path)
            
            # Take a 30-second sample (or the whole file if shorter)
            audio = whisper.pad_or_trim(audio)
            
            # Get log mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Detect language
            _, probs = self.whisper_model.detect_language(mel)
            
            # Get top language
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            logger.debug(f"Whisper language detection confidence: {confidence:.2f}")
            
            # Return language if confidence is high enough
            if confidence > 0.5:
                return detected_lang
            else:
                logger.warning(f"Low confidence in Whisper language detection: {confidence:.2f}")
                return None
                
        except Exception as e:
            logger.warning(f"Whisper language detection failed: {str(e)}")
            return None
    
    def _detect_with_langid(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect language using langid (text-based approach).
        
        First transcribes a portion of the audio, then detects language from the text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (detected language code, confidence)
        """
        try:
            # Try to get a small transcription to use for text-based detection
            if self.whisper_model is None:
                self._load_whisper_model()
            
            # Load audio and take first 30 seconds
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio[:self.config.audio_sample_rate * 30])
            
            # Get a quick transcription
            result = self.whisper_model.transcribe(
                audio,
                language=None,
                task="transcribe",
                fp16=self.device == "cuda"
            )
            
            # Extract transcription text
            text = result.get("text", "").strip()
            
            # If no text was found, try another approach (sample audio from different parts)
            if not text and len(audio) > self.config.audio_sample_rate * 60:
                # Try the middle of the file
                middle_idx = len(audio) // 2
                middle_audio = audio[middle_idx:middle_idx + self.config.audio_sample_rate * 30]
                middle_audio = whisper.pad_or_trim(middle_audio)
                
                result = self.whisper_model.transcribe(
                    middle_audio,
                    language=None,
                    task="transcribe",
                    fp16=self.device == "cuda"
                )
                
                text = result.get("text", "").strip()
            
            # If we still have no text, fall back to default
            if not text:
                logger.warning("No transcription available for langid language detection")
                return ("en", 0.0)  # Default to English with zero confidence
            
            # Detect language from text
            detected_lang, confidence = langid.classify(text)
            
            return (detected_lang, confidence)
            
        except Exception as e:
            logger.warning(f"langid language detection failed: {str(e)}")
            return ("en", 0.0)  # Default to English with zero confidence 