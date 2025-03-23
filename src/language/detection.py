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
        self.device = "cuda" if config.gpu and torch.cuda.is_available() else "cpu"
        
        # Load model during initialization
        self._load_model()
    
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
    
    # Keep _load_whisper_model as an alias for backward compatibility
    _load_whisper_model = _load_model
    
    def _process_audio_sample(self, audio_data: np.ndarray) -> List[Tuple[str, float]]:
        """
        Process audio sample for language detection.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            List of (language code, probability) tuples sorted by probability
        """
        try:
            if self.whisper_model is None:
                self._load_model()
            
            # Pad or trim audio
            audio_sample = whisper.pad_or_trim(audio_data)
            
            # Get log mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_sample).to(self.device)
            
            # Detect language
            _, probs = self.whisper_model.detect_language(mel)
            
            # Convert to list of tuples and sort by probability (descending)
            lang_probs = [(lang, prob) for lang, prob in probs.items()]
            lang_probs.sort(key=lambda x: x[1], reverse=True)
            
            return lang_probs
            
        except Exception as e:
            logger.warning(f"Audio sample processing failed: {str(e)}")
            # Return a default value if processing failed
            return [("en", 1.0)]
    
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
            
            # Process audio with our helper method
            try:
                # Load audio with librosa since this is more reliable than whisper.load_audio
                audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
                
                # Process using our dedicated method
                lang_probs = self._process_audio_sample(audio_data)
                
                # Extract top language
                whisper_lang = lang_probs[0][0] if lang_probs else None
                logger.info(f"Whisper detected language: {whisper_lang}")
            except Exception as e:
                logger.warning(f"Whisper language detection failed: {str(e)}")
                whisper_lang = None
            
            # Second approach: langid (text-based validation)
            try:
                # Load audio and extract a segment for transcription
                audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
                
                # Limit to first 30 seconds
                audio_data = audio_data[:16000 * 30]
                
                # Use langid to detect language from transcribed text
                if self.whisper_model:
                    # Transcribe a short segment
                    result = self.whisper_model.transcribe(
                        audio_data,
                        language=None,
                        fp16=self.device == "cuda"
                    )
                    text = result.get("text", "").strip()
                    
                    # Detect language from text
                    if text:
                        langid_tag, langid_confidence = langid.classify(text)
                        logger.info(f"langid detected language: {langid_tag} (confidence: {langid_confidence:.2f})")
                    else:
                        langid_tag, langid_confidence = None, 0.0
                else:
                    langid_tag, langid_confidence = None, 0.0
            except Exception as e:
                logger.warning(f"langid language detection failed: {str(e)}")
                langid_tag, langid_confidence = None, 0.0
            
            # Decision logic:
            # - If whisper is confident, use whisper result
            # - If whisper fails, fall back to langid
            # - If both fail, fall back to English
            
            if whisper_lang:
                detected_lang = whisper_lang
            elif langid_tag:
                detected_lang = langid_tag
            else:
                detected_lang = "en"  # Default to English as fallback
            
            # Validate detected language is supported
            # Check if we have supported_languages defined, if not just return the detection
            if hasattr(self.config, 'supported_languages') and self.config.supported_languages:
                if detected_lang not in self.config.supported_languages:
                    logger.warning(f"Detected language {detected_lang} is not in supported languages")
                    # Fall back to English if not supported
                    detected_lang = "en"
                
                language_name = self.config.supported_languages.get(detected_lang, "Unknown")
                logger.info(f"Final detected language: {detected_lang} ({language_name})")
            else:
                logger.info(f"Final detected language: {detected_lang}")
            
            return detected_lang
            
        except Exception as e:
            error_message = f"Language detection failed: {str(e)}"
            logger.error(error_message)
            raise LanguageDetectionError(error_message) 