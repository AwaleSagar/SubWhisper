"""
Centralized error handling for SubWhisper.

This module provides a standardized exception hierarchy for all SubWhisper components.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("subwhisper")

class SubWhisperError(Exception):
    """Base exception for all SubWhisper errors."""
    
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
        """
        Initialize with error message, optional code and details.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code
            details: Additional error details for debugging
        """
        self.message = message
        self.code = code
        self.details = details or {}
        
        # Log the error
        log_message = f"{code + ': ' if code else ''}{message}"
        if details:
            log_message += f" - Details: {details}"
        logger.error(log_message)
        
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.code:
            return f"{self.code}: {self.message}"
        return self.message


# Input/Output errors
class FileError(SubWhisperError):
    """Error related to file operations."""
    pass

class VideoError(SubWhisperError):
    """Error related to video processing."""
    pass

class AudioError(SubWhisperError):
    """Error related to audio processing."""
    pass


# Processing errors
class ProcessingError(SubWhisperError):
    """Error related to data processing."""
    pass

class VideoProcessingError(ProcessingError):
    """Error related to video processing."""
    pass

class AudioProcessingError(ProcessingError):
    """Error related to audio processing."""
    pass

class AudioExtractionError(AudioProcessingError):
    """Error related to extracting audio from video."""
    pass

class SpeechRecognitionError(ProcessingError):
    """Error related to speech recognition."""
    pass

class LanguageDetectionError(ProcessingError):
    """Error related to language detection."""
    pass

class SubtitleGenerationError(ProcessingError):
    """Error related to subtitle generation."""
    pass

class SubtitleFormattingError(ProcessingError):
    """Error related to subtitle formatting."""
    pass


# Model errors
class ModelError(SubWhisperError):
    """Error related to ML models."""
    pass

class ModelDownloadError(ModelError):
    """Error related to downloading models."""
    pass

class ModelLoadError(ModelError):
    """Error related to loading models."""
    pass

class ModelNotFoundError(ModelError):
    """Error when a required model is not found."""
    pass


# Configuration errors
class ConfigError(SubWhisperError):
    """Error related to configuration."""
    pass

class DependencyError(SubWhisperError):
    """Error related to missing dependencies."""
    pass


def handle_exception(func):
    """
    Decorator for standardized exception handling.
    
    This decorator catches exceptions and wraps them in appropriate SubWhisperError types.
    It also ensures proper logging of exceptions.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function with standardized exception handling
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SubWhisperError:
            # Already a SubWhisperError, just re-raise
            raise
        except FileNotFoundError as e:
            raise FileError(f"File not found: {str(e)}", code="FILE_NOT_FOUND")
        except PermissionError as e:
            raise FileError(f"Permission denied: {str(e)}", code="PERMISSION_DENIED")
        except IsADirectoryError as e:
            raise FileError(f"Is a directory: {str(e)}", code="IS_DIRECTORY")
        except ImportError as e:
            raise DependencyError(f"Missing dependency: {str(e)}", code="DEPENDENCY_MISSING")
        except Exception as e:
            # Generic fallback
            raise SubWhisperError(f"Unexpected error: {str(e)}", code="UNEXPECTED_ERROR")
    
    return wrapper 