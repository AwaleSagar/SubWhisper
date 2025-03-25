#!/usr/bin/env python3
"""
SubWhisper - Audio Extraction and Automatic Subtitling Tool
Main application entry point
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import project modules
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.video.input import VideoInput, VideoInputError
from src.video.extraction import AudioExtractor, AudioExtractionError
from src.audio.processing import AudioProcessor
from src.audio.speech import SpeechRecognizer
from src.language.detection import LanguageDetector
from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter
from src.utils.model_manager import check_model_exists, load_model

# Define supported subtitle formats
SUPPORTED_SUBTITLE_FORMATS = ["srt", "vtt", "ass"]

# Define supported Whisper models
SUPPORTED_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Define supported language codes (ISO 639-1)
SUPPORTED_LANGUAGE_CODES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar", "hi", 
    "tr", "pl", "sv", "da", "no", "fi", "hu", "cs", "el", "bg", "ro", "sk", "uk", 
    "hr", "sr", "sl", "et", "lt", "lv", "he", "th", "vi", "id", "ms", "fa", "ur"
]

# Define language display names for better user experience
LANGUAGE_DISPLAY_NAMES = {
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
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "hu": "Hungarian",
    "cs": "Czech",
    "el": "Greek",
    "bg": "Bulgarian",
    "ro": "Romanian",
    "sk": "Slovak",
    "uk": "Ukrainian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "fa": "Persian",
    "ur": "Urdu"
}

def validate_file_path(file_path: str, must_exist: bool = True) -> Optional[str]:
    """
    Validate a file path.
    
    Args:
        file_path: The file path to validate
        must_exist: If True, the file must exist
        
    Returns:
        Error message if validation fails, None otherwise
    """
    if not file_path:
        return "File path cannot be empty"
    
    path = Path(file_path)
    
    if must_exist and not path.exists():
        return f"File does not exist: {file_path}"
    
    if must_exist and not path.is_file():
        return f"Path is not a file: {file_path}"
    
    return None

def validate_output_file(file_path: str, format: str) -> Optional[str]:
    """
    Validate an output file path.
    
    Args:
        file_path: The file path to validate
        format: The expected file format
        
    Returns:
        Error message if validation fails, None otherwise
    """
    if not file_path:
        return "Output file path cannot be empty"
    
    path = Path(file_path)
    
    # Check if parent directory exists
    if not path.parent.exists():
        return f"Output directory does not exist: {path.parent}"
    
    # Check if file extension matches the format
    if path.suffix.lower() != f".{format}":
        return f"Output file extension does not match the specified format (expected .{format}): {file_path}"
    
    return None

def validate_directory(dir_path: str, must_exist: bool = False) -> Optional[str]:
    """
    Validate a directory path.
    
    Args:
        dir_path: The directory path to validate
        must_exist: If True, the directory must exist
        
    Returns:
        Error message if validation fails, None otherwise
    """
    if not dir_path:
        return "Directory path cannot be empty"
    
    path = Path(dir_path)
    
    if must_exist and not path.exists():
        return f"Directory does not exist: {dir_path}"
    
    if must_exist and not path.is_dir():
        return f"Path is not a directory: {dir_path}"
    
    return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SubWhisper - Audio Extraction and Automatic Subtitling Tool"
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input video file path"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output subtitle file path"
    )
    
    parser.add_argument(
        "--format", "-f", 
        choices=SUPPORTED_SUBTITLE_FORMATS,
        default="srt",
        help=f"Subtitle format (choices: {', '.join(SUPPORTED_SUBTITLE_FORMATS)}, default: srt)"
    )
    
    parser.add_argument(
        "--language", "-l", 
        help=f"Force specific language (ISO 639-1 code, e.g., 'en' for English). Supported codes: {', '.join(SUPPORTED_LANGUAGE_CODES[:10])} and more."
    )
    
    parser.add_argument(
        "--whisper-model", "-m",
        choices=SUPPORTED_WHISPER_MODELS,
        default="base",
        help=f"Whisper model size for speech recognition (choices: {', '.join(SUPPORTED_WHISPER_MODELS)}, default: base). Larger models are more accurate but slower and require more memory."
    )
    
    parser.add_argument(
        "--translate-to-english", "-t",
        choices=["auto", "always", "never"],
        default="never",
        help="Translate non-English subtitles to English: 'auto' (ask when detected), 'always' (always translate), 'never' (don't translate, default)"
    )
    
    parser.add_argument(
        "--translation-model-dir",
        help="Directory to store downloaded translation models (default: models/translation)"
    )
    
    parser.add_argument(
        "--gpu", "-g",
        action="store_true",
        help="Use GPU acceleration if available"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary files (default: system temp dir)"
    )
    
    return parser.parse_args()

def validate_arguments(args) -> List[str]:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of error messages (empty if validation succeeds)
    """
    errors = []
    
    # Validate input file
    error = validate_file_path(args.input, must_exist=True)
    if error:
        errors.append(f"Input error: {error}")
    
    # Validate output file and format
    error = validate_output_file(args.output, args.format)
    if error:
        errors.append(f"Output error: {error}")
    
    # Validate language code if specified
    if args.language and args.language not in SUPPORTED_LANGUAGE_CODES:
        errors.append(f"Unsupported language code: {args.language}. Supported codes: {', '.join(SUPPORTED_LANGUAGE_CODES[:10])}...")
    
    # Validate temp directory if specified
    if args.temp_dir:
        error = validate_directory(args.temp_dir, must_exist=True)
        if error:
            errors.append(f"Temporary directory error: {error}")
    
    return errors

def main():
    """Main application function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(verbose=args.verbose)
    logger.info("Starting SubWhisper")
    
    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        for error in validation_errors:
            logger.error(error)
        print("\nPlease fix the errors and try again.")
        return 1
    
    # Initialize configuration
    config = Config(args)
    
    try:
        # Validate input video
        logger.info(f"Processing input video: {args.input}")
        try:
            video_input = VideoInput(args.input)
            video_info = video_input.get_info()
            logger.info(f"Video info: {video_info}")
        except VideoInputError as e:
            logger.error(f"Video input error: {str(e)}")
            return 1
        
        # Extract audio
        logger.info("Extracting audio from video")
        try:
            audio_extractor = AudioExtractor(config)
            audio_path = audio_extractor.extract(args.input)
            logger.info(f"Audio extracted to: {audio_path}")
        except AudioExtractionError as e:
            logger.error(f"Audio extraction error: {str(e)}")
            return 1
        
        # Process audio
        logger.info("Processing audio")
        audio_processor = AudioProcessor(config)
        processed_audio = audio_processor.process(audio_path)
        
        # Identify language if not specified
        if not args.language:
            logger.info("Detecting language")
            language_detector = LanguageDetector(config)
            detected_language = language_detector.detect(processed_audio)
            logger.info(f"Detected language: {detected_language}")
            language = detected_language
        else:
            logger.info(f"Using specified language: {args.language}")
            language = args.language
        
        # Check if translation is needed
        translate_subtitles = False
        if args.translate_to_english != "never" and language != "en":
            if args.translate_to_english == "always":
                translate_subtitles = True
                logger.info(f"Will translate subtitles from {language} to English")
            elif args.translate_to_english == "auto":
                # Get language name for better user experience
                language_name = LANGUAGE_DISPLAY_NAMES.get(language, language)
                
                # Prompt user for translation preference
                translate_prompt = f"Detected language is {language_name} ({language}). Do you want to translate subtitles to English? (y/n): "
                user_response = input(translate_prompt).strip().lower()
                
                if user_response == 'y' or user_response == 'yes':
                    translate_subtitles = True
                    logger.info(f"Will translate subtitles from {language} to English")
                else:
                    logger.info("Skipping translation as per user choice")
        
        # Perform speech recognition
        logger.info(f"Performing speech recognition with {args.whisper_model} model")
        speech_recognizer = SpeechRecognizer(config)
        transcription = speech_recognizer.transcribe(processed_audio, language)
        
        # Generate subtitles
        logger.info("Generating subtitles")
        subtitle_generator = SubtitleGenerator(config)
        subtitles = subtitle_generator.generate(transcription)
        
        # Translate subtitles if needed
        if translate_subtitles:
            logger.info(f"Translating subtitles from {language} to English")
            from src.language.translation import Translator
            translator = Translator(config)
            subtitles = translator.translate_subtitles(subtitles, language)
            # Update language to English after translation
            language = "en"
        
        # Format and save subtitles
        logger.info(f"Formatting subtitles as {args.format}")
        subtitle_formatter = SubtitleFormatter(config)
        output_path = subtitle_formatter.format_and_save(
            subtitles, args.output, args.format
        )
        
        logger.info(f"Subtitles successfully saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 