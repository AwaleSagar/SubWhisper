#!/usr/bin/env python3
"""
SubWhisper - Audio Extraction and Automatic Subtitling Tool
Main application entry point
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import project modules
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.video.input import VideoInput
from src.video.extraction import AudioExtractor
from src.audio.processing import AudioProcessor
from src.audio.speech import SpeechRecognizer
from src.language.detection import LanguageDetector
from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter

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
        choices=["srt", "vtt", "ass"],
        default="srt",
        help="Subtitle format (default: srt)"
    )
    
    parser.add_argument(
        "--language", "-l", 
        help="Force specific language (ISO 639-1 code, e.g., 'en' for English)"
    )
    
    parser.add_argument(
        "--whisper-model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size for speech recognition (default: base). Larger models are more accurate but slower and require more memory."
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

def main():
    """Main application function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(verbose=args.verbose)
    logger.info("Starting SubWhisper")
    
    # Initialize configuration
    config = Config(args)
    
    try:
        # Validate input video
        logger.info(f"Processing input video: {args.input}")
        video_input = VideoInput(args.input)
        video_info = video_input.get_info()
        logger.info(f"Video info: {video_info}")
        
        # Extract audio
        logger.info("Extracting audio from video")
        audio_extractor = AudioExtractor(config)
        audio_path = audio_extractor.extract(args.input)
        logger.info(f"Audio extracted to: {audio_path}")
        
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
        
        # Perform speech recognition
        logger.info(f"Performing speech recognition with {args.whisper_model} model")
        speech_recognizer = SpeechRecognizer(config)
        transcription = speech_recognizer.transcribe(processed_audio, language)
        
        # Generate subtitles
        logger.info("Generating subtitles")
        subtitle_generator = SubtitleGenerator(config)
        subtitles = subtitle_generator.generate(transcription)
        
        # Format and save subtitles
        logger.info(f"Formatting subtitles as {args.format}")
        subtitle_formatter = SubtitleFormatter(config)
        output_path = subtitle_formatter.format_and_save(
            subtitles, args.output, args.format
        )
        
        logger.info(f"Subtitles successfully saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 