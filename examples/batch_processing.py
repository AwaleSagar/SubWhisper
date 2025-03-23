#!/usr/bin/env python3
"""
SubWhisper - Batch Processing Example

This example demonstrates how to use SubWhisper to process multiple video files in a directory.
"""

import os
import sys
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Tuple

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SubWhisper Batch Processing Example"
    )
    
    parser.add_argument(
        "--input-dir", "-i", 
        required=True,
        help="Input directory containing video files"
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        required=True,
        help="Output directory for subtitle files"
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
        help="Whisper model size for speech recognition (default: base)"
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
        "--max-workers", "-w",
        type=int,
        default=1,
        help="Maximum number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--extensions", "-e",
        default="mp4,avi,mkv,mov,webm,flv",
        help="Comma-separated list of video file extensions to process (default: mp4,avi,mkv,mov,webm,flv)"
    )
    
    return parser.parse_args()

def find_video_files(directory: str, extensions: List[str]) -> List[Path]:
    """
    Find all video files in a directory with the specified extensions.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of paths to video files
    """
    video_files = []
    for ext in extensions:
        video_files.extend(list(Path(directory).glob(f"*.{ext}")))
        video_files.extend(list(Path(directory).glob(f"*.{ext.upper()}")))
    
    return sorted(video_files)

def process_video(video_path: Path, output_dir: Path, config: Config) -> Tuple[bool, str]:
    """
    Process a single video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Output directory for subtitle file
        config: Configuration object
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger = setup_logger(verbose=config.verbose)
    logger.info(f"Processing video: {video_path}")
    
    try:
        # Validate input video
        video_input = VideoInput(str(video_path))
        video_info = video_input.get_info()
        
        # Extract audio
        audio_extractor = AudioExtractor(config)
        audio_path = audio_extractor.extract(str(video_path))
        
        # Process audio
        audio_processor = AudioProcessor(config)
        processed_audio = audio_processor.process(audio_path)
        
        # Identify language if not specified
        if not config.language:
            language_detector = LanguageDetector(config)
            language = language_detector.detect(processed_audio)
            logger.info(f"Detected language: {language}")
        else:
            language = config.language
        
        # Perform speech recognition
        speech_recognizer = SpeechRecognizer(config)
        transcription = speech_recognizer.transcribe(processed_audio, language)
        
        # Generate subtitles
        subtitle_generator = SubtitleGenerator(config)
        subtitles = subtitle_generator.generate(transcription)
        
        # Format and save subtitles
        subtitle_formatter = SubtitleFormatter(config)
        output_file = output_dir / f"{video_path.stem}.{config.format}"
        output_path = subtitle_formatter.format_and_save(
            subtitles, str(output_file), config.format
        )
        
        logger.info(f"Subtitles saved to: {output_path}")
        return True, f"Successfully processed {video_path.name}"
        
    except Exception as e:
        logger.error(f"Error processing {video_path.name}: {str(e)}")
        return False, f"Error processing {video_path.name}: {str(e)}"

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(verbose=args.verbose)
    logger.info("Starting SubWhisper Batch Processing")
    
    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    video_files = find_video_files(args.input_dir, extensions)
    
    if not video_files:
        logger.error(f"No video files found in: {input_dir}")
        return 1
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Create configuration
    class BatchConfig:
        def __init__(self):
            self.input = None  # Set per video
            self.output = None  # Set per video
            self.format = args.format
            self.language = args.language
            self.whisper_model = args.whisper_model
            self.gpu = args.gpu
            self.verbose = args.verbose
            self.temp_dir = None
    
    config = Config(BatchConfig())
    
    # Process videos
    success_count = 0
    failure_count = 0
    
    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit processing tasks
        future_to_video = {
            executor.submit(process_video, video_path, output_dir, config): video_path
            for video_path in video_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                    logger.info(message)
                else:
                    failure_count += 1
                    logger.error(message)
            except Exception as e:
                failure_count += 1
                logger.error(f"Error processing {video_path.name}: {str(e)}")
    
    # Print summary
    logger.info("Batch processing completed")
    logger.info(f"Successfully processed: {success_count} videos")
    logger.info(f"Failed to process: {failure_count} videos")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 