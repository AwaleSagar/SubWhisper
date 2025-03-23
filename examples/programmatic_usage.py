#!/usr/bin/env python3
"""
Example demonstrating how to use the AI Video TTS library programmatically.
"""

import os
import sys
import argparse
from typing import List

# Add the parent directory to the sys.path to import the library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.video.input import VideoInput
from src.video.extraction import AudioExtractor
from src.audio.processing import AudioProcessor
from src.audio.speech import SpeechRecognizer
from src.language.detection import LanguageDetector
from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the example script.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Example of programmatic usage of AI Video TTS")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", required=True, help="Output subtitle file path")
    parser.add_argument("--format", "-f", default="srt", choices=["srt", "vtt", "ass"], 
                        help="Subtitle format (default: srt)")
    parser.add_argument("--language", "-l", help="Force specific language (ISO 639-1 code)")
    parser.add_argument("--model", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Model size (default: base)")
    parser.add_argument("--gpu", "-g", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--temp-dir", help="Directory for temporary files")
    
    return parser.parse_args()


def process_video(config: Config, input_path: str, output_path: str) -> None:
    """Process a video file to generate subtitles.
    
    Args:
        config (Config): Configuration object
        input_path (str): Path to the input video file
        output_path (str): Path to save the output subtitle file
    """
    # Initialize components
    video_input = VideoInput(input_path)
    audio_extractor = AudioExtractor(config)
    audio_processor = AudioProcessor(config)
    language_detector = LanguageDetector(config)
    speech_recognizer = SpeechRecognizer(config)
    subtitle_generator = SubtitleGenerator(config)
    subtitle_formatter = SubtitleFormatter.create(config.args.format, config)
    
    # Step 1: Validate video input
    if not video_input.is_valid():
        print(f"Error: Invalid video file - {input_path}")
        sys.exit(1)
    
    # Step 2: Extract audio from video
    print(f"Extracting audio from {input_path}...")
    audio_path = audio_extractor.extract(input_path)
    
    # Step 3: Process audio
    print("Processing audio...")
    audio_segment = audio_processor.load_audio(audio_path)
    audio_segment = audio_processor.convert_to_mono(audio_segment)
    audio_segment = audio_processor.normalize_audio(audio_segment)
    audio_data = audio_processor.get_numpy_array(audio_segment)
    
    # Step 4: Detect language (if not specified)
    language = config.args.language
    if not language:
        print("Detecting language...")
        language = language_detector.detect(audio_data)
        print(f"Detected language: {language}")
        # Update config with detected language
        config.args.language = language
    
    # Step 5: Transcribe speech
    print(f"Transcribing speech using {config.args.model} model...")
    transcription = speech_recognizer.transcribe(audio_data)
    
    # Step 6: Generate subtitles
    print("Generating subtitles...")
    subtitles = subtitle_generator.generate(transcription)
    
    # Step 7: Format and save subtitles
    formatted_subtitles = subtitle_formatter.format(subtitles)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_subtitles)
    
    print(f"Subtitles successfully saved to {output_path}")
    
    # Cleanup temporary files
    if os.path.exists(audio_path) and os.path.dirname(audio_path) == config.temp_dir:
        os.remove(audio_path)


def batch_process_videos(config: Config, input_files: List[str], output_directory: str) -> None:
    """Process multiple video files and save subtitles to the specified directory.
    
    Args:
        config (Config): Configuration object
        input_files (List[str]): List of input video file paths
        output_directory (str): Directory to save output subtitle files
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for input_file in input_files:
        # Generate output filename based on input filename
        filename = os.path.basename(input_file)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_directory, f"{name}.{config.args.format}")
        
        print(f"\nProcessing {input_file}...")
        try:
            process_video(config, input_file, output_path)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")


def main() -> None:
    """Main entry point for the example script."""
    args = parse_args()
    config = Config(args)
    
    # Single file processing
    process_video(config, args.input, args.output)
    
    # Example of batch processing (uncomment to use)
    """
    input_directory = "/path/to/videos"
    output_directory = "/path/to/subtitles"
    video_files = [
        os.path.join(input_directory, f) 
        for f in os.listdir(input_directory) 
        if f.endswith(('.mp4', '.avi', '.mkv'))
    ]
    batch_process_videos(config, video_files, output_directory)
    """


if __name__ == "__main__":
    main() 