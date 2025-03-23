# SubWhisper - Usage Guide

This document explains how to use the SubWhisper tool for extracting audio and generating subtitles from videos.

## Installation

Before using the tool, you need to install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/AwaleSagar/SubWhisper
cd subwhisper

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

## Basic Usage

The simplest way to use the tool is to provide an input video file and an output subtitle file:

```bash
python main.py --input video.mp4 --output subtitles.srt
```

This will:
1. Extract audio from the video
2. Automatically detect the spoken language
3. Transcribe the speech
4. Generate subtitles in SRT format

## Command Line Options

The tool supports various command line options:

```
--input, -i          Input video file path (required)
--output, -o         Output subtitle file path (required)
--format, -f         Subtitle format: srt, vtt, or ass (default: srt)
--language, -l       Force specific language (ISO 639-1 code, e.g., 'en' for English)
--whisper-model, -m  Whisper model size: tiny, base, small, medium, or large (default: base)
                     Larger models are more accurate but slower and require more memory
--gpu, -g            Use GPU acceleration if available
--verbose, -v        Enable verbose logging
--temp-dir           Directory for temporary files (default: system temp dir)
```

## Examples

### Generate Subtitles with Known Language

If you know the language of the video, you can specify it to improve accuracy:

```bash
python main.py --input spanish_video.mp4 --output subtitles.srt --language es
```

### Generate WebVTT Format

To generate subtitles in WebVTT format (useful for web videos):

```bash
python main.py --input video.mp4 --output subtitles.vtt --format vtt
```

### Use Larger Model for Better Accuracy

For better accuracy (at the cost of speed), use a larger Whisper model:

```bash
python main.py --input video.mp4 --output subtitles.srt --whisper-model large
```

### Use Smaller Model for Faster Processing

For faster processing (with lower accuracy), use a smaller Whisper model:

```bash
python main.py --input video.mp4 --output subtitles.srt --whisper-model tiny
```

### Use GPU Acceleration

If you have a compatible GPU, you can speed up processing:

```bash
python main.py --input video.mp4 --output subtitles.srt --whisper-model medium --gpu
```

## Batch Processing

For processing multiple video files at once, you can use the batch processing example:

```bash
python examples/batch_processing.py --input-dir /path/to/videos --output-dir /path/to/subtitles
```

The batch processing script supports the following arguments:

```
--input-dir, -i      Input directory containing video files (required)
--output-dir, -o     Output directory for subtitle files (required)
--format, -f         Subtitle format: srt, vtt, or ass (default: srt)
--language, -l       Force specific language (ISO 639-1 code)
--whisper-model, -m  Whisper model size: tiny, base, small, medium, or large (default: base)
--gpu, -g            Use GPU acceleration if available
--verbose, -v        Enable verbose logging
--max-workers, -w    Maximum number of worker processes (default: 1)
--extensions, -e     Comma-separated list of video file extensions to process 
                     (default: mp4,avi,mkv,mov,webm,flv)
```

To process all .mp4 and .mkv files in a directory using the tiny model with 4 worker threads:

```bash
python examples/batch_processing.py --input-dir videos/ --output-dir subtitles/ --format srt --whisper-model tiny --extensions mp4,mkv --max-workers 4
```

## Processing Long Videos

For long videos, the tool will automatically process the audio in segments to manage memory usage. However, this might take considerable time depending on your hardware.

For very long videos, consider breaking them into smaller parts before processing:

```bash
# Extract a 10-minute segment starting at 5 minutes
ffmpeg -i long_video.mp4 -ss 00:05:00 -t 00:10:00 -c copy segment.mp4

# Process the segment
python main.py --input segment.mp4 --output segment_subtitles.srt
```

## Non-interactive Mode

When using SubWhisper in scripts or automated environments, you can use the API with the `no_prompt` parameter to prevent interactive prompts for model downloads:

```python
from src.utils.model_manager import load_model

# Load model without prompting for confirmation
success, model, message = load_model("whisper", "tiny", auto_download=True, no_prompt=True)
```

Similarly, when using the Speech Recognition API:

```python
from src.audio.speech import SpeechRecognizer

# Initialize the recognizer
recognizer = SpeechRecognizer(config)

# Transcribe with no prompts
transcription = recognizer.transcribe("audio.wav", language="en", no_prompt=True)
```

## Troubleshooting

### Language Detection Issues

If the language detection is incorrect, try specifying the language explicitly:

```bash
python main.py --input video.mp4 --output subtitles.srt --language en
```

### Memory Issues

If you encounter memory issues, try:

1. Using a smaller model: `--whisper-model tiny` or `--whisper-model base`
2. Processing shorter video segments
3. Ensuring you have enough free disk space for temporary files

### Poor Subtitle Timing

If subtitle timing is off, it might be due to:

1. Background noise in the video
2. Multiple speakers talking over each other
3. Music or sound effects

In these cases, you might need to manually adjust the subtitle timing using a subtitle editor like Aegisub or Subtitle Edit.

### Model Downloads

When using the tool for the first time with a specific Whisper model size, the tool will check if the model is already downloaded:

1. If the model is already downloaded, it will be loaded directly.
2. If the model is not found, you'll be prompted to download it:
   ```
   Whisper model 'medium' is not installed. Do you want to download it now? (y/n):
   ```
3. Enter 'y' or 'yes' to download the model (this might take some time depending on your internet connection and the model size).
4. Enter 'n' or 'no' to cancel and try a different model size.

You can skip the prompt by using the API with the `no_prompt` parameter as described in the Non-interactive Mode section.

Model sizes and approximate download sizes:
- tiny: ~75 MB
- base: ~150 MB
- small: ~500 MB
- medium: ~1.5 GB
- large: ~3 GB

The models are downloaded once and stored in the `models/whisper` directory for future use.

## Advanced Usage

For advanced users, you can import the modules in your own Python scripts:

```python
from src.video.input import VideoInput
from src.video.extraction import AudioExtractor
from src.audio.processing import AudioProcessor
from src.audio.speech import SpeechRecognizer
from src.language.detection import LanguageDetector
from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter
from src.utils.config import Config

# Create a custom configuration
class CustomArgs:
    def __init__(self):
        self.input = "video.mp4"
        self.output = "subtitles.srt"
        self.format = "srt"
        self.language = "en"
        self.whisper_model = "base"
        self.gpu = True
        self.verbose = True
        self.temp_dir = None

config = Config(CustomArgs())

# Use the components as needed
video_input = VideoInput("video.mp4")
audio_extractor = AudioExtractor(config)
audio_path = audio_extractor.extract("video.mp4")
# ... and so on
```

Check the `examples/programmatic_usage.py` file for a more complete example of using SubWhisper as a library. 