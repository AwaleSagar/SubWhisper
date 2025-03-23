# SubWhisper API Documentation

This document provides detailed information about the SubWhisper API for developers who want to use SubWhisper as a library in their own projects.

## Module Structure

The SubWhisper API is organized into the following modules:

- `video`: Video input and audio extraction
- `audio`: Audio processing and speech recognition
- `language`: Language detection
- `subtitles`: Subtitle generation and formatting
- `utils`: Configuration, logging, and model management

## Configuration

```python
from src.utils.config import Config

# Create configuration from command line arguments
config = Config(args)

# Or create configuration manually
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
```

## Video Processing

### Video Input

```python
from src.video.input import VideoInput

# Initialize video input
video_input = VideoInput("path/to/video.mp4")

# Get video information
video_info = video_input.get_info()
# Returns: {"duration": 120.5, "width": 1920, "height": 1080, "fps": 30, ...}

# Validate video file
video_input.validate()  # Raises VideoInputError if invalid
```

### Audio Extraction

```python
from src.video.extraction import AudioExtractor

# Initialize audio extractor
audio_extractor = AudioExtractor(config)

# Extract audio from video
audio_path = audio_extractor.extract("path/to/video.mp4")
# Returns: Path to extracted audio file
```

## Audio Processing

### Audio Processing

```python
from src.audio.processing import AudioProcessor

# Initialize audio processor
audio_processor = AudioProcessor(config)

# Process audio file
processed_audio = audio_processor.process("path/to/audio.wav")
# Returns: Path to processed audio file
```

### Speech Recognition

```python
from src.audio.speech import SpeechRecognizer

# Initialize speech recognizer
speech_recognizer = SpeechRecognizer(config)

# Transcribe audio
transcription = speech_recognizer.transcribe("path/to/audio.wav", "en")
# Returns: Transcription object with segments
```

## Language Detection

```python
from src.language.detection import LanguageDetector

# Initialize language detector
language_detector = LanguageDetector(config)

# Detect language
language = language_detector.detect("path/to/audio.wav")
# Returns: Language code (e.g., "en", "fr", "es")
```

## Subtitle Generation

### Subtitle Generator

```python
from src.subtitles.generator import SubtitleGenerator

# Initialize subtitle generator
subtitle_generator = SubtitleGenerator(config)

# Generate subtitles from transcription
subtitles = subtitle_generator.generate(transcription)
# Returns: List of Subtitle objects
```

### Subtitle Formatter

```python
from src.subtitles.formatter import SubtitleFormatter

# Initialize subtitle formatter
subtitle_formatter = SubtitleFormatter(config)

# Format and save subtitles
output_path = subtitle_formatter.format_and_save(subtitles, "path/to/output.srt", "srt")
# Returns: Path to output subtitle file
```

## Model Management

```python
from src.utils.model_manager import check_model_exists, load_model, download_model

# Check if model exists
exists = check_model_exists("whisper", "base")
# Returns: True or False

# Download model
success, message = download_model("whisper", "base")
# Returns: (True, "Model downloaded successfully") or (False, "Error message")

# Load model
success, model, message = load_model("whisper", "base")
# Returns: (True, model_object, "Model loaded successfully") or (False, None, "Error message")
```

## Complete Example

Here's a complete example of using the SubWhisper API:

```python
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.video.input import VideoInput
from src.video.extraction import AudioExtractor
from src.audio.processing import AudioProcessor
from src.language.detection import LanguageDetector
from src.audio.speech import SpeechRecognizer
from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter

# Set up configuration
class Args:
    def __init__(self):
        self.input = "video.mp4"
        self.output = "subtitles.srt"
        self.format = "srt"
        self.language = None  # Auto-detect
        self.whisper_model = "base"
        self.gpu = True
        self.verbose = True
        self.temp_dir = None

# Initialize components
config = Config(Args())
logger = setup_logger(verbose=True)

# Process video
video_input = VideoInput("video.mp4")
audio_extractor = AudioExtractor(config)
audio_path = audio_extractor.extract("video.mp4")

# Process audio
audio_processor = AudioProcessor(config)
processed_audio = audio_processor.process(audio_path)

# Detect language
language_detector = LanguageDetector(config)
language = language_detector.detect(processed_audio)
print(f"Detected language: {language}")

# Transcribe audio
speech_recognizer = SpeechRecognizer(config)
transcription = speech_recognizer.transcribe(processed_audio, language)

# Generate subtitles
subtitle_generator = SubtitleGenerator(config)
subtitles = subtitle_generator.generate(transcription)

# Save subtitles
subtitle_formatter = SubtitleFormatter(config)
output_path = subtitle_formatter.format_and_save(subtitles, "subtitles.srt", "srt")
print(f"Subtitles saved to: {output_path}")
```

## Error Handling

All SubWhisper components raise specific exceptions that can be caught and handled:

```python
from src.video.input import VideoInputError
from src.video.extraction import AudioExtractionError
from src.audio.processing import AudioProcessingError
from src.audio.speech import SpeechRecognitionError
from src.language.detection import LanguageDetectionError
from src.subtitles.generator import SubtitleGenerationError
from src.subtitles.formatter import SubtitleFormattingError

try:
    # Your code here
except VideoInputError as e:
    print(f"Video input error: {str(e)}")
except AudioExtractionError as e:
    print(f"Audio extraction error: {str(e)}")
# And so on...
``` 