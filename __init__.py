"""
SubWhisper - Audio extraction and automatic subtitle generation from video files.

This package provides tools to extract audio from videos, transcribe speech,
and generate subtitle files in various formats.
"""

__version__ = "0.1.0"
__author__ = "Sagar Awale"

# Make important classes available at the package level
from src.video.input import VideoInput
from src.video.extraction import AudioExtractor
from src.audio.processing import AudioProcessor
from src.audio.speech import SpeechRecognizer
from src.language.detection import LanguageDetector
from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter
from src.utils.config import Config 