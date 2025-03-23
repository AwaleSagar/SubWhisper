"""
Tests for the subtitles module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
from datetime import timedelta

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.subtitles.generator import SubtitleGenerator
from src.subtitles.formatter import SubtitleFormatter, SubtitleFormattingError
from src.utils.config import Config

class MockArgs:
    """Mock arguments for testing."""
    
    def __init__(self):
        self.input = "test_video.mp4"
        self.output = "test_subtitles.srt"
        self.format = "srt"
        self.language = None
        self.whisper_model = "tiny"
        self.gpu = False
        self.verbose = False
        self.temp_dir = None

class MockTranscript:
    """Mock transcript for testing."""
    
    def __init__(self):
        self.segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello, world."},
            {"start": 3.0, "end": 6.0, "text": "This is a test."},
            {"start": 7.5, "end": 10.0, "text": "Goodbye, world."}
        ]

class TestSubtitleGenerator(unittest.TestCase):
    """Test cases for SubtitleGenerator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config(MockArgs())
        self.generator = SubtitleGenerator(self.config)
        self.mock_transcript = MockTranscript()
    
    def test_initialization(self):
        """Test initialization of SubtitleGenerator."""
        generator = SubtitleGenerator(self.config)
        self.assertIsNotNone(generator)
    
    def test_generate_subtitles(self):
        """Test subtitle generation from transcript."""
        subtitles = self.generator.generate(self.mock_transcript)
        
        self.assertEqual(len(subtitles), 3)
        self.assertEqual(subtitles[0].text, "Hello, world.")
        self.assertEqual(subtitles[1].text, "This is a test.")
        self.assertEqual(subtitles[2].text, "Goodbye, world.")
        
        # Check timing
        self.assertEqual(subtitles[0].start.total_seconds(), 0.0)
        self.assertEqual(subtitles[0].end.total_seconds(), 2.5)
        self.assertEqual(subtitles[1].start.total_seconds(), 3.0)
        self.assertEqual(subtitles[1].end.total_seconds(), 6.0)

class TestSubtitleFormatter(unittest.TestCase):
    """Test cases for SubtitleFormatter class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config(MockArgs())
        self.formatter = SubtitleFormatter(self.config)
        
        # Create mock subtitles
        self.mock_subtitles = [
            MagicMock(start=timedelta(seconds=0), end=timedelta(seconds=2.5), text="Hello, world."),
            MagicMock(start=timedelta(seconds=3), end=timedelta(seconds=6), text="This is a test."),
            MagicMock(start=timedelta(seconds=7.5), end=timedelta(seconds=10), text="Goodbye, world.")
        ]
    
    def test_initialization(self):
        """Test initialization of SubtitleFormatter."""
        formatter = SubtitleFormatter(self.config)
        self.assertIsNotNone(formatter)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_format_and_save_srt(self, mock_file):
        """Test formatting and saving subtitles in SRT format."""
        output_path = "test_subtitles.srt"
        
        result = self.formatter.format_and_save(self.mock_subtitles, output_path, "srt")
        
        mock_file.assert_called_once_with(output_path, 'w', encoding='utf-8')
        self.assertEqual(result, output_path)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_format_and_save_vtt(self, mock_file):
        """Test formatting and saving subtitles in WebVTT format."""
        output_path = "test_subtitles.vtt"
        
        result = self.formatter.format_and_save(self.mock_subtitles, output_path, "vtt")
        
        mock_file.assert_called_once_with(output_path, 'w', encoding='utf-8')
        self.assertEqual(result, output_path)
    
    def test_unsupported_format(self):
        """Test handling of unsupported subtitle format."""
        output_path = "test_subtitles.xyz"
        
        with self.assertRaises(SubtitleFormattingError):
            self.formatter.format_and_save(self.mock_subtitles, output_path, "xyz")

if __name__ == '__main__':
    unittest.main() 