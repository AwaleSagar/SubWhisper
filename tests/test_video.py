"""
Tests for the video module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.video.input import VideoInput, VideoInputError
from src.video.extraction import AudioExtractor, AudioExtractionError
from src.utils.config import Config

class MockArgs:
    """Mock arguments for testing."""
    
    def __init__(self):
        self.input = "test_video.mp4"
        self.output = "test_subtitles.srt"
        self.format = "srt"
        self.language = None
        self.model = "tiny"
        self.whisper_model = "tiny"
        self.gpu = False
        self.verbose = False
        self.temp_dir = None

class TestVideoInput(unittest.TestCase):
    """Test cases for VideoInput class."""
    
    @patch("os.path.isfile")
    def test_video_not_found(self, mock_isfile):
        """Test handling of non-existent video file."""
        mock_isfile.return_value = False
        
        with self.assertRaises(VideoInputError):
            VideoInput("non_existent_video.mp4")
    
    @patch("os.path.isfile")
    def test_unsupported_format(self, mock_isfile):
        """Test handling of unsupported video format."""
        mock_isfile.return_value = True
        
        with self.assertRaises(VideoInputError):
            VideoInput("video.unsupported")
    
    @patch("os.path.isfile")
    @patch("subprocess.run")
    def test_validation_no_streams(self, mock_run, mock_isfile):
        """Test validation of video with no video stream."""
        mock_isfile.return_value = True
        
        # Mock ffprobe output with no video stream
        mock_process = MagicMock()
        mock_process.stdout = '{"streams": [{"codec_type": "audio"}]}'
        mock_run.return_value = mock_process
        
        with self.assertRaises(VideoInputError):
            VideoInput("video.mp4")
    
    @patch("os.path.isfile")
    @patch("subprocess.run")
    def test_get_info(self, mock_run, mock_isfile):
        """Test getting video information."""
        mock_isfile.return_value = True
        
        # Mock ffprobe output for validation
        mock_validation = MagicMock()
        mock_validation.stdout = '{"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]}'
        
        # Mock ffprobe output for get_info
        mock_info = MagicMock()
        mock_info.stdout = '''{
            "format": {"duration": "60.5", "size": "1000000", "bit_rate": "1000000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": 48000, "channels": 2}
            ]
        }'''
        
        # Mock run to return different values on different calls
        mock_run.side_effect = [mock_validation, mock_info]
        
        video_input = VideoInput("video.mp4")
        info = video_input.get_info()
        
        self.assertEqual(info["duration"], 60.5)
        self.assertEqual(info["size"], 1000000)
        self.assertEqual(len(info["streams"]), 2)
        self.assertEqual(info["streams"][0]["width"], 1920)
        self.assertEqual(info["streams"][1]["sample_rate"], 48000)

class TestAudioExtractor(unittest.TestCase):
    """Test cases for AudioExtractor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config(MockArgs())
    
    @patch("subprocess.run")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    def test_extract_success(self, mock_getsize, mock_isfile, mock_run):
        """Test successful audio extraction."""
        mock_run.return_value = MagicMock()
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000  # Non-zero file size
        
        extractor = AudioExtractor(self.config)
        output_path = extractor.extract("video.mp4", "audio.wav")
        
        self.assertEqual(output_path, "audio.wav")
        mock_run.assert_called_once()
    
    @patch("subprocess.run")
    def test_extract_failure(self, mock_run):
        """Test failed audio extraction."""
        # Mock subprocess.run to raise an exception
        mock_run.side_effect = Exception("Extraction failed")
        
        extractor = AudioExtractor(self.config)
        
        with self.assertRaises(AudioExtractionError):
            extractor.extract("video.mp4", "audio.wav")
    
    @patch("subprocess.run")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    def test_extract_segment(self, mock_getsize, mock_isfile, mock_run):
        """Test extraction of audio segment."""
        mock_run.return_value = MagicMock()
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000  # Non-zero file size
        
        extractor = AudioExtractor(self.config)
        output_path = extractor.extract_segment("video.mp4", 10.0, 5.0, "segment.wav")
        
        self.assertEqual(output_path, "segment.wav")
        mock_run.assert_called_once()

if __name__ == "__main__":
    unittest.main() 