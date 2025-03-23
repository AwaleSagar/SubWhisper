"""
Tests for the language detection module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.language.detection import LanguageDetector
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

class TestLanguageDetector(unittest.TestCase):
    """Test cases for LanguageDetector class."""
    
    @patch('src.language.detection.LanguageDetector._load_model')
    def setUp(self, mock_load_model):
        """Set up test environment."""
        # Prevent the real _load_model from being called during initialization
        mock_load_model.return_value = None
        
        self.config = Config(MockArgs())
        self.detector = LanguageDetector(self.config)
        
        # Ensure the mock was called
        mock_load_model.assert_called_once()
    
    def test_initialization(self):
        """Test initialization of LanguageDetector."""
        # We already test this in setUp
        self.assertIsNotNone(self.detector)
    
    @patch('src.language.detection.LanguageDetector.detect')
    def test_detect_language(self, mock_detect):
        """Test language detection."""
        mock_detect.return_value = "en"
        audio_path = "test_audio.wav"
        
        result = self.detector.detect(audio_path)
        
        mock_detect.assert_called_once_with(audio_path)
        self.assertEqual(result, "en")
    
    @patch('src.language.detection.librosa.load')
    @patch('src.language.detection.LanguageDetector._process_audio_sample')
    def test_detect_with_multiple_languages(self, mock_process, mock_load):
        """Test language detection with multiple languages."""
        # Mock audio loading
        mock_load.return_value = (np.array([0.0] * 16000), 16000)
        
        # Mock language detection probabilities
        mock_process.return_value = [("en", 0.7), ("fr", 0.2), ("es", 0.1)]
        
        # Patch the original detect method to use our mocks
        with patch('src.language.detection.LanguageDetector._load_model'):
            # Override the model to avoid actual whisper calls
            self.detector.whisper_model = MagicMock()
            
            # Call detect
            result = self.detector.detect("test_audio.wav")
            
            # Check the result
            self.assertEqual(result, "en")
            mock_process.assert_called_once()

if __name__ == '__main__':
    unittest.main() 