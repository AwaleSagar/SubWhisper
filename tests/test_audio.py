"""
Tests for the audio processing and speech recognition modules.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.processing import AudioProcessor
from src.audio.speech import SpeechRecognizer
from src.utils.config import Config


class TestAudioProcessor(unittest.TestCase):
    """Test cases for the AudioProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock config
        args = MagicMock()
        args.temp_dir = self.temp_dir
        args.verbose = False
        
        self.config = Config(args)
        self.audio_processor = AudioProcessor(self.config)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    @patch('src.audio.processing.AudioSegment')
    def test_load_audio(self, mock_audio_segment):
        """Test loading audio from a file."""
        # Mock the AudioSegment.from_file method
        mock_segment = MagicMock()
        mock_audio_segment.from_file.return_value = mock_segment
        mock_segment.frame_rate = 16000
        mock_segment.channels = 1
        
        # Test loading an audio file
        test_path = "test_audio.wav"
        result = self.audio_processor.load_audio(test_path)
        
        # Verify that AudioSegment.from_file was called correctly
        mock_audio_segment.from_file.assert_called_once_with(test_path)
        
        # Verify that the result is the mock segment
        self.assertEqual(result, mock_segment)

    @patch('src.audio.processing.AudioSegment')
    def test_convert_to_mono(self, mock_audio_segment):
        """Test converting stereo audio to mono."""
        # Create a mock stereo segment
        mock_segment = MagicMock()
        mock_segment.channels = 2
        mock_segment.set_channels.return_value = MagicMock(channels=1)
        
        # Test converting to mono
        result = self.audio_processor.convert_to_mono(mock_segment)
        
        # Verify that set_channels was called with 1
        mock_segment.set_channels.assert_called_once_with(1)
        
        # Verify that the result has 1 channel
        self.assertEqual(result.channels, 1)

    @patch('src.audio.processing.AudioSegment')
    def test_normalize_audio(self, mock_audio_segment):
        """Test normalizing audio volume."""
        # Create a mock audio segment
        mock_segment = MagicMock()
        mock_segment.dBFS = -20
        mock_segment.apply_gain.return_value = MagicMock(dBFS=-3)
        
        # Test normalizing audio
        result = self.audio_processor.normalize_audio(mock_segment)
        
        # Verify that apply_gain was called with the correct gain
        # Should be increasing volume by (target_dBFS - current_dBFS)
        mock_segment.apply_gain.assert_called_once_with(17)  # -3 - (-20) = 17
        
        # Verify that the result has the expected dBFS
        self.assertEqual(result.dBFS, -3)


class TestSpeechRecognizer(unittest.TestCase):
    """Test cases for the SpeechRecognizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        args = MagicMock()
        args.model = "base"
        args.gpu = False
        args.language = None
        args.verbose = False
        
        self.config = Config(args)
        
        # Patch the whisper module
        self.whisper_patcher = patch('src.audio.speech.whisper')
        self.mock_whisper = self.whisper_patcher.start()
        
        # Mock the model
        self.mock_model = MagicMock()
        self.mock_whisper.load_model.return_value = self.mock_model
        
        self.speech_recognizer = SpeechRecognizer(self.config)

    def tearDown(self):
        """Tear down test fixtures."""
        self.whisper_patcher.stop()

    def test_load_model(self):
        """Test loading the speech recognition model."""
        # Load the model
        model = self.speech_recognizer.load_model()
        
        # Verify that whisper.load_model was called with the correct parameters
        self.mock_whisper.load_model.assert_called_once_with("base", device="cpu")
        
        # Verify that the returned model is the mock model
        self.assertEqual(model, self.mock_model)

    def test_transcribe(self):
        """Test transcribing audio."""
        # Mock audio data
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Mock transcription result
        mock_result = {
            "text": "This is a test transcription.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "This is a test transcription."
                }
            ],
            "language": "en"
        }
        self.mock_model.transcribe.return_value = mock_result
        
        # Test transcribing audio
        result = self.speech_recognizer.transcribe(audio_data)
        
        # Verify that the model's transcribe method was called
        self.mock_model.transcribe.assert_called_once()
        
        # Verify that the result matches the mock result
        self.assertEqual(result, mock_result)

    def test_transcribe_with_language(self):
        """Test transcribing audio with a specified language."""
        # Update the config to specify a language
        self.config.args.language = "es"
        
        # Create a new speech recognizer with the updated config
        speech_recognizer = SpeechRecognizer(self.config)
        
        # Mock audio data
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Mock transcription result
        mock_result = {
            "text": "Esta es una prueba de transcripción.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Esta es una prueba de transcripción."
                }
            ],
            "language": "es"
        }
        self.mock_model.transcribe.return_value = mock_result
        
        # Test transcribing audio with language specified
        result = speech_recognizer.transcribe(audio_data)
        
        # Verify that the model's transcribe method was called with the language parameter
        self.mock_model.transcribe.assert_called_once()
        call_args = self.mock_model.transcribe.call_args[1]
        self.assertEqual(call_args.get("language"), "es")
        
        # Verify that the result matches the mock result
        self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main() 