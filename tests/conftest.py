"""
Test configuration for SubWhisper.
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_input(monkeypatch):
    """Mock user input to always return 'y'."""
    monkeypatch.setattr('builtins.input', lambda _: 'y')

@pytest.fixture
def test_env():
    """Set up test environment variables."""
    with patch.dict('os.environ', {
        'SUBWHISPER_WHISPER_MODEL': 'tiny',
        'SUBWHISPER_GPU': 'false',
        'SUBWHISPER_VERBOSE': 'false'
    }):
        yield

@pytest.fixture
def mock_whisper_model():
    """Mock whisper model for testing."""
    with patch('whisper.load_model') as mock_load:
        mock_model = patch('whisper.model').start()
        mock_load.return_value = mock_model
        yield mock_model
        patch('whisper.model').stop()

@pytest.fixture
def temp_dir(tmpdir):
    """Create a temporary directory for test files."""
    return tmpdir.mkdir("subwhisper_test") 