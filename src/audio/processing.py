"""
Audio preprocessing module for SubWhisper.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import librosa
import soundfile as sf

from src.utils.config import Config

logger = logging.getLogger("subwhisper")

class AudioProcessingError(Exception):
    """Exception raised for errors in audio processing."""
    pass

class AudioSegment:
    """Class for handling audio segments."""
    
    def __init__(self, path: str = None, data: np.ndarray = None, sample_rate: int = 16000, channels: int = 1):
        """Initialize audio segment."""
        self.path = path
        self.data = data
        self.sample_rate = sample_rate
        self.channels = channels
        self._duration = None
        self.dBFS = self._calculate_dBFS() if data is not None else None
    
    @property
    def duration(self) -> float:
        """Get duration of audio segment in seconds."""
        if self._duration is None and self.data is not None:
            self._duration = len(self.data) / self.sample_rate
        return self._duration
    
    @property
    def frame_rate(self) -> int:
        """Get sample rate of audio segment."""
        return self.sample_rate
    
    def _calculate_dBFS(self) -> float:
        """Calculate dBFS (decibels relative to full scale)."""
        if self.data is None or len(self.data) == 0:
            return -np.inf
        
        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(self.data)))
        
        # Convert to dBFS (assuming full scale is 1.0)
        if rms <= 0:
            return -np.inf
        return 20 * np.log10(rms)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'AudioSegment':
        """Load audio from file."""
        try:
            data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # If stereo, convert data shape to (samples, channels)
            if len(data.shape) > 1 and data.shape[0] == 2:
                data = data.T
                channels = 2
            else:
                # If mono, ensure data is the right shape
                data = np.atleast_2d(data).T if len(data.shape) == 1 else data
                channels = 1
            
            segment = cls(path=file_path, data=data, sample_rate=sample_rate, channels=channels)
            return segment
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise AudioProcessingError(f"Failed to load audio file: {str(e)}")
    
    def set_channels(self, channels: int) -> 'AudioSegment':
        """Set number of channels in the audio segment."""
        if self.channels == channels:
            return self
        
        if channels == 1 and self.channels == 2:
            # Convert stereo to mono
            mono_data = np.mean(self.data, axis=1, keepdims=True)
            result = AudioSegment(path=self.path, data=mono_data, sample_rate=self.sample_rate, channels=1)
            return result
        elif channels == 2 and self.channels == 1:
            # Convert mono to stereo (duplicate channel)
            stereo_data = np.tile(self.data, (1, 2))
            result = AudioSegment(path=self.path, data=stereo_data, sample_rate=self.sample_rate, channels=2)
            return result
        else:
            raise AudioProcessingError(f"Unsupported channel conversion: {self.channels} to {channels}")
    
    def apply_gain(self, gain_db: float) -> 'AudioSegment':
        """Apply gain to audio segment."""
        if self.data is None:
            return self
        
        # Convert dB gain to amplitude multiplier
        gain_factor = 10 ** (gain_db / 20)
        
        # Apply gain
        gained_data = self.data * gain_factor
        
        # Create new segment with gained data
        result = AudioSegment(
            path=self.path, 
            data=gained_data, 
            sample_rate=self.sample_rate, 
            channels=self.channels
        )
        
        return result
    
    def save(self, output_path: str) -> str:
        """Save audio to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save audio data
            if self.channels == 2:
                # For stereo, transpose data to (channels, samples)
                data_to_save = self.data.T
            else:
                # For mono, flatten if needed
                data_to_save = self.data.flatten() if self.data.shape[1] == 1 else self.data
            
            sf.write(output_path, data_to_save, self.sample_rate)
            return output_path
        except Exception as e:
            logger.error(f"Error saving audio file {output_path}: {str(e)}")
            raise AudioProcessingError(f"Failed to save audio file: {str(e)}")


class AudioProcessor:
    """Audio processing class for preprocessing audio before speech recognition."""
    
    def __init__(self, config: Config):
        """
        Initialize audio processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def load_audio(self, audio_path: str) -> AudioSegment:
        """
        Load audio from file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            AudioSegment object
        """
        return AudioSegment.from_file(audio_path)
    
    def convert_to_mono(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Convert audio segment to mono.
        
        Args:
            audio_segment: Audio segment to convert
            
        Returns:
            Mono audio segment
        """
        return audio_segment.set_channels(1)
    
    def normalize_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Normalize audio volume to target dBFS.
        
        Args:
            audio_segment: Audio segment to normalize
            
        Returns:
            Normalized audio segment
        """
        target_dBFS = -3
        gain = target_dBFS - audio_segment.dBFS
        return audio_segment.apply_gain(gain)
    
    def process(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Process audio file with preprocessing steps like noise reduction and normalization.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the processed audio file (optional)
            
        Returns:
            Path to the processed audio file
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        try:
            # If output path is not provided, create a temporary file
            if output_path is None:
                output_path = self.config.get_temp_file_path(
                    "processed_audio", f".{self.config.audio_format}"
                )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Load audio file
            logger.info(f"Loading audio file: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate, mono=True)
            
            # Apply preprocessing steps
            logger.info("Applying audio preprocessing")
            processed_audio = self._preprocess_audio(audio)
            
            # Save processed audio
            logger.info(f"Saving processed audio to: {output_path}")
            sf.write(output_path, processed_audio, sr)
            
            return output_path
            
        except Exception as e:
            error_message = f"Audio processing failed: {str(e)}"
            logger.error(error_message)
            raise AudioProcessingError(error_message)
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to the audio data.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        # Step 1: Noise reduction using spectral gating
        audio = self._reduce_noise(audio)
        
        # Step 2: Normalize audio
        audio = self._normalize_audio(audio)
        
        # Step 3: Apply high-pass filter to remove low frequency noise
        audio = self._apply_highpass_filter(audio)
        
        return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using spectral gating.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Noise-reduced audio data
        """
        # Simple spectral gating noise reduction
        # This is a basic implementation, could be improved with more sophisticated methods
        
        # Calculate the Short-Time Fourier Transform (STFT)
        stft = librosa.stft(audio)
        
        # Calculate magnitude and phase
        magnitude, phase = librosa.magphase(stft)
        
        # Estimate noise profile from the first 500ms of audio (assuming it's silence or background noise)
        noise_samples = int(self.config.audio_sample_rate * 0.5)
        noise_profile = np.mean(magnitude[:, :noise_samples], axis=1)
        
        # Apply spectral gating: Subtract noise profile and apply a threshold
        threshold = 2.0  # Adjust based on your needs
        magnitude = np.maximum(0, magnitude - threshold * np.reshape(noise_profile, (-1, 1)))
        
        # Reconstruct the audio signal
        audio_denoised = librosa.istft(magnitude * phase)
        
        return audio_denoised
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have a maximum absolute amplitude of 0.95.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        # Calculate the maximum absolute amplitude
        max_amplitude = np.max(np.abs(audio))
        
        # If audio is silent, return as is
        if max_amplitude < 1e-10:
            return audio
        
        # Normalize to have max amplitude of 0.95 (avoiding clipping)
        normalized_audio = 0.95 * audio / max_amplitude
        
        return normalized_audio
    
    def _apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply a high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Filtered audio data
        """
        # Apply a high-pass filter at 80Hz
        filter_cutoff = 80  # Hz
        filtered_audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        return filtered_audio
    
    def get_audio_segments(self, audio_path: str, segment_duration: float = 30.0) -> list:
        """
        Split audio into segments for processing.
        
        Args:
            audio_path: Path to the audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of segment file paths
        """
        try:
            # Load audio file
            logger.info(f"Loading audio for segmentation: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate, mono=True)
            
            # Calculate total duration
            duration = len(audio) / sr
            
            # Create output directory for segments
            segments_dir = os.path.join(self.config.temp_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)
            
            # Split audio into segments
            segment_paths = []
            for i, start in enumerate(np.arange(0, duration, segment_duration)):
                end = min(start + segment_duration, duration)
                segment_length = int((end - start) * sr)
                
                if segment_length <= 0:
                    continue
                
                # Extract segment
                segment = audio[int(start * sr):int(end * sr)]
                
                # Save segment
                segment_path = os.path.join(
                    segments_dir, 
                    f"segment_{i}_{start:.2f}_{end:.2f}.{self.config.audio_format}"
                )
                sf.write(segment_path, segment, sr)
                segment_paths.append(segment_path)
            
            logger.info(f"Audio split into {len(segment_paths)} segments")
            return segment_paths
            
        except Exception as e:
            error_message = f"Audio segmentation failed: {str(e)}"
            logger.error(error_message)
            raise AudioProcessingError(error_message) 