"""
Configuration module for SubWhisper.

This module provides tools for loading and managing application configuration.
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from src.utils.errors import ConfigError, handle_exception

logger = logging.getLogger("subwhisper")

class Config:
    """Configuration handler for SubWhisper."""
    
    def __init__(self, args: Any, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            args: Arguments object with configuration values
            config_file: Path to configuration file (optional)
        """
        self.args = args  # Store original args for compatibility with tests
        
        # Try to load configuration from file first
        self.config_file = config_file
        self._file_config = self._load_config_file() if config_file else {}
        
        # Input/output
        self.input_file: Optional[str] = self._get_config_value('input', None)
        self.output_file: Optional[str] = self._get_config_value('output', None)
        self.format: str = self._get_config_value('format', 'srt')
        
        # Speech recognition
        self.language: Optional[str] = self._get_config_value('language', None)
        self.whisper_model: str = self._get_config_value('whisper_model', 'base')
        
        # Processing options
        self.gpu: bool = self._get_config_value('gpu', False)
        self.verbose: bool = self._get_config_value('verbose', False)
        
        # Set up temp directory
        self.temp_dir: str = self._get_config_value('temp_dir', tempfile.gettempdir())
        
        # Audio processing settings
        self.audio_format: str = self._get_config_value('audio_format', 'wav')
        self.audio_sample_rate: int = self._get_config_value('audio_sample_rate', 16000)
        
        # Transcription options
        self.transcription_options: Dict[str, Any] = self._get_config_value('transcription_options', {})
        
        # Load environment configuration (overrides file and args)
        self._load_env_config()
        
        # Print configuration summary if verbose
        if self.verbose:
            self._print_config()
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """
        Get configuration value from args, file, or default.
        
        Args:
            key: Configuration key
            default: Default value if not found in args or file
            
        Returns:
            Configuration value
        """
        # Check command line args first (highest priority)
        if hasattr(self.args, key) and getattr(self.args, key) is not None:
            return getattr(self.args, key)
        
        # Check file config next
        if key in self._file_config:
            return self._file_config[key]
        
        # Fall back to default
        return default
    
    def _load_config_file(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary with configuration values
        """
        if not self.config_file or not os.path.exists(self.config_file):
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                # Determine format from extension
                ext = os.path.splitext(self.config_file)[1].lower()
                if ext == '.json':
                    return json.load(f)
                elif ext == '.yaml' or ext == '.yml':
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        logger.warning("YAML support not available. Install PyYAML for YAML config support.")
                        return {}
                elif ext == '.toml':
                    try:
                        import toml
                        return toml.load(f)
                    except ImportError:
                        logger.warning("TOML support not available. Install toml for TOML config support.")
                        return {}
                else:
                    logger.warning(f"Unsupported configuration format: {ext}")
                    return {}
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {str(e)}")
            return {}
    
    def _load_env_config(self) -> None:
        """
        Load configuration from environment variables.
        
        This method will override configuration with values from environment variables.
        Environment variables are prefixed with SUBWHISPER_.
        """
        # Environment variable mappings
        env_mappings = {
            'SUBWHISPER_WHISPER_MODEL': ('whisper_model', str),
            'SUBWHISPER_GPU': ('gpu', lambda x: x.lower() in ('true', '1', 'yes')),
            'SUBWHISPER_VERBOSE': ('verbose', lambda x: x.lower() in ('true', '1', 'yes')),
            'SUBWHISPER_TEMP_DIR': ('temp_dir', str),
            'SUBWHISPER_AUDIO_FORMAT': ('audio_format', str),
            'SUBWHISPER_AUDIO_SAMPLE_RATE': ('audio_sample_rate', int),
        }
        
        # Override config with environment variables
        for env_var, (attr_name, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    setattr(self, attr_name, value)
                    logger.debug(f"Config override from environment: {attr_name}={value}")
                except Exception as e:
                    logger.warning(f"Failed to parse environment variable {env_var}: {str(e)}")
    
    def _print_config(self) -> None:
        """Print configuration summary."""
        logger.info("SubWhisper Configuration:")
        logger.info(f"  Input: {self.input_file or 'Not specified'}")
        logger.info(f"  Output: {self.output_file or 'Not specified'}")
        logger.info(f"  Format: {self.format}")
        logger.info(f"  Language: {self.language or 'Auto-detect'}")
        logger.info(f"  Whisper Model: {self.whisper_model}")
        logger.info(f"  GPU: {'Enabled' if self.gpu else 'Disabled'}")
        logger.info(f"  Temp Directory: {self.temp_dir}")
        logger.info(f"  Audio Format: {self.audio_format}")
        logger.info(f"  Audio Sample Rate: {self.audio_sample_rate}")
        
        if self.config_file:
            logger.info(f"  Config File: {self.config_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with configuration values
        """
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "format": self.format,
            "language": self.language,
            "whisper_model": self.whisper_model,
            "gpu": self.gpu,
            "verbose": self.verbose,
            "temp_dir": self.temp_dir,
            "audio_format": self.audio_format,
            "audio_sample_rate": self.audio_sample_rate,
            "transcription_options": self.transcription_options,
        }
    
    @handle_exception
    def save_to_file(self, config_path: str) -> str:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            Path to saved configuration file
            
        Raises:
            ConfigError: If configuration cannot be saved
        """
        # Create parent directories if needed
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Convert config to dictionary
        config_dict = self.to_dict()
        
        try:
            # Determine format from extension
            ext = os.path.splitext(config_path)[1].lower()
            if ext == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2)
            elif ext == '.yaml' or ext == '.yml':
                try:
                    import yaml
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config_dict, f)
                except ImportError:
                    raise ConfigError(
                        "YAML support not available. Install PyYAML for YAML config support.",
                        code="YAML_NOT_AVAILABLE"
                    )
            elif ext == '.toml':
                try:
                    import toml
                    with open(config_path, 'w', encoding='utf-8') as f:
                        toml.dump(config_dict, f)
                except ImportError:
                    raise ConfigError(
                        "TOML support not available. Install toml for TOML config support.",
                        code="TOML_NOT_AVAILABLE"
                    )
            else:
                raise ConfigError(
                    f"Unsupported configuration format: {ext}",
                    code="UNSUPPORTED_CONFIG_FORMAT"
                )
            
            logger.info(f"Configuration saved to: {config_path}")
            return config_path
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(
                f"Failed to save configuration file: {str(e)}",
                code="CONFIG_SAVE_FAILED"
            )
    
    @handle_exception
    def get_temp_file_path(self, prefix: str, suffix: str) -> str:
        """
        Generate a path for a temporary file.
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix/extension
            
        Returns:
            Path to the temporary file
            
        Raises:
            ConfigError: If temporary directory cannot be created
        """
        # Ensure the temporary directory exists
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except Exception as e:
            raise ConfigError(
                f"Failed to create temporary directory: {str(e)}",
                code="TEMP_DIR_CREATE_FAILED"
            )
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}{suffix}"
        
        return os.path.join(self.temp_dir, filename)
    
    @handle_exception
    def get_model_path(self, model_type: str) -> str:
        """
        Get the directory path for storing models.
        
        Args:
            model_type: Type of model (e.g., 'whisper', 'language')
            
        Returns:
            Path to the model directory
            
        Raises:
            ConfigError: If model directory cannot be created
        """
        # Determine the base models directory relative to the module
        base_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models"
        ))
        
        # Create type-specific model directory
        model_dir = os.path.join(base_dir, model_type)
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            return model_dir
        except Exception as e:
            raise ConfigError(
                f"Failed to create model directory: {str(e)}",
                code="MODEL_DIR_CREATE_FAILED"
            )
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns:
            Dictionary with model configuration
        """
        # Default model configuration
        model_config = {
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0,  # Use greedy decoding
            "fp16": self.gpu,  # Use fp16 on GPU
            "verbose": self.verbose
        }
        
        # Override with transcription options if provided
        model_config.update(self.transcription_options)
        
        return model_config
    
    @handle_exception
    def get_output_file(self, input_file: Optional[str] = None) -> str:
        """
        Get the output file path, generating one if not specified.
        
        Args:
            input_file: Path to the input file (for generating output name)
            
        Returns:
            Path to the output file
            
        Raises:
            ConfigError: If output file cannot be determined
        """
        # If output file is specified, use it
        if self.output_file:
            return self.output_file
        
        # No output file specified, generate from input file
        if not input_file and not self.input_file:
            raise ConfigError(
                "No output file specified and no input file to generate from",
                code="NO_OUTPUT_FILE"
            )
        
        input_path = input_file or self.input_file
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Generate output in the same directory as input
        output_dir = os.path.dirname(os.path.abspath(input_path))
        output_file = os.path.join(output_dir, f"{input_name}.{self.format}")
        
        return output_file 