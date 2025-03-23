"""
Configuration file handling module for SubWhisper.

This module provides functionality to load and save configuration from/to
configuration files in various formats (JSON, YAML, TOML).
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from src.utils.errors import ConfigError, handle_exception

logger = logging.getLogger("subwhisper")

CONFIG_FORMATS = ["json", "yaml", "toml"]
DEFAULT_CONFIG_NAME = "subwhisper.json"

# Try to import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("YAML support not available. Install PyYAML for YAML config support.")

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False
    logger.debug("TOML support not available. Install toml for TOML config support.")


def get_default_config_path() -> str:
    """
    Get the default configuration file path.
    
    Returns:
        Path to the default configuration file
    """
    # Use XDG_CONFIG_HOME or ~/.config on Unix-like systems
    if os.name == "posix":
        config_dir = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        return os.path.join(config_dir, "subwhisper", DEFAULT_CONFIG_NAME)
    # Use AppData on Windows
    elif os.name == "nt":
        config_dir = os.environ.get("APPDATA", os.path.expanduser("~/AppData/Roaming"))
        return os.path.join(config_dir, "SubWhisper", DEFAULT_CONFIG_NAME)
    # Fallback to current directory
    else:
        return os.path.join(os.getcwd(), DEFAULT_CONFIG_NAME)


@handle_exception
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
                    (if None, uses default path)
    
    Returns:
        Dictionary with configuration values
    
    Raises:
        ConfigError: If configuration file cannot be loaded
    """
    # Use default path if not specified
    if config_path is None:
        config_path = get_default_config_path()
    
    if not os.path.exists(config_path):
        logger.debug(f"Configuration file not found: {config_path}")
        return {}
    
    # Determine format from file extension
    ext = os.path.splitext(config_path)[1].lower().lstrip(".")
    
    try:
        if ext == "json":
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif ext == "yaml" or ext == "yml":
            if not YAML_AVAILABLE:
                raise ConfigError(
                    "YAML support not available. Install PyYAML for YAML config support.",
                    code="YAML_NOT_AVAILABLE"
                )
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif ext == "toml":
            if not TOML_AVAILABLE:
                raise ConfigError(
                    "TOML support not available. Install toml for TOML config support.",
                    code="TOML_NOT_AVAILABLE"
                )
            with open(config_path, "r", encoding="utf-8") as f:
                return toml.load(f)
        else:
            raise ConfigError(
                f"Unsupported configuration format: {ext}",
                code="UNSUPPORTED_CONFIG_FORMAT"
            )
    except Exception as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(
            f"Failed to load configuration file: {str(e)}",
            code="CONFIG_LOAD_FAILED"
        )


@handle_exception
def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> str:
    """
    Save configuration to a file.
    
    Args:
        config: Dictionary with configuration values
        config_path: Path to the configuration file
                    (if None, uses default path)
    
    Returns:
        Path to the saved configuration file
    
    Raises:
        ConfigError: If configuration file cannot be saved
    """
    # Use default path if not specified
    if config_path is None:
        config_path = get_default_config_path()
    
    # Create parent directories if needed
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Determine format from file extension
    ext = os.path.splitext(config_path)[1].lower().lstrip(".")
    
    try:
        if ext == "json":
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        elif ext == "yaml" or ext == "yml":
            if not YAML_AVAILABLE:
                raise ConfigError(
                    "YAML support not available. Install PyYAML for YAML config support.",
                    code="YAML_NOT_AVAILABLE"
                )
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)
        elif ext == "toml":
            if not TOML_AVAILABLE:
                raise ConfigError(
                    "TOML support not available. Install toml for TOML config support.",
                    code="TOML_NOT_AVAILABLE"
                )
            with open(config_path, "w", encoding="utf-8") as f:
                toml.dump(config, f)
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


def get_config_as_dict(config_obj: Any) -> Dict[str, Any]:
    """
    Convert a configuration object to a dictionary.
    
    Args:
        config_obj: Configuration object
    
    Returns:
        Dictionary with configuration values
    """
    # If it's a Config object with a to_dict method, use it
    if hasattr(config_obj, "to_dict") and callable(config_obj.to_dict):
        return config_obj.to_dict()
    
    # Otherwise, get all public attributes
    config_dict = {}
    for key, value in vars(config_obj).items():
        if not key.startswith("_"):
            config_dict[key] = value
    
    return config_dict 