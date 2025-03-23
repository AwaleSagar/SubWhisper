"""
Environment variable utilities for SubWhisper.
"""

import os
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file if it exists
load_dotenv()

# Define environment variable prefixes
ENV_PREFIX = "SUBWHISPER_"

# Define environment variable mappings (env_var_name => config_attr_name)
ENV_MAPPING = {
    "INPUT": "input",
    "OUTPUT": "output",
    "FORMAT": "format",
    "LANGUAGE": "language",
    "WHISPER_MODEL": "whisper_model",
    "GPU": "gpu",
    "VERBOSE": "verbose",
    "TEMP_DIR": "temp_dir",
}

def get_env_var(name: str, default: Any = None) -> Any:
    """
    Get an environment variable with the SubWhisper prefix.
    
    Args:
        name: The name of the environment variable without the prefix
        default: Default value if the environment variable is not set
        
    Returns:
        The value of the environment variable or the default value
    """
    env_name = f"{ENV_PREFIX}{name}"
    return os.environ.get(env_name, default)

def get_env_bool(name: str, default: bool = False) -> bool:
    """
    Get a boolean environment variable.
    
    Args:
        name: The name of the environment variable without the prefix
        default: Default value if the environment variable is not set
        
    Returns:
        True if the environment variable is "1", "true", "yes", or "y" (case-insensitive)
        False if the environment variable is "0", "false", "no", or "n" (case-insensitive)
        The default value if the environment variable is not set or has another value
    """
    value = get_env_var(name)
    
    if value is None:
        return default
    
    value = value.lower()
    if value in ["1", "true", "yes", "y"]:
        return True
    elif value in ["0", "false", "no", "n"]:
        return False
    else:
        logger.warning(f"Invalid boolean environment variable {ENV_PREFIX}{name}: {value}")
        return default

def get_env_path(name: str, default: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Get a path environment variable.
    
    Args:
        name: The name of the environment variable without the prefix
        default: Default value if the environment variable is not set
        
    Returns:
        The path as a Path object or the default value
    """
    value = get_env_var(name)
    
    if value is None:
        if default is None:
            return None
        elif isinstance(default, str):
            return Path(default)
        else:
            return default
    
    return Path(value)

def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary of configuration values loaded from environment variables
    """
    config = {}
    
    # Load each environment variable
    for env_name, config_name in ENV_MAPPING.items():
        # Process boolean variables
        if config_name in ["gpu", "verbose"]:
            value = get_env_bool(env_name)
            if value:  # Only set if True to avoid overriding command line defaults
                config[config_name] = value
        # Process path variables
        elif config_name in ["input", "output", "temp_dir"]:
            value = get_env_path(env_name)
            if value is not None:
                config[config_name] = value
        # Process other variables
        else:
            value = get_env_var(env_name)
            if value is not None:
                config[config_name] = value
    
    return config

def update_config_from_env(config_obj: Any) -> None:
    """
    Update a configuration object with values from environment variables.
    Environment variables take precedence over existing configuration values.
    
    Args:
        config_obj: Configuration object to update
    """
    env_config = load_env_config()
    
    # Update the config object with environment variable values
    for key, value in env_config.items():
        if hasattr(config_obj, key):
            logger.debug(f"Updating config.{key} from environment variable: {value}")
            setattr(config_obj, key, value)

def save_env_file(config_dict: Dict[str, Any], file_path: Union[str, Path] = ".env") -> None:
    """
    Save configuration to a .env file.
    
    Args:
        config_dict: Dictionary of configuration values
        file_path: Path to the .env file
    """
    lines = []
    
    for config_name, value in config_dict.items():
        # Skip None values
        if value is None:
            continue
        
        # Find the environment variable name
        env_name = None
        for k, v in ENV_MAPPING.items():
            if v == config_name:
                env_name = k
                break
        
        if env_name is None:
            continue
        
        # Format the value based on type
        if isinstance(value, bool):
            formatted_value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            formatted_value = str(value)
        elif isinstance(value, Path):
            formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        # Add the line to the .env file
        lines.append(f"{ENV_PREFIX}{env_name}={formatted_value}")
    
    # Write the .env file
    with open(file_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    logger.info(f"Configuration saved to {file_path}")

def create_example_env_file(file_path: Union[str, Path] = ".env.example") -> None:
    """
    Create an example .env file with all available environment variables.
    
    Args:
        file_path: Path to the example .env file
    """
    lines = [
        "# SubWhisper Environment Variables",
        "# Uncomment and modify as needed",
        "",
        "# Input/Output",
        "# SUBWHISPER_INPUT=/path/to/video.mp4",
        "# SUBWHISPER_OUTPUT=/path/to/subtitles.srt",
        "# SUBWHISPER_FORMAT=srt  # srt, vtt, or ass",
        "",
        "# Processing",
        "# SUBWHISPER_LANGUAGE=en  # ISO 639-1 language code",
        "# SUBWHISPER_WHISPER_MODEL=base  # tiny, base, small, medium, or large",
        "# SUBWHISPER_GPU=true  # true or false",
        "",
        "# Other",
        "# SUBWHISPER_VERBOSE=false  # true or false",
        "# SUBWHISPER_TEMP_DIR=/path/to/temp/dir",
    ]
    
    # Write the example .env file
    with open(file_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    logger.info(f"Example environment file created at {file_path}") 