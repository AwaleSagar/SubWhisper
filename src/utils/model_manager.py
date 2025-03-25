"""
Model management utilities for SubWhisper.
"""

import os
import json
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Check if optional dependencies are available
FASTTEXT_AVAILABLE = False
GENSIM_AVAILABLE = False

# Import optional dependencies with proper type checking
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    logger.debug("Native FastText is not installed. Will try to use gensim implementation.")
    try:
        import gensim
        import gensim.models  # Pre-import to avoid unresolved import error later
        GENSIM_AVAILABLE = True
        logger.info("Using gensim's FastText implementation for language detection.")
    except ImportError:
        logger.warning("Neither FastText nor gensim is installed. Language detection features may be limited.")

# Model URLs and expected checksums
MODEL_CONFIGS = {
    "whisper": {
        "tiny": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
            "size": 75000000,  # ~75MB
            "md5": "d3dd57d32accea0b295c96e26691aa14",
        },
        "base": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.en.pt",
            "size": 150000000,  # ~150MB
            "md5": "ed3a0b6b1c0edf879ad9b11b1af5a0e6",
        },
        "small": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.en.pt",
            "size": 500000000,  # ~500MB
            "md5": "9ecf779972d90ba49c06d968637d720d",
        },
        "medium": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.en.pt",
            "size": 1500000000,  # ~1.5GB
            "md5": "345ae4da62f9b3d59415adc60127b97c",
        },
        "large": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0575f80fca842/large.pt",
            "size": 3000000000,  # ~3GB
            "md5": "e4b87e7e0bf463eb8e6956e646f1e277",
        },
    },
    "language": {
        "fasttext": {
            "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            "size": 131000000,  # ~131MB
            "md5": "7e69ec5451bc261275e5e38408472270",
        }
    },
    "translation": {
        # We don't predefine translation models since they depend on the language pair
    }
}

def get_model_dir(model_type: str) -> Path:
    """
    Get the directory for a specific model type.
    
    Args:
        model_type: The type of model (e.g., 'whisper', 'language')
        
    Returns:
        Path to the model directory
    """
    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models")))
    model_dir = base_dir / model_type
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    return model_dir

def get_model_path(model_type: str, model_name: str) -> Path:
    """
    Get the path for a specific model.
    
    Args:
        model_type: The type of model (e.g., 'whisper', 'language')
        model_name: The name of the model (e.g., 'tiny', 'base', 'fasttext')
        
    Returns:
        Path to the model file
    """
    model_dir = get_model_dir(model_type)
    
    if model_type == "whisper":
        if model_name in ["tiny", "base", "small", "medium"]:
            return model_dir / f"{model_name}.en.pt"
        else:
            return model_dir / f"{model_name}.pt"
    elif model_type == "language" and model_name == "fasttext":
        return model_dir / "lid.176.bin"
    else:
        return model_dir / f"{model_name}.bin"

def check_model_exists(model_type: str, model_name: str) -> bool:
    """
    Check if a specific model exists locally.
    
    Args:
        model_type: The type of model (e.g., 'whisper', 'language')
        model_name: The name of the model (e.g., 'tiny', 'base', 'fasttext')
        
    Returns:
        True if the model exists locally, False otherwise
    """
    model_path = get_model_path(model_type, model_name)
    return model_path.exists()

def calculate_md5(file_path: Path) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash as a hexadecimal string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_model(model_type: str, model_name: str, force: bool = False, no_prompt: bool = False) -> Tuple[bool, str]:
    """
    Download a model if it doesn't exist locally or if force is True.
    
    Args:
        model_type: The type of model (e.g., 'whisper', 'language')
        model_name: The name of the model (e.g., 'tiny', 'base', 'fasttext')
        force: If True, download the model even if it exists locally
        no_prompt: If True, skip the user confirmation prompt
        
    Returns:
        Tuple containing (success: bool, message: str)
    """
    if model_type not in MODEL_CONFIGS:
        return False, f"Unknown model type: {model_type}"
    
    if model_name not in MODEL_CONFIGS[model_type]:
        return False, f"Unknown model name: {model_name} for type: {model_type}"
    
    model_path = get_model_path(model_type, model_name)
    
    # Check if model exists and is valid
    if not force and model_path.exists():
        logger.info(f"Model already exists: {model_path}")
        
        # Optionally validate the model
        try:
            calculated_md5 = calculate_md5(model_path)
            expected_md5 = MODEL_CONFIGS[model_type][model_name]["md5"]
            
            if calculated_md5 == expected_md5 or calculated_md5.startswith(expected_md5):
                logger.info(f"Model validation successful for {model_type}/{model_name}")
                return True, f"Model already exists: {model_path}"
            else:
                logger.warning(f"Model validation failed for {model_type}/{model_name}")
                # Continue with download to replace invalid model
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            # Continue with download to be safe
    
    # Get model URL and expected size
    url = MODEL_CONFIGS[model_type][model_name]["url"]
    expected_size = MODEL_CONFIGS[model_type][model_name]["size"]
    expected_md5 = MODEL_CONFIGS[model_type][model_name]["md5"]
    
    logger.info(f"Downloading {model_type}/{model_name} model from {url}")
    
    # Create temporary file for download
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Download the model
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {model_name}") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Validate downloaded model
        calculated_md5 = calculate_md5(Path(temp_path))
        if calculated_md5 != expected_md5 and not calculated_md5.startswith(expected_md5):
            return False, f"Model validation failed: MD5 mismatch for {model_type}/{model_name}"
        
        # Move the model to the final location
        model_dir = get_model_dir(model_type)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.move(temp_path, model_path)
        
        logger.info(f"Model downloaded successfully: {model_path}")
        return True, f"Model downloaded successfully: {model_path}"
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        logger.error(f"Error downloading model: {str(e)}")
        return False, f"Error downloading model: {str(e)}"

def load_model(model_type: str, model_name: str, auto_download: bool = True, no_prompt: bool = False) -> Tuple[bool, Any, str]:
    """
    Load a model, downloading it if necessary.
    
    Args:
        model_type: The type of model (e.g., 'whisper', 'language')
        model_name: The name of the model (e.g., 'tiny', 'base', 'fasttext')
        auto_download: If True, automatically download the model if it doesn't exist
        no_prompt: If True, skip the user confirmation prompt
        
    Returns:
        Tuple containing (success: bool, model: Any, message: str)
    """
    if not check_model_exists(model_type, model_name):
        if not auto_download:
            return False, None, f"Model does not exist: {model_type}/{model_name}"
        
        # Ask user for confirmation before downloading if prompt is not disabled
        if not no_prompt:
            print(f"\nModel '{model_name}' for {model_type} is not installed.")
            print(f"Size: ~{MODEL_CONFIGS[model_type][model_name]['size'] // 1000000} MB")
            user_input = input("Do you want to download it now? (y/n): ")
            
            if user_input.lower() not in ["y", "yes"]:
                return False, None, "Model download cancelled by user"
        
        success, message = download_model(model_type, model_name, no_prompt=no_prompt)
        if not success:
            return False, None, message
    
    # Load the model
    model_path = get_model_path(model_type, model_name)
    
    try:
        if model_type == "whisper":
            # Return None for whisper models - they will be loaded by the whisper library
            return True, None, f"Whisper model path: {model_path}"
        
        elif model_type == "language" and model_name == "fasttext":
            # Try to load with fasttext first, fallback to gensim
            if FASTTEXT_AVAILABLE:
                model = fasttext.load_model(str(model_path))
                return True, model, f"FastText model loaded: {model_path}"
            elif GENSIM_AVAILABLE:
                # Gensim doesn't directly support Facebook's .bin format, so we'll return None
                # The actual loading will be handled by the language detection module
                logger.info(f"Using gensim instead of native fasttext for language detection")
                return True, None, f"Using gensim for language detection"
            else:
                return False, None, "Neither FastText nor gensim is available"
                
        else:
            return False, None, f"Unknown model type/name combination: {model_type}/{model_name}"
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False, None, f"Error loading model: {str(e)}"

def get_translation_model_path(config, lang_code):
    """
    Get the path for a translation model.
    
    Args:
        config: Application configuration
        lang_code: Language code for the source language
        
    Returns:
        Path where the translation model should be stored
    """
    base_dir = config.translation_model_dir
    os.makedirs(base_dir, exist_ok=True)
    
    # Handle special language codes
    language_group_mappings = {
        "zh": "zh-en",
        "ja": "jap-en",
        "ko": "kor-en",
    }
    
    model_name = language_group_mappings.get(lang_code, f"{lang_code}-en")
    return os.path.join(base_dir, f"opus-mt-{model_name}") 