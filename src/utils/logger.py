"""
Logging functionality for SubWhisper.
"""

import os
import sys
import logging
from pathlib import Path

def setup_logger(verbose=False):
    """
    Set up and configure the logger.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("subwhisper")
    
    # Set level based on verbosity
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set formatter for handler
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Optionally add file handler for persistent logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "subwhisper.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger 