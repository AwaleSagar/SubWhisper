"""
Caching utilities for SubWhisper.
"""

import os
import json
import hashlib
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Set, Tuple
import time

logger = logging.getLogger(__name__)

class Cache:
    """Cache for storing processed data."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, max_size_gb: float = 2.0):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached data. If None, a subdirectory in the system temp directory is used.
            max_size_gb: Maximum cache size in gigabytes
        """
        if cache_dir is None:
            # Use a subdirectory in the system temp directory
            self.cache_dir = Path(tempfile.gettempdir()) / "subwhisper_cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of cached data
        self.audio_dir = self.cache_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.transcription_dir = self.cache_dir / "transcription"
        self.transcription_dir.mkdir(exist_ok=True)
        
        self.subtitle_dir = self.cache_dir / "subtitle"
        self.subtitle_dir.mkdir(exist_ok=True)
        
        # Set maximum cache size (in bytes)
        self.max_size = int(max_size_gb * 1024 * 1024 * 1024)
        
        # Initialize index file path
        self.index_path = self.cache_dir / "index.json"
        
        # Load index if it exists
        self.index = self._load_index()
        
        # Clean cache if necessary
        self._clean_cache()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load cache index from disk.
        
        Returns:
            Dictionary mapping cache keys to metadata
        """
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache index: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self.index, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache index: {str(e)}")
    
    def _clean_cache(self) -> None:
        """
        Clean cache if it exceeds the maximum size.
        Removes the oldest entries first.
        """
        # Get current cache size
        total_size = 0
        for entry in self.index.values():
            total_size += entry.get("size", 0)
        
        if total_size <= self.max_size:
            return
        
        logger.info(f"Cache size ({total_size / 1024 / 1024:.2f} MB) exceeds maximum ({self.max_size / 1024 / 1024:.2f} MB), cleaning up...")
        
        # Sort entries by last access time (oldest first)
        entries = sorted(self.index.items(), key=lambda x: x[1].get("last_access", 0))
        
        # Remove entries until cache size is below the limit
        for key, entry in entries:
            if total_size <= self.max_size:
                break
            
            # Remove the entry from disk
            path = Path(entry.get("path", ""))
            if path.exists():
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    
                    # Update total size
                    total_size -= entry.get("size", 0)
                    
                    # Remove entry from index
                    del self.index[key]
                    
                    logger.debug(f"Removed cache entry: {key}")
                except IOError as e:
                    logger.warning(f"Failed to remove cache entry {key}: {str(e)}")
        
        # Save updated index
        self._save_index()
    
    def generate_key(self, data: Dict[str, Any]) -> str:
        """
        Generate a cache key from data.
        
        Args:
            data: Dictionary of data to generate key from
            
        Returns:
            Cache key as a hexadecimal string
        """
        # Convert data to a stable string representation
        data_str = json.dumps(data, sort_keys=True)
        
        # Generate MD5 hash
        hash_md5 = hashlib.md5(data_str.encode("utf-8")).hexdigest()
        
        return hash_md5
    
    def has(self, key: str) -> bool:
        """
        Check if a cache entry exists.
        
        Args:
            key: Cache key
            
        Returns:
            True if the cache entry exists, False otherwise
        """
        if key not in self.index:
            return False
        
        # Check if the cached file exists
        path = Path(self.index[key].get("path", ""))
        if not path.exists():
            # Remove entry from index if the file doesn't exist
            del self.index[key]
            self._save_index()
            return False
        
        return True
    
    def get(self, key: str) -> Optional[Union[str, Path]]:
        """
        Get a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cached file if it exists, None otherwise
        """
        if not self.has(key):
            return None
        
        # Update last access time
        self.index[key]["last_access"] = time.time()
        self._save_index()
        
        return Path(self.index[key]["path"])
    
    def set(self, key: str, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set a cache entry.
        
        Args:
            key: Cache key
            path: Path to the cached file
            metadata: Additional metadata to store with the entry
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Cannot cache non-existent file: {path}")
            return
        
        # Get file size
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
        else:
            logger.warning(f"Cannot determine size of cache entry: {path}")
            size = 0
        
        # Create entry
        entry = {
            "path": str(path),
            "size": size,
            "created": time.time(),
            "last_access": time.time(),
        }
        
        if metadata:
            entry.update(metadata)
        
        # Add entry to index
        self.index[key] = entry
        
        # Save index
        self._save_index()
        
        # Clean cache if necessary
        self._clean_cache()
    
    def audio_path(self, filename: str) -> Path:
        """
        Get the path for a cached audio file.
        
        Args:
            filename: Filename for the audio file
            
        Returns:
            Path to the cached audio file
        """
        return self.audio_dir / filename
    
    def transcription_path(self, filename: str) -> Path:
        """
        Get the path for a cached transcription file.
        
        Args:
            filename: Filename for the transcription file
            
        Returns:
            Path to the cached transcription file
        """
        return self.transcription_dir / filename
    
    def subtitle_path(self, filename: str) -> Path:
        """
        Get the path for a cached subtitle file.
        
        Args:
            filename: Filename for the subtitle file
            
        Returns:
            Path to the cached subtitle file
        """
        return self.subtitle_dir / filename
    
    def clear_audio(self) -> None:
        """Clear all cached audio files."""
        # Remove all files in the audio directory
        for path in self.audio_dir.glob("*"):
            if path.is_file():
                path.unlink()
        
        # Remove corresponding entries from the index
        keys_to_remove = []
        for key, entry in self.index.items():
            path = Path(entry.get("path", ""))
            if path.parent == self.audio_dir:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index[key]
        
        # Save index
        self._save_index()
    
    def clear_transcriptions(self) -> None:
        """Clear all cached transcription files."""
        # Remove all files in the transcription directory
        for path in self.transcription_dir.glob("*"):
            if path.is_file():
                path.unlink()
        
        # Remove corresponding entries from the index
        keys_to_remove = []
        for key, entry in self.index.items():
            path = Path(entry.get("path", ""))
            if path.parent == self.transcription_dir:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index[key]
        
        # Save index
        self._save_index()
    
    def clear_subtitles(self) -> None:
        """Clear all cached subtitle files."""
        # Remove all files in the subtitle directory
        for path in self.subtitle_dir.glob("*"):
            if path.is_file():
                path.unlink()
        
        # Remove corresponding entries from the index
        keys_to_remove = []
        for key, entry in self.index.items():
            path = Path(entry.get("path", ""))
            if path.parent == self.subtitle_dir:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index[key]
        
        # Save index
        self._save_index()
    
    def clear_all(self) -> None:
        """Clear all cached files."""
        self.clear_audio()
        self.clear_transcriptions()
        self.clear_subtitles()
        
        # Also remove the index file
        if self.index_path.exists():
            self.index_path.unlink()
        
        # Reset index
        self.index = {}
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        audio_count = len(list(self.audio_dir.glob("*")))
        audio_size = sum(f.stat().st_size for f in self.audio_dir.glob("*") if f.is_file())
        
        transcription_count = len(list(self.transcription_dir.glob("*")))
        transcription_size = sum(f.stat().st_size for f in self.transcription_dir.glob("*") if f.is_file())
        
        subtitle_count = len(list(self.subtitle_dir.glob("*")))
        subtitle_size = sum(f.stat().st_size for f in self.subtitle_dir.glob("*") if f.is_file())
        
        total_count = audio_count + transcription_count + subtitle_count
        total_size = audio_size + transcription_size + subtitle_size
        
        return {
            "audio_count": audio_count,
            "audio_size": audio_size,
            "transcription_count": transcription_count,
            "transcription_size": transcription_size,
            "subtitle_count": subtitle_count,
            "subtitle_size": subtitle_size,
            "total_count": total_count,
            "total_size": total_size,
            "max_size": self.max_size,
        } 