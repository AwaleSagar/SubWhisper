#!/usr/bin/env python3
"""
Example demonstrating how to use the progress tracking functionality.
"""

import os
import sys
import time
import random
import argparse
from typing import Dict, List, Any

# Add the parent directory to the sys.path to import the library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.progress import track_progress, track_function, ProgressTracker


def simulate_processing(data_size: int, progress_tracker: ProgressTracker) -> Dict[str, Any]:
    """
    Simulate a data processing operation.
    
    Args:
        data_size: Size of data to process
        progress_tracker: Progress tracker to update
        
    Returns:
        Dictionary with processing results
    """
    # Simulate processing in chunks
    chunk_size = max(1, data_size // 100)
    total_chunks = (data_size + chunk_size - 1) // chunk_size
    
    progress_tracker.set_description(f"Processing {data_size} items")
    
    # Process each chunk
    processed_data = []
    for i in range(0, data_size, chunk_size):
        # Calculate actual chunk size (last chunk might be smaller)
        current_chunk_size = min(chunk_size, data_size - i)
        
        # Simulate processing time
        time.sleep(0.05 + random.random() * 0.05)
        
        # Add processed data
        processed_data.append(f"Chunk {i//chunk_size + 1}/{total_chunks}")
        
        # Update progress
        progress_tracker.update(1)
        
        # Set postfix with additional information
        progress_tracker.set_postfix(
            chunk=f"{i//chunk_size + 1}/{total_chunks}",
            items_processed=i + current_chunk_size
        )
    
    # Return results
    return {
        "processed_data": processed_data,
        "total_items": data_size,
        "total_chunks": total_chunks
    }


@track_function(name="text_analysis", description="Analyzing text", unit="files")
def analyze_text(files: List[str], progress_tracker: ProgressTracker) -> Dict[str, List[str]]:
    """
    Simulate text analysis on multiple files.
    
    Args:
        files: List of file paths to analyze
        progress_tracker: Progress tracker to update
        
    Returns:
        Dictionary with analysis results
    """
    # Update total to match number of files
    progress_tracker.total = len(files)
    progress_tracker.progress_bar.reset(total=len(files))
    
    # Track results
    results = {"analyzed_files": []}
    
    # Process each file
    for i, file in enumerate(files):
        # Simulate analysis time
        time.sleep(0.2 + random.random() * 0.3)
        
        # Add to results
        results["analyzed_files"].append(file)
        
        # Update progress
        progress_tracker.update(1)
        progress_tracker.set_postfix(current_file=file)
    
    return results


def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="Progress tracking example")
    parser.add_argument("--items", "-i", type=int, default=100, 
                        help="Number of items to process")
    parser.add_argument("--files", "-f", type=int, default=5,
                        help="Number of files to analyze")
    args = parser.parse_args()
    
    print("\n=== Basic Progress Tracking Example ===\n")
    
    # Create a progress tracker for data processing
    with track_progress("data_processing", 100, "Processing data", "chunks") as tracker:
        # Simulate processing data
        results = simulate_processing(args.items, tracker)
    
    print(f"\nProcessed {len(results['processed_data'])} chunks of data")
    
    print("\n=== Function Decorator Example ===\n")
    
    # Create a list of dummy files
    dummy_files = [f"document_{i}.txt" for i in range(1, args.files + 1)]
    
    # Analyze text using the decorated function
    analysis_results = analyze_text(dummy_files)
    
    print(f"\nAnalyzed {len(analysis_results['analyzed_files'])} files")
    

if __name__ == "__main__":
    main() 