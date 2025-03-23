# Development Guide

This document provides guidelines for developers who want to contribute to the SubWhisper project.

## Code Organization

The project follows a modular architecture to ensure maintainability and extensibility:

```
subwhisper/
├── src/
│   ├── video/          # Video handling
│   ├── audio/          # Audio processing
│   ├── language/       # Language detection
│   ├── subtitles/      # Subtitle generation
│   └── utils/          # Utility functions
```

Each module is responsible for a specific part of the processing pipeline:

1. **Video Module**: Handles video input validation and audio extraction
2. **Audio Module**: Processes audio for speech recognition
3. **Language Module**: Detects spoken language
4. **Subtitles Module**: Generates and formats subtitles
5. **Utils Module**: Provides configuration, logging, and other utilities

## Development Environment Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black isort mypy
   ```

## Coding Standards

We follow these coding standards:

1. **PEP 8** compliant code
2. **Type hints** for all function parameters and return values
3. **Docstrings** for all modules, classes, and functions (Google style)
4. **Unit tests** for all functionality
5. **Logging** instead of print statements

### Code Formatting

We use the following tools to maintain code quality:

- **Black** for code formatting:
  ```bash
  black .
  ```

- **isort** for import sorting:
  ```bash
  isort .
  ```

- **mypy** for type checking:
  ```bash
  mypy src tests
  ```

## Adding New Features

When adding new features:

1. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow the existing architecture**:
   - Place code in the appropriate module
   - Keep files and functions focused on a single responsibility

3. **Write tests** in the `tests/` directory:
   ```python
   # Example test
   def test_new_feature():
       # Setup
       input_data = "test input"
       expected_output = "expected result"
       
       # Execute
       actual_output = your_new_function(input_data)
       
       # Assert
       assert actual_output == expected_output
   ```

4. **Run tests** to ensure your feature works and doesn't break existing functionality:
   ```bash
   pytest
   ```

## Extending the Project

### Adding New Subtitle Formats

To add support for a new subtitle format:

1. Modify `src/subtitles/formatter.py` to add a new formatter class
2. Implement format-specific logic in the new class
3. Update the CLI argument parser in `main.py` to accept the new format option

Example:

```python
class NewFormatFormatter(SubtitleFormatter):
    def format(self, subtitles: List[SubtitleEntry]) -> str:
        # Implementation for the new format
        result = ""
        for entry in subtitles:
            # Format the entry according to the new format
            result += f"...custom formatting..."
        return result
```

### Adding New Language Models

To add support for a new language detection model:

1. Add the model files to `models/language/`
2. Implement a new detector in `src/language/detection.py`
3. Update the language detection factory to use the new model

### Adding New Speech Recognition Models

To add support for a new speech recognition model:

1. Add the model files to `models/speech/`
2. Implement a new recognizer in `src/audio/speech.py`
3. Update the speech recognition factory to use the new model

## Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Update documentation** to reflect changes

3. **Submit a pull request** describing:
   - What was changed
   - Why it was changed
   - How to test the changes

4. **Address review comments** and update your PR as needed

## Debugging Tips

1. Use the `--verbose` flag to enable detailed logging:
   ```bash
   python main.py --input video.mp4 --output subs.srt --verbose
   ```

2. Check the logs in `logs/subwhisper.log`

3. For debugging speech recognition issues, use the `--temp-dir` flag to preserve intermediate files:
   ```bash
   python main.py --input video.mp4 --output subs.srt --temp-dir ./debug_files
   ```

## Common Issues

### FFmpeg-related problems

If you encounter issues with FFmpeg:

1. Ensure FFmpeg is installed and in your PATH
2. Check FFmpeg version (should be 4.0+):
   ```bash
   ffmpeg -version
   ```
3. For custom FFmpeg installations, you can specify the path in the configuration

### Memory issues with large videos

For processing large videos:

1. Use a smaller model with the `--model tiny` option
2. Process videos in segments
3. Increase system swap space if possible

## Performance Optimization

When optimizing for performance:

1. Profile the code to identify bottlenecks:
   ```bash
   python -m cProfile -o profile.pstats main.py --input video.mp4 --output subs.srt
   ```

2. Focus on the most time-consuming parts of the pipeline

3. Consider implementing multiprocessing for independent tasks

4. Use GPU acceleration where applicable 