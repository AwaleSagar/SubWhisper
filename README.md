# SubWhisper

A standalone tool for audio extraction and automatic subtitle generation from video files. This tool can automatically detect language, transcribe speech, and generate subtitle files without relying on external APIs.

## Features

- **Video Input Support**: Process videos in common formats (MP4, AVI, MKV, etc.)
- **Audio Extraction**: Automatically extract audio tracks from video files
- **Language Detection**: Automatic detection of spoken language
- **Speech Recognition**: Convert speech to text with high accuracy
- **Subtitle Generation**: Generate subtitles with precise timestamps
- **Multiple Format Support**: Output subtitles in various formats (SRT, WebVTT, ASS)
- **Standalone Operation**: Works completely offline without external API dependencies
- **GPU Acceleration**: Optional GPU support for faster processing
- **On-demand Model Download**: Prompts for downloading Whisper models when needed

## Requirements

- Python 3.8+
- FFmpeg (for audio extraction)
- GPU (optional, for faster processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_video_tts.git
   cd ai_video_tts
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   - **Linux**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Quick Start

Generate subtitles for a video file:

```bash
python main.py --input video.mp4 --output subtitles.srt
```

Specify a language (if known) for better accuracy:

```bash
python main.py --input video.mp4 --output subtitles.srt --language en
```

Choose a specific Whisper model size (tiny, base, small, medium, large):

```bash
python main.py --input video.mp4 --output subtitles.srt --whisper-model medium
```

Use GPU acceleration for faster processing:

```bash
python main.py --input video.mp4 --output subtitles.srt --whisper-model medium --gpu
```

For more options and examples, see the [Usage Documentation](docs/usage.md).

## Project Structure

```
ai_video_tts/
├── docs/
│   ├── usage.md                 # Detailed usage documentation
│   └── development.md           # Development guidelines
├── models/                      # Pre-trained models directory
├── src/
│   ├── audio/                   # Audio processing modules
│   ├── language/                # Language detection modules
│   ├── subtitles/               # Subtitle generation modules
│   ├── video/                   # Video processing modules
│   └── utils/                   # Utility modules
├── tests/                       # Test suite
├── main.py                      # Main application
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## How It Works

1. **Video Input**: Validates the video file format and existence
2. **Audio Extraction**: Extracts audio track from the video
3. **Language Detection**: Identifies the spoken language
4. **Speech Recognition**: Transcribes speech to text
5. **Subtitle Generation**: Creates properly timed subtitle entries
6. **Format Conversion**: Outputs subtitles in the requested format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FFmpeg](https://ffmpeg.org/) - For audio/video processing
- [Whisper](https://github.com/openai/whisper) - For speech recognition
- [FastText](https://fasttext.cc/) - For language identification
- [Pydub](https://github.com/jiaaro/pydub) - For audio processing
- [PyTorch](https://pytorch.org/) - For deep learning models 