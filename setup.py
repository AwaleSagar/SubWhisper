#!/usr/bin/env python3
"""
Setup script for SubWhisper package.
"""

import os
from setuptools import setup, find_packages

# Read requirements from requirements.txt if it exists
requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
        # Filter out comments and empty lines
        requirements = [r for r in requirements if r and not r.startswith('#')]
except FileNotFoundError:
    # Define requirements here as a fallback
    requirements = [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "ffmpeg-python>=0.2.0",
        "pydub>=0.25.1",
        "librosa>=0.8.1",
        "moviepy>=1.0.3",
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "transformers>=4.12.0",
        "whisper>=1.0.0",
        "langid>=1.1.6",
        "fasttext>=0.9.2",
        "pysrt>=1.1.2",
        "webvtt-py>=0.4.6",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "loguru>=0.5.3",
        "python-dotenv>=0.19.0",
        "requests>=2.27.0",
    ]

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="subwhisper",
    version="0.1.0",
    description="Audio extraction and automatic subtitle generation from video files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sagar Awale",
    author_email="awalessagar@gmail.com",
    url="https://github.com/AwaleSagar/SubWhisper",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "subwhisper=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 