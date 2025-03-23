#!/usr/bin/env python3
"""
Setup script for SubWhisper package.
"""

import os
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Filter out comments and empty lines
requirements = [r for r in requirements if r and not r.startswith('#')]

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
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    license="MIT",
) 