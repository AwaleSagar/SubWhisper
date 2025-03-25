"""
Translation functionality test script for SubWhisper.

Usage:
    python -m tests.test_translation

This script helps test and evaluate the offline translation functionality.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from src.utils.config import Config
from src.language.translation import Translator, TranslationError
from src.subtitles.generator import Subtitle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("translation-test")

# Define test sentences in different languages
TEST_SENTENCES = {
    "ja": [
        "こんにちは、元気ですか？",
        "私は日本語を勉強しています。",
        "東京は美しい都市です。"
    ],
    "zh": [
        "你好，你好吗？",
        "我正在学习中文。",
        "北京是一个美丽的城市。"
    ],
    "es": [
        "Hola, ¿cómo estás?",
        "Estoy aprendiendo español.",
        "Madrid es una ciudad hermosa."
    ],
    "fr": [
        "Bonjour, comment ça va?",
        "J'apprends le français.",
        "Paris est une belle ville."
    ],
    "de": [
        "Hallo, wie geht es dir?",
        "Ich lerne Deutsch.",
        "Berlin ist eine schöne Stadt."
    ]
}

class MockArgs:
    """Mock arguments for testing."""
    
    def __init__(self, gpu=False, translation_model_dir=None, translation_batch_size=4):
        self.gpu = gpu
        self.translation_model_dir = translation_model_dir
        self.translation_batch_size = translation_batch_size
        self.verbose = True

def test_translation_for_language(translator: Translator, lang_code: str) -> Dict:
    """Test translation for a specific language."""
    print(f"\n===== Testing translation for {lang_code} =====")
    
    if lang_code not in TEST_SENTENCES:
        print(f"No test sentences for {lang_code}, skipping...")
        return {"success": False, "reason": "No test sentences available"}

    sentences = TEST_SENTENCES[lang_code]
    results = {"language": lang_code, "success": True, "translations": []}
    
    try:
        # Single sentence test
        print(f"\nSingle sentence translation test:")
        for sentence in sentences:
            print(f"\nOriginal: {sentence}")
            translated = translator.translate_text(sentence, lang_code)
            print(f"Translated: {translated}")
            results["translations"].append({"original": sentence, "translated": translated})
        
        # Batch translation test
        print(f"\nBatch translation test:")
        batch_translated = translator.translate_batch(sentences, lang_code)
        for i, (original, translated) in enumerate(zip(sentences, batch_translated)):
            print(f"\nOriginal {i+1}: {original}")
            print(f"Translated {i+1}: {translated}")
            
        # Test with subtitle objects
        print(f"\nSubtitle translation test:")
        subtitles = [
            Subtitle(index=1, start=0.0, end=2.0, text=sentences[0]),
            Subtitle(index=2, start=2.0, end=4.0, text=sentences[1]),
            Subtitle(index=3, start=4.0, end=6.0, text=sentences[2])
        ]
        translated_subs = translator.translate_subtitles(subtitles, lang_code)
        for i, sub in enumerate(translated_subs):
            print(f"\nSubtitle {i+1} Original: {sentences[i]}")
            print(f"Subtitle {i+1} Translated: {sub.text}")
            
    except TranslationError as e:
        print(f"Translation error: {str(e)}")
        results["success"] = False
        results["error"] = str(e)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        results["success"] = False
        results["error"] = str(e)
        
    return results

def main():
    """Run translation tests."""
    parser = argparse.ArgumentParser(description="Test SubWhisper translation functionality")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--language", "-l", help="Specific language to test (e.g., 'ja', 'zh', 'es', 'fr')")
    parser.add_argument("--model-dir", help="Directory to store translation models")
    parser.add_argument("--sample-video", help="Run a full test with a sample video file")
    
    args = parser.parse_args()
    
    # Configure model directory
    model_dir = args.model_dir
    if not model_dir:
        # Use default location
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "translation")
    
    # Make sure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    print(f"Using model directory: {model_dir}")
    
    # Setup mock config for testing
    mock_args = MockArgs(
        gpu=args.gpu,
        translation_model_dir=model_dir
    )
    config = Config(mock_args)
    
    # Create translator instance
    translator = Translator(config)
    
    # Define languages to test
    languages_to_test = []
    if args.language:
        if args.language in TEST_SENTENCES:
            languages_to_test.append(args.language)
        else:
            print(f"No test sentences available for language: {args.language}")
            print(f"Available languages: {', '.join(TEST_SENTENCES.keys())}")
            return 1
    else:
        languages_to_test = list(TEST_SENTENCES.keys())
    
    results = {}
    
    # Run tests for each language
    for lang in languages_to_test:
        results[lang] = test_translation_for_language(translator, lang)
    
    # Print summary
    print("\n===== Translation Test Summary =====")
    for lang, result in results.items():
        success_status = "✓ Success" if result["success"] else "✗ Failed"
        print(f"{lang}: {success_status}")
        if not result["success"] and "error" in result:
            print(f"  Error: {result['error']}")
    
    # Test with sample video if specified
    if args.sample_video:
        if not os.path.exists(args.sample_video):
            print(f"Sample video file not found: {args.sample_video}")
            return 1
            
        print(f"\n===== Testing with sample video: {args.sample_video} =====")
        print("This will run the full SubWhisper pipeline including translation.")
        print("Running the main application with the sample video...")
        
        # Import and run main with arguments for the sample video
        from main import main as run_main
        import sys
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        # Set up arguments for the main application
        output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   "output_test_translation.srt")
        sys.argv = [
            "subwhisper",
            "--input", args.sample_video,
            "--output", output_file,
            "--format", "srt",
            "--translate-to-english", "always",
            "--verbose"
        ]
        
        if args.gpu:
            sys.argv.append("--gpu")
            
        try:
            run_main()
            print(f"\nTranslation test with video completed. Output saved to: {output_file}")
        except Exception as e:
            print(f"Error during video translation test: {str(e)}")
            return 1
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
    return 0

if __name__ == "__main__":
    sys.exit(main())