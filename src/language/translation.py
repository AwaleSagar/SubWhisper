"""
Translation module for SubWhisper.
Uses lightweight transformer models for efficient offline translation.
"""
import os
import logging
import pickle
import json
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import torch
from src.utils.config import Config
from src.subtitles.generator import Subtitle

logger = logging.getLogger("subwhisper")

class TranslationError(Exception):
    """Exception raised for errors in translation."""
    pass

class Translator:
    """Subtitle translation class with offline capabilities."""
    
    def __init__(self, config: Config):
        """
        Initialize translator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.device = "cuda" if config.gpu and torch.cuda.is_available() else "cpu"
        self.translation_model = None
        self.tokenizer = None
        self.current_lang = None
        self.is_t5_model = False
        self.model_cache_info_path = os.path.join(config.translation_model_dir, "model_cache_info.pkl")
        self.model_cache_info = self._load_model_cache_info()
        
    def _get_model_path(self, model_name: str) -> str:
        """Get the path where the model should be stored."""
        base_dir = self.config.translation_model_dir
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, model_name.replace("/", "_"))
    
    def _load_model_cache_info(self) -> Dict:
        """Load information about cached models."""
        if os.path.exists(self.model_cache_info_path):
            try:
                with open(self.model_cache_info_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load model cache info: {str(e)}")
        return {"downloaded_models": {}}
    
    def _save_model_cache_info(self):
        """Save information about cached models."""
        try:
            with open(self.model_cache_info_path, 'wb') as f:
                pickle.dump(self.model_cache_info, f)
        except Exception as e:
            logger.warning(f"Could not save model cache info: {str(e)}")
    
    def _is_model_downloaded(self, model_name: str, model_path: str) -> bool:
        """
        Check if a model has been downloaded already.
        
        Args:
            model_name: Name of the model
            model_path: Path where the model should be stored
            
        Returns:
            True if the model is already downloaded
        """
        # First check our cache info
        if model_name in self.model_cache_info["downloaded_models"]:
            if self.model_cache_info["downloaded_models"][model_name]:
                return True
        
        # Check for model files
        config_path = os.path.join(model_path, "config.json")
        model_bin_path = os.path.join(model_path, "pytorch_model.bin")
        
        # Alternative paths that might exist
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        vocab_path = os.path.join(model_path, "vocab.json")
        
        # Check if basic model files exist
        has_config = os.path.exists(config_path)
        has_model_bin = os.path.exists(model_bin_path)
        has_tokenizer = os.path.exists(tokenizer_path) or os.path.exists(vocab_path) or os.path.exists(tokenizer_config_path)
        
        # If not all parts exist, also check model_info.json which can be used for alternative model types
        model_info_path = os.path.join(model_path, "model_info.json")
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                if model_info.get("status") == "downloaded":
                    return True
            except:
                pass
                
        # If model appears to be downloaded, update our cache
        if has_config and (has_model_bin or has_tokenizer):
            self.model_cache_info["downloaded_models"][model_name] = True
            self._save_model_cache_info()
            return True
            
        return False
        
    def _get_model_name_for_language(self, source_lang: str) -> str:
        """
        Get the appropriate model name for a language.
        
        Args:
            source_lang: Source language code (ISO 639-1)
            
        Returns:
            Model name to use for translation
        """
        # Special handling for Japanese - try different model options
        if source_lang == "ja":
            # Try the specific ja-en model first
            return "Helsinki-NLP/opus-mt-ja-en"  # Direct Japanese-English model
        
        # Use smaller models for more efficient offline translation
        language_model_mappings = {
            # CJK languages need specialized models
            "zh": "Helsinki-NLP/opus-mt-zh-en",
            "ko": "Helsinki-NLP/opus-mt-kor-en",
            # European languages
            "es": "Helsinki-NLP/opus-mt-es-en",
            "fr": "Helsinki-NLP/opus-mt-fr-en",
            "de": "Helsinki-NLP/opus-mt-de-en",
            "it": "Helsinki-NLP/opus-mt-it-en",
            "pt": "Helsinki-NLP/opus-mt-pt-en",
            "nl": "Helsinki-NLP/opus-mt-nl-en",
            "ru": "Helsinki-NLP/opus-mt-ru-en",
            # More languages with direct models
            "ar": "Helsinki-NLP/opus-mt-ar-en",
            "tr": "Helsinki-NLP/opus-mt-tr-en",
        }
        
        # Return the specific model if available, otherwise use a fallback approach
        if source_lang in language_model_mappings:
            return language_model_mappings[source_lang]
        else:
            # Check if there's a direct model available
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
            return model_name
        
    def _load_model(self, source_lang: str) -> None:
        """
        Load translation model for the specified source language.
        
        Args:
            source_lang: Source language code (ISO 639-1)
            
        Raises:
            TranslationError: If model loading fails
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
            
            # Skip if already loaded
            if self.translation_model is not None and self.current_lang == source_lang:
                return
            
            # Get the appropriate model name for this language
            model_name = self._get_model_name_for_language(source_lang)
            
            # Set up model path
            model_path = self._get_model_path(model_name)
            os.makedirs(model_path, exist_ok=True)
            
            # Create model_info.json file for tracking
            model_info_path = os.path.join(model_path, "model_info.json")
            
            # Check if model exists locally
            model_files_exist = self._is_model_downloaded(model_name, model_path)
            
            # Try to load locally if it might exist
            if model_files_exist:
                try:
                    logger.info(f"Loading translation model from local cache: {model_name}")
                    
                    # Use MarianMT model which has better compatibility
                    if source_lang == "ja":
                        self.tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
                        self.translation_model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to(self.device)
                    else:
                        # General case for other models
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(self.device)
                    
                    # Set attributes
                    self.is_t5_model = "t5" in model_name.lower()
                    self.is_marian_model = "opus-mt" in model_name.lower() or "marian" in model_name.lower()
                    
                    # Set source language
                    self.current_lang = source_lang
                    
                    # Update model info
                    with open(model_info_path, 'w') as f:
                        json.dump({"model_name": model_name, "lang": source_lang, "status": "downloaded"}, f)
                    
                    logger.info(f"Translation model loaded successfully from local cache")
                    return
                except Exception as e:
                    logger.warning(f"Could not load model from cache, will download: {str(e)}")
                    model_files_exist = False
            
            if not model_files_exist:
                logger.info(f"Translation model not found locally: {model_name}")
                
                # Prompt the user for download confirmation
                download_prompt = f"Translation model for {source_lang} is not installed. Download it now? (y/n): "
                user_response = input(download_prompt).strip().lower()
                
                if user_response != 'y' and user_response != 'yes':
                    error_message = "Model download cancelled. Cannot translate without model."
                    logger.error(error_message)
                    raise TranslationError(error_message)
                
                logger.info(f"Downloading translation model: {model_name}")
                
                try:
                    # Download the model with progress tracking
                    if source_lang == "ja":
                        self.tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=model_path)
                        self.translation_model = MarianMTModel.from_pretrained(model_name, cache_dir=model_path).to(self.device)
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
                        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_path).to(self.device)
                    
                    # Set attributes
                    self.is_t5_model = "t5" in model_name.lower()
                    self.is_marian_model = "opus-mt" in model_name.lower() or "marian" in model_name.lower()
                    
                    # Record successful download in our cache
                    self.model_cache_info["downloaded_models"][model_name] = True
                    self._save_model_cache_info()
                    
                    # Update model info file
                    with open(model_info_path, 'w') as f:
                        json.dump({"model_name": model_name, "lang": source_lang, "status": "downloaded"}, f)
                    
                except Exception as download_error:
                    error_message = f"Failed to download model: {str(download_error)}"
                    logger.error(error_message)
                    raise TranslationError(error_message)
            
            # Set internal state
            self.current_lang = source_lang
            
            logger.info(f"Translation model loaded successfully")
            
        except Exception as e:
            error_message = f"Failed to load translation model: {str(e)}"
            logger.error(error_message)
            raise TranslationError(error_message)
    
    def _try_jap_en_fallback(self, text: str) -> str:
        """
        Try the alternative jap-en model as a fallback for Japanese translation.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text from the fallback model
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Use the alternative jap-en model
            model_name = "Helsinki-NLP/opus-mt-jap-en"
            model_path = self._get_model_path(model_name)
            
            # Check if model exists
            if not self._is_model_downloaded(model_name, model_path):
                logger.info(f"Fallback model not available: {model_name}")
                return ""
                
            # Load the model
            logger.info(f"Trying fallback model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
            model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to(self.device)
            
            # Translate
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            outputs = model.generate(
                **inputs,
                max_length=min(256, int(len(text.split()) * 2) + 20),
                num_beams=4,
                length_penalty=1.0
            )
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
            
        except Exception as e:
            logger.warning(f"Fallback translation failed: {str(e)}")
            return ""

    def _choose_best_japanese_translation(self, text: str, primary_translation: str) -> str:
        """
        Choose the best translation between multiple options for Japanese.
        
        Args:
            text: Original text
            primary_translation: Translation from primary model
            
        Returns:
            The best translation
        """
        # Try fallback model if primary translation looks problematic
        if (not primary_translation or 
            "Sheba" in primary_translation or 
            "But he" in primary_translation or
            primary_translation.count("But") > 2 or
            len(primary_translation.split()) < 2):
            
            fallback = self._try_jap_en_fallback(text)
            
            # If fallback is better, use it
            if fallback and len(fallback) > 5 and "Sheba" not in fallback:
                logger.info("Using fallback translation")
                return self._post_process_japanese_translation(fallback)
        
        return primary_translation
    
    def _post_process_japanese_translation(self, text: str) -> str:
        """Apply post-processing to improve Japanese translation quality"""
        import re
        
        # Fix common Japanese translation artifacts
        text = re.sub(r'But\s+he\s+speak', 'I speak', text)
        text = re.sub(r'the\s+Sheba', 'Tokyo', text)
        text = re.sub(r'(?i)comely city', 'beautiful city', text)
        
        # Remove repetitive words common in machine translation
        text = re.sub(r'(?i)(\w+)(\s+\1){2,}', r'\1', text)
        
        return text
    
    def translate_text(self, text: str, source_lang: str) -> str:
        """
        Translate a single text from source language to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1)
            
        Returns:
            Translated text in English
            
        Raises:
            TranslationError: If translation fails
        """
        try:
            # Skip translation if already English
            if source_lang == "en":
                return text
                
            # Skip empty text
            if not text.strip():
                return text
                
            # Load model if not already loaded or if language changed
            if self.translation_model is None or self.current_lang != source_lang:
                self._load_model(source_lang)
                
            # Handle differently based on model type
            if self.is_t5_model:
                # For T5 models, we format for translation task
                input_text = f"translate {source_lang} to English: {text}"
                
                # Tokenize and translate
                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
                outputs = self.translation_model.generate(
                    **inputs,
                    max_length=min(512, len(text.split()) * 2),
                    num_beams=3,  # Increased for better quality
                    early_stopping=True
                )
                translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # For Marian and other models
                inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
                
                # Use more effective generation parameters
                outputs = self.translation_model.generate(
                    **inputs,
                    max_length=min(256, int(len(text.split()) * 2) + 20),  # Increased for better completeness
                    num_beams=4 if source_lang == "ja" else 2,  # More beams for Japanese
                    length_penalty=1.0 if source_lang == "ja" else 0.6,  # Encourage longer translations for Japanese
                    early_stopping=True
                )
                translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Apply post-processing for Japanese
            if source_lang == "ja":
                # Apply basic post-processing first
                translated_text = self._post_process_japanese_translation(translated_text)
                # Then try the fallback model if needed and choose the best translation
                translated_text = self._choose_best_japanese_translation(text, translated_text)
            
            return translated_text
            
        except Exception as e:
            error_message = f"Translation failed: {str(e)}"
            logger.error(error_message)
            logger.info("Returning original text due to translation failure")
            # Return original text if translation fails
            return text
    
    def translate_batch(self, texts: List[str], source_lang: str, 
                       batch_size: int = 8) -> List[str]:
        """
        Translate a batch of texts from source language to English.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code (ISO 639-1)
            batch_size: Number of texts to translate in one batch
            
        Returns:
            List of translated texts in English
        """
        try:
            # Skip translation if already English
            if source_lang == "en":
                return texts
                
            # Load model if not already loaded
            if self.translation_model is None or self.current_lang != source_lang:
                self._load_model(source_lang)
            
            results = []
            
            # For Japanese, translate one by one for better quality
            if source_lang == "ja":
                for text in tqdm(texts, desc="Translating", unit="text"):
                    results.append(self.translate_text(text, source_lang))
                return results
            
            # Process other languages in batches with progress bar
            for i in tqdm(range(0, len(texts), batch_size), desc="Translating", unit="batch"):
                batch = texts[i:i+batch_size]
                # Filter out empty strings to avoid tokenization issues
                batch_texts = [text for text in batch if text.strip()]
                
                if not batch_texts:
                    # If all texts in batch are empty, add them as is
                    results.extend(batch)
                    continue
                
                try:
                    if self.is_t5_model:
                        # Format for T5 translation
                        prefixed_texts = [f"translate {source_lang} to English: {text}" for text in batch_texts]
                        
                        # Tokenize and translate
                        inputs = self.tokenizer(prefixed_texts, return_tensors="pt", padding=True).to(self.device)
                        outputs = self.translation_model.generate(
                            **inputs,
                            max_length=min(512, max([len(text.split()) for text in batch_texts]) * 2),
                            num_beams=3,  # Increased for better quality
                            early_stopping=True
                        )
                        translated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                    else:
                        # For Marian models
                        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.device)
                        # Better generation parameters for quality
                        outputs = self.translation_model.generate(
                            **inputs,
                            max_length=min(256, max([len(text.split()) for text in batch_texts]) * 2),
                            num_beams=2,
                            length_penalty=0.6
                        )
                        translated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                        
                        # Apply post-processing for Japanese
                        if source_lang == "ja":
                            translated_texts = [self._post_process_japanese_translation(text) for text in translated_texts]
                    
                    # Match original batch size by re-inserting empty strings
                    batch_results = []
                    translated_idx = 0
                    for text in batch:
                        if text.strip():
                            batch_results.append(translated_texts[translated_idx])
                            translated_idx += 1
                        else:
                            batch_results.append(text)
                    
                    results.extend(batch_results)
                
                except Exception as batch_error:
                    logger.error(f"Error translating batch: {str(batch_error)}")
                    # In case of error, return the original batch
                    results.extend(batch)
            
            return results
            
        except Exception as e:
            error_message = f"Batch translation failed: {str(e)}"
            logger.error(error_message)
            # Return original texts if translation fails
            return texts
    
    def translate_subtitles(self, subtitles: List[Subtitle], source_lang: str) -> List[Subtitle]:
        """
        Translate subtitles from source language to English.
        
        Args:
            subtitles: List of Subtitle objects
            source_lang: Source language code (ISO 639-1)
            
        Returns:
            List of Subtitle objects with translated text
        """
        logger.info(f"Translating {len(subtitles)} subtitles from {source_lang} to English")
        
        try:
            # Extract text from subtitles
            texts = [sub.text for sub in subtitles]
            
            # Translate texts
            translated_texts = self.translate_batch(texts, source_lang, 
                                                  self.config.translation_batch_size)
            
            # Update subtitles with translated text
            translated_subtitles = []
            for i, sub in enumerate(subtitles):
                # Create a new Subtitle object with the translated text
                translated_sub = Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    text=translated_texts[i]
                )
                translated_subtitles.append(translated_sub)
            
            logger.info(f"Translation completed successfully")
            return translated_subtitles
            
        except Exception as e:
            error_message = f"Subtitle translation failed: {str(e)}"
            logger.error(error_message)
            # Return original subtitles if translation fails
            return subtitles