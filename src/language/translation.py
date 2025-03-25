"""
Translation module for SubWhisper.
Uses T5 models from Google for efficient and accurate translation.
"""

import os
import logging
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
        
    def _get_model_path(self, model_name: str) -> str:
        """Get the path where the model should be stored."""
        base_dir = self.config.translation_model_dir
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, model_name.replace("/", "_"))
        
    def _load_model(self, source_lang: str) -> None:
        """
        Load translation model for the specified source language.
        
        Args:
            source_lang: Source language code (ISO 639-1)
            
        Raises:
            TranslationError: If model loading fails
        """
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
            
            # Skip if already loaded
            if self.translation_model is not None and self.current_lang == source_lang:
                return
            
            # For Japanese we need to use a specialized model as T5 doesn't handle it well
            if source_lang in ["ja", "zh", "ko"]:
                # Use Helsinki models for CJK languages
                language_group_mappings = {
                    "zh": "zh-en",
                    "ja": "jap-en",
                    "ko": "kor-en",
                }
                model_name = f"Helsinki-NLP/opus-mt-{language_group_mappings[source_lang]}"
                
                model_path = self._get_model_path(model_name)
                
                # Check if model exists locally
                if not os.path.exists(model_path):
                    logger.info(f"Translation model not found locally: {model_name}")
                    
                    # Prompt the user for download confirmation
                    download_prompt = f"Translation model for {source_lang} is not installed. Do you want to download it now? (y/n): "
                    user_response = input(download_prompt).strip().lower()
                    
                    if user_response != 'y' and user_response != 'yes':
                        error_message = "Model download cancelled. Cannot translate without model."
                        logger.error(error_message)
                        raise TranslationError(error_message)
                    
                    logger.info(f"Downloading translation model: {model_name}")
                
                logger.info(f"Loading translation model: {model_name}")
                
                # Load model and tokenizer (using Marian for CJK languages)
                self.tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=model_path)
                self.translation_model = MarianMTModel.from_pretrained(model_name, cache_dir=model_path).to(self.device)
                self.current_lang = source_lang
                self.is_t5_model = False
                
            else:
                # For other languages, use standard T5 model
                model_name = "t5-base"  # standard English T5 model
                
                model_path = self._get_model_path(model_name)
                
                # Check if model exists locally
                if not os.path.exists(model_path):
                    logger.info(f"Translation model not found locally: {model_name}")
                    
                    # Prompt the user for download confirmation
                    download_prompt = f"T5 translation model is not installed. Do you want to download it now? (y/n): "
                    user_response = input(download_prompt).strip().lower()
                    
                    if user_response != 'y' and user_response != 'yes':
                        error_message = "Model download cancelled. Cannot translate without model."
                        logger.error(error_message)
                        raise TranslationError(error_message)
                    
                    logger.info(f"Downloading translation model: {model_name}")
                
                logger.info(f"Loading translation model: {model_name}")
                
                # Load model and tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=model_path)
                self.translation_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=model_path).to(self.device)
                self.current_lang = source_lang
                self.is_t5_model = True
            
            logger.info(f"Translation model loaded successfully")
            
        except Exception as e:
            error_message = f"Failed to load translation model: {str(e)}"
            logger.error(error_message)
            raise TranslationError(error_message)
    
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
                    max_length=512,
                    num_beams=4,  # Beam search for better quality
                    early_stopping=True
                )
                translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # For Marian models
                inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
                outputs = self.translation_model.generate(**inputs)
                translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_text
            
        except Exception as e:
            error_message = f"Translation failed: {str(e)}"
            logger.error(error_message)
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
            
            # Process in batches with progress bar
            for i in tqdm(range(0, len(texts), batch_size), desc="Translating", unit="batch"):
                batch = texts[i:i+batch_size]
                # Filter out empty strings to avoid tokenization issues
                batch_texts = [text for text in batch if text.strip()]
                
                if not batch_texts:
                    # If all texts in batch are empty, add them as is
                    results.extend(batch)
                    continue
                
                if self.is_t5_model:
                    # Format for T5 translation
                    prefixed_texts = [f"translate {source_lang} to English: {text}" for text in batch_texts]
                    
                    # Tokenize and translate
                    inputs = self.tokenizer(prefixed_texts, return_tensors="pt", padding=True).to(self.device)
                    outputs = self.translation_model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,  # Beam search for better quality
                        early_stopping=True
                    )
                else:
                    # For Marian models
                    inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.device)
                    outputs = self.translation_model.generate(**inputs)
                
                translated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) 
                                   for output in outputs]
                
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
        model_type = "T5" if source_lang not in ["ja", "zh", "ko"] else "Helsinki-NLP"
        logger.info(f"Translating subtitles from {source_lang} to English using {model_type} model")
        
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