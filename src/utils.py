# Utility functions for the RAG system.
# Handles logging, configuration, file operations, and helper functions.

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import colorlog
import tiktoken

from exceptions import ConfigurationError, ValidationError


class Logger:
    #Centralized logging configuration for the RAG system

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('logging', {})
        self.setup_logging()

    def setup_logging(self):
        #Setup logging with both console and file handlers
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.get('log_directory', 'logs'))
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler if enabled
        if self.config.get('log_to_file', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'rag_system_{timestamp}.log'

            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Log the log file location
            logger.info(f"Logging to file: {log_file}")


class ConfigManager:
    #Manages configuration loading and validation

    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.validate_config()

    def load_config(self) -> Dict[str, Any]:
        #Load configuration from JSON file
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Config file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Load environment variables for API keys
            config['api_keys'] = {
                'openai': os.getenv('OPENAI_API_KEY')
            }

            return config

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")

    def validate_config(self):
        #Validate configuration parameters
        required_sections = ['models', 'chunking', 'retrieval', 'vector_store', 'output']

        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required config section: {section}")

        # Validate chunking parameters
        chunking = self.config['chunking']
        if chunking['chunk_size_tokens'] <= chunking['overlap_tokens']:
            raise ConfigurationError("Chunk size must be larger than overlap size")

        # # Validate API key
        # if not self.config['api_keys']['openai']:
        #     raise ConfigurationError(
        #         "OPENAI_API_KEY environment variable not set. "
        #         "Please set it with your OpenAI API key."
        #     )

    def get(self, key_path: str, default: Any = None) -> Any:
        #Get configuration value using dot notation (e.g., 'models.embedding_model')
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


class TokenCounter:
    #Handles token counting for different models

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.encoder = self._get_encoder()

    def _get_encoder(self):
        #Get appropriate tokenizer for the model
        try:
            # Map model names to tokenizer encodings
            model_encodings = {
                "gpt-3.5-turbo": "cl100k_base",
                "gpt-4": "cl100k_base",
                "text-davinci-003": "p50k_base",
                "text-davinci-002": "p50k_base",
            }

            encoding_name = model_encodings.get(self.model_name, "cl100k_base")
            return tiktoken.get_encoding(encoding_name)

        except Exception as e:
            logging.warning(f"Failed to load tokenizer for {self.model_name}: {e}")
            # Fallback to default
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        #Count tokens in text
        if not text:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logging.warning(f"Token counting failed: {e}")
            # Fallback: rough estimation
            return len(text.split()) * 1.3  # Approximate conversion

    def truncate_text(self, text: str, max_tokens: int) -> str:
        #Truncate text to fit within token limit
        try:
            tokens = self.encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            return self.encoder.decode(truncated_tokens)

        except Exception as e:
            logging.warning(f"Text truncation failed: {e}")
            # Fallback: word-based truncation
            words = text.split()
            estimated_words = int(max_tokens * 0.75)  # Rough conversion
            return ' '.join(words[:estimated_words])


class FileManager:
    #Handles file operations and directory management

    @staticmethod
    def ensure_directory(path: str) -> Path:
        #Create directory if it doesn't exist
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def get_safe_filename(filename: str) -> str:
        #Generate safe filename by removing invalid characters
        invalid_chars = '<>:"/\\|?*'
        safe_name = filename
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        return safe_name

    @staticmethod
    def save_text_output(content: str, filename: str, output_dir: str):
        #Save text content to file with timestamp
        output_path = FileManager.ensure_directory(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = FileManager.get_safe_filename(filename)

        full_filename = f"{timestamp}_{safe_filename}.txt"
        file_path = output_path / full_filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logging.info(f"Output saved to: {file_path}")
            return file_path

        except Exception as e:
            logging.error(f"Failed to save output: {e}")
            raise


class PerformanceTracker:
    #Tracks performance metrics for the RAG system

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, operation: str):
        #Start timing an operation
        self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        #End timing and return duration
        if operation not in self.start_times:
            logging.warning(f"Timer for '{operation}' was not started")
            return 0.0

        duration = time.time() - self.start_times[operation]

        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration)
        del self.start_times[operation]

        return duration

    def log_metrics(self):
        #Log performance summary
        if not self.metrics:
            logging.info("No performance metrics to report")
            return

        logging.info("=== Performance Metrics ===")
        for operation, durations in self.metrics.items():
            avg_time = sum(durations) / len(durations)
            total_time = sum(durations)
            count = len(durations)

            logging.info(
                f"{operation}: {count} operations, "
                f"avg: {avg_time:.2f}s, total: {total_time:.2f}s"
            )


def validate_input(value: Any, expected_type: type, name: str) -> None:
    #Validate input parameter type
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Invalid {name}: expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def clean_text(text: str) -> str:
    #Clean and normalize text content
    if not text:
        return ""

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Remove non-printable characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

    return text.strip()