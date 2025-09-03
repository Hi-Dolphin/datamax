"""Cleaner CLI Class

Provides object-oriented interface for cleaner operations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger

from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization


class CleanerCLI:
    """Object-oriented interface for text cleaning operations.

    Provides programmatic access to text cleaning functionality
    that can be used by other applications or scripts.
    """

    def __init__(self, verbose: bool = False):
        """Initialize cleaner CLI.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

        # Configure logging
        if verbose:
            logger.remove()
            logger.add(
                lambda msg: print(msg, end=''),
                level="DEBUG",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
            )

    def clean_abnormal(self, text: str) -> Dict[str, Any]:
        """Clean abnormal characters and normalize text.

        Args:
            text: Input text to clean

        Returns:
            Dictionary with cleaned text
        """
        try:
            if self.verbose:
                logger.info("Performing abnormal text cleaning...")

            cleaner = AbnormalCleaner(text)
            result = cleaner.to_clean()

            if self.verbose:
                logger.info("Abnormal cleaning completed successfully")

            return result
        except Exception as e:
            logger.error(f"Abnormal cleaning failed: {str(e)}")
            raise

    def clean_filter(self,
                    text: str,
                    filter_threshold: float = 0.6,
                    min_chars: int = 30,
                    max_chars: int = 500000,
                    numeric_threshold: float = 0.6) -> Dict[str, Any]:
        """Filter text by quality metrics.

        Args:
            text: Input text to filter
            filter_threshold: Word repetition threshold (0.0-1.0)
            min_chars: Minimum character count
            max_chars: Maximum character count
            numeric_threshold: Numeric content threshold (0.0-1.0)

        Returns:
            Dictionary with filtered text
        """
        try:
            if self.verbose:
                logger.info("Performing text filtering...")

            # Apply abnormal cleaning first
            cleaner = AbnormalCleaner(text)
            cleaned = cleaner.no_html_clean()

            if isinstance(cleaned, dict) and 'text' in cleaned and cleaned['text']:
                filter_obj = TextFilter(cleaned['text'])
                filter_obj.filter_by_word_repetition(filter_threshold)
                filter_obj.filter_by_char_count(min_chars, max_chars)
                filter_obj.filter_by_numeric_content(numeric_threshold)
                result = filter_obj.to_filter()
            else:
                result = cleaned

            if self.verbose:
                logger.info("Text filtering completed successfully")

            return result
        except Exception as e:
            logger.error(f"Text filtering failed: {str(e)}")
            raise

    def clean_privacy(self, text: str) -> Dict[str, Any]:
        """Remove sensitive information from text.

        Args:
            text: Input text to desensitize

        Returns:
            Dictionary with cleaned text
        """
        try:
            if self.verbose:
                logger.info("Performing privacy desensitization...")

            privacy = PrivacyDesensitization(text)
            result = privacy.to_private()

            if self.verbose:
                logger.info("Privacy desensitization completed successfully")

            return result
        except Exception as e:
            logger.error(f"Privacy desensitization failed: {str(e)}")
            raise

    def clean_full(self, text: str, **kwargs) -> Dict[str, Any]:
        """Perform complete cleaning pipeline.

        Args:
            text: Input text to clean
            **kwargs: Filtering parameters

        Returns:
            Dictionary with fully cleaned text
        """
        try:
            if self.verbose:
                logger.info("Performing full cleaning pipeline...")

            # Extract filtering parameters
            filter_threshold = kwargs.get('filter_threshold', 0.6)
            min_chars = kwargs.get('min_chars', 30)
            max_chars = kwargs.get('max_chars', 500000)
            numeric_threshold = kwargs.get('numeric_threshold', 0.6)

            # Step 1: Abnormal cleaning
            cleaner = AbnormalCleaner(text)
            cleaned = cleaner.to_clean()

            if not (isinstance(cleaned, dict) and 'text' in cleaned and cleaned['text']):
                return cleaned

            # Step 2: Filtering
            filter_obj = TextFilter(cleaned['text'])
            filtered = filter_obj.to_filter()

            if not (isinstance(filtered, dict) and 'text' in filtered and filtered['text']):
                return filtered

            # Step 3: Privacy desensitization
            privacy = PrivacyDesensitization(filtered['text'])
            result = privacy.to_private()

            if self.verbose:
                logger.info("Full cleaning pipeline completed successfully")

            return result
        except Exception as e:
            logger.error(f"Full cleaning failed: {str(e)}")
            raise

    def clean_no_html(self, text: str) -> Dict[str, Any]:
        """Perform basic cleaning without HTML removal.

        Args:
            text: Input text to clean

        Returns:
            Dictionary with cleaned text
        """
        try:
            if self.verbose:
                logger.info("Performing no-HTML cleaning...")

            cleaner = AbnormalCleaner(text)
            result = cleaner.no_html_clean()

            if self.verbose:
                logger.info("No-HTML cleaning completed successfully")

            return result
        except Exception as e:
            logger.error(f"No-HTML cleaning failed: {str(e)}")
            raise

    def clean_file(self,
                   input_file: Union[str, Path],
                   output_file: Optional[Union[str, Path]] = None,
                   mode: str = 'full',
                   **kwargs) -> str:
        """Clean text from file and save result.

        Args:
            input_file: Input file path
            output_file: Output file path (optional)
            mode: Cleaning mode
            **kwargs: Additional cleaning parameters

        Returns:
            Path to output file
        """
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file '{input_file}' not found")

            if self.verbose:
                logger.info(f"Reading input file: {input_file}")

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Perform cleaning
            if mode == 'full':
                result = self.clean_full(text, **kwargs)
            elif mode == 'abnormal':
                result = self.clean_abnormal(text)
            elif mode == 'filter':
                result = self.clean_filter(text, **kwargs)
            elif mode == 'privacy':
                result = self.clean_privacy(text)
            elif mode == 'no_html':
                result = self.clean_no_html(text)
            else:
                raise ValueError(f"Unknown cleaning mode: {mode}")

            # Determine output path
            if output_file:
                output_path = Path(output_file)
            else:
                output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(result, dict) and 'text' in result:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result['text'])

                if self.verbose:
                    logger.info(f"Cleaned text saved to: {output_path}")

                return str(output_path)
            else:
                raise ValueError("Cleaning returned empty result")

        except Exception as e:
            logger.error(f"File cleaning failed: {str(e)}")
            raise

    def clean_stdin(self, mode: str = 'full', **kwargs) -> str:
        """Clean text from standard input.

        Args:
            mode: Cleaning mode
            **kwargs: Additional cleaning parameters

        Returns:
            Cleaned text
        """
        try:
            if self.verbose:
                logger.info("Reading from stdin...")

            text = sys.stdin.read()
            if not text.strip():
                raise ValueError("No input received from stdin")

            # Perform cleaning
            if mode == 'full':
                result = self.clean_full(text, **kwargs)
            elif mode == 'abnormal':
                result = self.clean_abnormal(text)
            elif mode == 'filter':
                result = self.clean_filter(text, **kwargs)
            elif mode == 'privacy':
                result = self.clean_privacy(text)
            elif mode == 'no_html':
                result = self.clean_no_html(text)
            else:
                raise ValueError(f"Unknown cleaning mode: {mode}")

            if isinstance(result, dict) and 'text' in result:
                if self.verbose:
                    logger.info("Cleaning completed successfully")
                return result['text']
            else:
                raise ValueError("Cleaning returned empty result")

        except Exception as e:
            logger.error(f"Stdin cleaning failed: {str(e)}")
            raise

    def list_cleaning_modes(self) -> Dict[str, str]:
        """List available cleaning modes with descriptions.

        Returns:
            Dictionary of mode names and descriptions
        """
        return {
            'full': 'Complete cleaning pipeline (abnormal + filter + privacy)',
            'abnormal': 'Remove abnormal characters, HTML, normalize text',
            'filter': 'Filter by content quality (repetition, length, numeric content)',
            'privacy': 'Remove sensitive information (emails, phones, IDs, etc.)',
            'no_html': 'Basic cleaning without HTML removal'
        }

    def get_cleaning_info(self, mode: str) -> Dict[str, Any]:
        """Get information about a specific cleaning mode.

        Args:
            mode: Cleaning mode name

        Returns:
            Dictionary with mode information
        """
        modes_info = {
            'full': {
                'name': 'Full Cleaning',
                'description': 'Complete text cleaning pipeline',
                'steps': ['Abnormal cleaning', 'Content filtering', 'Privacy desensitization'],
                'parameters': ['filter_threshold', 'min_chars', 'max_chars', 'numeric_threshold']
            },
            'abnormal': {
                'name': 'Abnormal Cleaning',
                'description': 'Basic text normalization and character cleaning',
                'steps': ['Remove abnormal chars', 'HTML removal', 'Text normalization'],
                'parameters': []
            },
            'filter': {
                'name': 'Content Filtering',
                'description': 'Filter text based on quality metrics',
                'steps': ['Word repetition check', 'Character count validation', 'Numeric content check'],
                'parameters': ['filter_threshold', 'min_chars', 'max_chars', 'numeric_threshold']
            },
            'privacy': {
                'name': 'Privacy Desensitization',
                'description': 'Remove sensitive personal information',
                'steps': ['IP address removal', 'Email/phone masking', 'ID number replacement'],
                'parameters': []
            },
            'no_html': {
                'name': 'No-HTML Cleaning',
                'description': 'Basic cleaning without HTML tag removal',
                'steps': ['Character normalization', 'Text formatting', 'Encoding cleanup'],
                'parameters': []
            }
        }

        return modes_info.get(mode, {
            'name': f'{mode.title()} Cleaning',
            'description': 'Custom cleaning mode',
            'steps': ['Custom operations'],
            'parameters': []
        })
