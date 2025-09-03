"""Generator CLI Class

Provides object-oriented interface for generator operations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import click
from loguru import logger

from datamax.generator import (
    full_qa_labeling_process,
    generate_multimodal_qa_pairs
)


class GeneratorCLI:
    """Object-oriented interface for generator operations.

    Provides programmatic access to generator functionality
    that can be used by other applications or scripts.
    """

    def __init__(self, verbose: bool = False):
        """Initialize generator CLI.

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

    def generate_qa(self,
                   input_file: str,
                   output_file: Optional[str] = None,
                   api_key: str = None,
                   base_url: str = None,
                   model: str = None,
                   chunk_size: int = 500,
                   chunk_overlap: int = 100,
                   question_number: int = 5,
                   max_workers: int = 5) -> Dict[str, Any]:
        """Generate QA pairs from text files.

        Args:
            input_file: Input file path
            output_file: Output file path (optional)
            api_key: API key
            base_url: API base URL
            model: Model name
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
            question_number: Number of questions per chunk
            max_workers: Maximum number of workers

        Returns:
            Dictionary with generated QA pairs
        """
        try:
            if self.verbose:
                logger.info("Generating QA pairs from text...")

            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file '{input_file}' not found")

            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                raise ValueError("Input file is empty")

            # Get API credentials
            if not api_key:
                api_key = self._get_api_key('DASHSCOPE_API_KEY')

            if not base_url:
                base_url = self._get_api_key('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/api/v1')

            if not model:
                model = 'qwen-max'

            # Generate QA pairs
            result = full_qa_labeling_process(
                content=content,
                file_path=str(input_path),
                api_key=api_key,
                base_url=base_url,
                model_name=model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                question_number=question_number,
                max_workers=max_workers,
                use_mineru=False,
                debug=self.verbose
            )

            if self.verbose:
                qa_count = len(result.get('qa_pairs', []))
                logger.info(f"Generated {qa_count} QA pairs successfully")

            return result

        except Exception as e:
            logger.error(f"QA generation failed: {str(e)}")
            raise

    def generate_multimodal_qa(self,
                              input_file: str,
                              output_file: Optional[str] = None,
                              api_key: str = None,
                              model: str = 'gpt-4-vision-preview',
                              chunk_size: int = 2000,
                              chunk_overlap: int = 300,
                              question_number: int = 2,
                              max_workers: int = 5) -> List[Dict[str, Any]]:
        """Generate multimodal QA pairs from markdown files with images.

        Args:
            input_file: Input markdown file path
            output_file: Output file path (optional)
            api_key: OpenAI API key
            model: Model name (default: gpt-4-vision-preview)
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
            question_number: Number of questions per chunk
            max_workers: Maximum number of workers

        Returns:
            List of generated multimodal QA pairs
        """
        try:
            if self.verbose:
                logger.info("Generating multimodal QA pairs...")

            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file '{input_file}' not found")

            if input_path.suffix.lower() != '.md':
                raise ValueError("Multimodal QA generation requires markdown files")

            # Get API credentials
            if not api_key:
                api_key = self._get_api_key('OPENAI_API_KEY')

            if not api_key:
                raise ValueError("OpenAI API key is required for multimodal generation. "
                               "Set OPENAI_API_KEY environment variable or use --api-key option.")

            # Generate multimodal QA pairs
            result = generate_multimodal_qa_pairs(
                file_path=str(input_path),
                api_key=api_key,
                model_name=model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                question_number=question_number,
                max_workers=max_workers,
                debug=self.verbose
            )

            if self.verbose:
                qa_count = len(result)
                logger.info(f"Generated {qa_count} multimodal QA pairs successfully")

            return result

        except Exception as e:
            logger.error(f"Multimodal QA generation failed: {str(e)}")
            raise

    def save_result(self, result: Any, output_file: str, format: str = 'json') -> str:
        """Save generation result to file.

        Args:
            result: Generation result
            output_file: Output file path
            format: Output format ('json' only for now)

        Returns:
            Path to saved file
        """
        try:
            output_path = Path(output_file)

            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

            if self.verbose:
                logger.info(f"Result saved to: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save result: {str(e)}")
            raise

    def list_generators(self) -> Dict[str, str]:
        """List available generators with descriptions.

        Returns:
            Dictionary of generator names and descriptions
        """
        return {
            'qa': '纯文本问答对生成器 - 支持多种文档格式，生成结构化QA数据',
            'multimodal': '多模态问答对生成器 - 从包含图片的Markdown生成视觉QA数据'
        }

    def _get_api_key(self, env_var: str, default: str = None) -> str:
        """Get API key from environment variables.

        Args:
            env_var: Environment variable name
            default: Default value if not found

        Returns:
            API key value
        """
        import os
        return os.getenv(env_var, default)
