"""Parser CLI Class

Provides object-oriented interface for parser operations.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from loguru import logger
from tqdm import tqdm

from datamax.parser import DataMax


class ParserCLI:
    """Object-oriented interface for parser operations.

    Provides programmatic access to parser functionality
    that can be used by other applications or scripts.
    """

    def __init__(self, verbose: bool = False):
        """Initialize parser CLI.

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

    def parse_file(self,
                  input_file: str,
                  output_file: Optional[str] = None,
                  format: str = 'markdown',
                  domain: str = 'Technology',
                  use_mineru: bool = False,
                  use_qwen_vl_ocr: bool = False,
                  use_mllm: bool = False,
                  mllm_system_prompt: str = "描述图片内容，包括图片中的文字、图片中的对象、图片中的场景等。输出一份专业的中文markdown报告",
                  api_key: Optional[str] = None,
                  base_url: Optional[str] = None,
                  model_name: Optional[str] = None,
                  to_markdown: bool = False) -> Dict[str, Any]:
        """Parse a single file using DataMax parser.

        Args:
            input_file: Input file path
            output_file: Output file path (optional)
            format: Output format ('markdown', 'json', 'text')
            domain: Document domain
            use_mineru: Use MinerU for PDF parsing
            use_qwen_vl_ocr: Use Qwen-VL OCR for PDF
            use_mllm: Use Vision model for images
            mllm_system_prompt: System prompt for Vision model
            api_key: API key for services
            base_url: Base URL for API
            model_name: Model name
            to_markdown: Convert to Markdown format

        Returns:
            Dictionary with parsed result
        """
        try:
            if self.verbose:
                logger.info(f"Parsing file: {input_file}")

            # Validate input file
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file '{input_file}' not found")

            # Get API credentials from environment if not provided
            if use_qwen_vl_ocr or use_mllm:
                if not api_key:
                    if use_mllm:
                        api_key = os.getenv('OPENAI_API_KEY')
                    elif use_qwen_vl_ocr:
                        api_key = os.getenv('DASHSCOPE_API_KEY')
                    if not api_key:
                        raise ValueError("API key is required for OCR/MLLM features")

                if not base_url:
                    if use_mllm:
                        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                    elif use_qwen_vl_ocr:
                        base_url = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/api/v1')

                if not model_name:
                    if use_mllm:
                        model_name = 'gpt-4o'
                    elif use_qwen_vl_ocr:
                        model_name = 'qwen-vl-max-latest'

            # Initialize DataMax parser
            datamax = DataMax(
                file_path=str(input_path),
                domain=domain,
                use_mineru=use_mineru,
                use_qwen_vl_ocr=use_qwen_vl_ocr,
                use_mllm=use_mllm,
                mllm_system_prompt=mllm_system_prompt,
                to_markdown=to_markdown,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )

            # Parse the file
            result = datamax.get_data()

            if self.verbose:
                logger.info(f"Parsing completed successfully")

            # Save result if output file specified
            if output_file:
                self._save_result(result, output_file, format)

            return result

        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            raise

    def parse_batch(self,
                   input_dir: str,
                   output_dir: str,
                   format: str = 'markdown',
                   pattern: str = '*.*',
                   recursive: bool = False,
                   max_workers: int = 4,
                   continue_on_error: bool = True,
                   **parse_options) -> List[Dict[str, Any]]:
        """Parse multiple files in batch mode.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            format: Output format
            pattern: File pattern to match
            recursive: Process directories recursively
            max_workers: Maximum concurrent workers
            continue_on_error: Continue processing on errors
            **parse_options: Additional parsing options

        Returns:
            List of parsing results
        """
        try:
            if self.verbose:
                logger.info(f"Starting batch parsing: {input_dir} -> {output_dir}")

            # Validate input directory
            input_path = Path(input_dir)
            if not input_path.exists() or not input_path.is_dir():
                raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory")

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Find files to process
            if recursive:
                files = list(input_path.rglob(pattern))
            else:
                files = list(input_path.glob(pattern))

            # Filter out directories
            files = [f for f in files if f.is_file()]

            if not files:
                logger.warning(f"No files found matching pattern '{pattern}' in {input_dir}")
                return []

            if self.verbose:
                logger.info(f"Found {len(files)} files to process")

            # Process files
            results = []
            start_time = time.time()

            if self.verbose:
                # Use tqdm for progress display
                with tqdm(total=len(files), desc="Parsing files") as pbar:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_file = {
                            executor.submit(self._parse_single_file, file_path, output_path, format, parse_options, continue_on_error): file_path
                            for file_path in files
                        }

                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            try:
                                result = future.result()
                                results.append(result)
                            except Exception as e:
                                error_result = {
                                    'success': False,
                                    'file': str(file_path),
                                    'error': str(e)
                                }
                                results.append(error_result)

                            pbar.update(1)
                            pbar.set_postfix({
                                'processed': len(results),
                                'successful': len([r for r in results if r.get('success', False)])
                            })
            else:
                # Process without progress bar
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self._parse_single_file, file_path, output_path, format, parse_options, continue_on_error): file_path
                        for file_path in files
                    }

                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            error_result = {
                                'success': False,
                                'file': str(file_path),
                                'error': str(e)
                            }
                            results.append(error_result)

            # Show summary
            total_time = time.time() - start_time
            successful = len([r for r in results if r.get('success', True)])
            failed = len(results) - successful

            if self.verbose or not continue_on_error:
                click.echo(f"\n📊 Batch Processing Summary:")
                click.echo(f"   📁 Files processed: {len(results)}")
                click.echo(f"   ✅ Successful: {successful}")
                click.echo(f"   ❌ Failed: {failed}")
                click.echo(f"   ⏱️  Total time: {total_time:.2f}s")
                click.echo(f"   📈 Average speed: {len(results)/total_time:.2f} files/s")

            return results

        except Exception as e:
            logger.error(f"Batch parsing failed: {str(e)}")
            raise

    def _parse_single_file(self, file_path: Path, output_dir: Path, format: str,
                          parse_options: dict, continue_on_error: bool) -> Dict[str, Any]:
        """Parse a single file in batch mode."""
        try:
            # Determine output file path
            relative_path = file_path.relative_to(file_path.parents[-2]) if len(file_path.parents) > 1 else file_path.name
            output_file = output_dir / f"{relative_path.stem}_parsed.{self._get_extension(format)}"

            # Ensure output subdirectory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Parse the file
            result = self.parse_file(
                input_file=str(file_path),
                output_file=str(output_file),
                format=format,
                **parse_options
            )

            # Add metadata
            result_with_meta = {
                'success': True,
                'input_file': str(file_path),
                'output_file': str(output_file),
                'result': result
            }

            return result_with_meta

        except Exception as e:
            if continue_on_error:
                logger.warning(f"Failed to parse {file_path}: {str(e)}")
                return {
                    'success': False,
                    'input_file': str(file_path),
                    'error': str(e)
                }
            else:
                raise

    def _save_result(self, result: Any, output_file: str, format: str) -> str:
        """Save parsing result to file.

        Args:
            result: Parsing result
            output_file: Output file path
            format: Output format

        Returns:
            Path to saved file
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            elif format == 'markdown':
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(result, dict):
                        content = result.get('content', str(result))
                    else:
                        content = str(result)
                    f.write(content)
            elif format == 'text':
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(result, dict):
                        content = result.get('content', str(result))
                    else:
                        content = str(result)
                    f.write(content)
            else:
                raise ValueError(f"Unsupported format: {format}")

            if self.verbose:
                logger.info(f"Result saved to: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save result: {str(e)}")
            raise

    def _get_extension(self, format: str) -> str:
        """Get file extension for format."""
        extensions = {
            'markdown': 'md',
            'json': 'json',
            'text': 'txt'
        }
        return extensions.get(format, 'txt')

    def list_supported_formats(self) -> Dict[str, str]:
        """List all supported file formats with descriptions.

        Returns:
            Dictionary of format names and descriptions
        """
        return {
            # Document formats
            '.pdf': 'PDF documents (supports OCR with MinerU/Qwen-VL)',
            '.docx': 'Microsoft Word documents (can convert to Markdown)',
            '.doc': 'Legacy Microsoft Word documents',
            '.wps': 'WPS Office documents',
            '.epub': 'EPUB e-book files',
            '.md': 'Markdown files',

            # Spreadsheet formats
            '.xlsx': 'Microsoft Excel spreadsheets',
            '.xls': 'Legacy Excel spreadsheets',
            '.csv': 'Comma-separated values files',

            # Presentation formats
            '.pptx': 'Microsoft PowerPoint presentations',
            '.ppt': 'Legacy PowerPoint presentations',

            # Web formats
            '.html': 'HTML web pages',

            # Text formats
            '.txt': 'Plain text files',

            # Image formats
            '.jpg': 'JPEG images (supports Vision model analysis)',
            '.jpeg': 'JPEG images (supports Vision model analysis)',
            '.png': 'PNG images (supports Vision model analysis)',
            '.webp': 'WebP images (supports Vision model analysis)',

            # Code formats
            '.py': 'Python source code',
            '.js': 'JavaScript source code',
            '.java': 'Java source code',
            '.cpp': 'C++ source code',
            '.c': 'C source code',
            '.go': 'Go source code',
            '.rs': 'Rust source code',
        }

    def get_parse_options_help(self) -> str:
        """Get help text for parsing options.

        Returns:
            Help text for parsing options
        """
        return """
Parsing Options:

Basic Options:
  --format FORMAT        Output format: markdown, json, text (default: markdown)
  --domain DOMAIN        Document domain for processing (default: Technology)

PDF Specific Options:
  --use-mineru           Use MinerU for advanced PDF layout analysis
  --use-qwen-vl-ocr      Use Qwen-VL OCR for image-based PDF processing
  --api-key KEY          API key for OCR services (or set env var)
  --base-url URL         Base URL for OCR services (or set env var)
  --model MODEL          Model name for OCR services (or set env var)

Image Specific Options:
  --use-mllm             Use Vision model for image analysis
  --mllm-prompt PROMPT   System prompt for Vision model

Document Conversion:
  --to-markdown          Convert documents to Markdown format (Word, etc.)

Batch Processing Options:
  --pattern PATTERN      File pattern to match (default: *.*)
  --recursive            Process directories recursively
  --max-workers N        Maximum concurrent workers (default: 4)

Environment Variables:
  OPENAI_API_KEY         API key for OpenAI services
  OPENAI_BASE_URL        Base URL for OpenAI API
  DASHSCOPE_API_KEY      API key for DashScope services
  DASHSCOPE_BASE_URL     Base URL for DashScope API
        """.strip()
