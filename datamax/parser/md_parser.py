import pathlib
import sys
from typing import Union

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo
from loguru import logger
import os

class MarkdownParser(BaseLife):
    """
    Parser for Markdown files that follows the same pattern as PdfParser.
    Handles .md and .markdown file extensions.
    """

    def __init__(self,
                 file_path: Union[str, list],
                 ):
        super().__init__()
        self.file_path = file_path

    @staticmethod
    def read_markdown_file(file_path: str) -> str:
        """
        Reads the content of a markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            str: Content of the markdown file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading markdown file {file_path}: {e}")
            raise

    def parse(self, file_path: str) -> MarkdownOutputVo:
        """
        Parses a markdown file and returns a MarkdownOutputVo.

        Args:
            file_path: Path to the markdown file

        Returns:
            MarkdownOutputVo: Structured output containing the markdown content
        """
        try:
            title = os.path.splitext(os.path.basename(file_path))[0]

            # Read markdown content
            md_content = self.read_markdown_file(file_path)

            # Generate lifecycle metadata
            lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
                usage_purpose="Documentation",
                life_type="LLM_ORIGIN"
            )

            # Create and return output VO
            output_vo = MarkdownOutputVo(title, md_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()

        except Exception as e:
            logger.error(f"Failed to parse markdown file {file_path}: {e}")
            raise