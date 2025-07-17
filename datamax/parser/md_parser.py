
import loguru
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class MarkdownParser(BaseLife):
    """
    Parser for Markdown files that follows the same pattern as PdfParser.
    Handles .md and .markdown file extensions.
    """

    def __init__(self, file_path: str | list, domain: str = "Technology"):
        super().__init__(domain=domain)
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
            with open(file_path, encoding="utf-8") as f:
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
            extension = self.get_file_extension(file_path)

            # 1) Generate "start processing" lifecycle
            start_lc = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSING,
            )

            # 2) Read Markdown content
            md_content = self.read_markdown_file(file_path)

            # 3) Create output VO and add start event
            output_vo = MarkdownOutputVo(extension, md_content)
            output_vo.add_lifecycle(start_lc)

            # 4) Generate "processing completed" lifecycle
            end_lc = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            output_vo.add_lifecycle(end_lc)

            return output_vo.to_dict()

        except Exception as e:
            loguru.logger.error(f"Failed to parse markdown file {file_path}: {e}")
            # (Optional) Record a failure lifecycle
            fail_lc = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            # If you want to return VO even on failure, you can do this:
            # output_vo = MarkdownOutputVo(self.get_file_extension(file_path), "")
            # output_vo.add_lifecycle(fail_lc)
            raise
