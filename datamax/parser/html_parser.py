from typing import Union, List

from langchain_text_splitters import HTMLHeaderTextSplitter

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class HtmlParser(BaseLife):
    def __init__(self, file_path: str | list, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path
        
        # Configure headers to split on for better content organization
        self.headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"), 
            ("h3", "Header 3"),
            ("h4", "Header 4"),
            ("h5", "Header 5"),
            ("h6", "Header 6"),
        ]
        
        # Initialize the HTML splitter
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            return_each_element=False  # Combine elements with same metadata
        )

    def parse_html_content(self, file_path: str) -> str:
        """
        Parse HTML file using LangChain's HTMLHeaderTextSplitter for structure-aware parsing
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Structured markdown content with header metadata
        """
        try:
            # Read HTML file
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            
            # Use HTMLHeaderTextSplitter to parse the content
            html_documents = self.html_splitter.split_text(html_content)
            
            # Convert documents to structured markdown format
            markdown_content = self._convert_documents_to_markdown(html_documents)
            
            return markdown_content
            
        except Exception as e:
            raise Exception(f"Failed to parse HTML content: {str(e)}")

    def _convert_documents_to_markdown(self, documents: List) -> str:
        """
        Convert LangChain documents to structured markdown format
        
        Args:
            documents: List of Document objects from HTMLHeaderTextSplitter
            
        Returns:
            Formatted markdown content
        """
        markdown_parts = []
        
        for doc in documents:
            # Extract metadata and content
            metadata = doc.metadata
            content = doc.page_content.strip()
            
            if not content:
                continue
                
            # Build header hierarchy from metadata
            header_parts = []
            for i in range(1, 7):  # h1 to h6
                header_key = f"Header {i}"
                if header_key in metadata:
                    header_level = "#" * i
                    header_parts.append(f"{header_level} {metadata[header_key]}")
            
            # Add structured content
            if header_parts:
                # Add the deepest header
                markdown_parts.append(header_parts[-1])
                
            # Add content with proper spacing
            if content:
                markdown_parts.append(content)
                markdown_parts.append("")  # Add spacing between sections
        
        return "\n".join(markdown_parts).strip()

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            # 1) Extract extension and generate "processing start" event
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )

            # 2) Core parsing using LangChain HTMLHeaderTextSplitter
            mk_content = self.parse_html_content(file_path=file_path)

            # 3) Generate "processing completed" or "processing failed" event based on content
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if mk_content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )

            # 4) Package output and add lifecycle
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception:
            raise
