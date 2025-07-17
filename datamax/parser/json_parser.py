import json
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveJsonSplitter
from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class JsonParser(BaseLife):

    def __init__(self, file_path, domain: str = "Technology", max_chunk_size: int = 2000, convert_lists: bool = True):
        super().__init__(domain=domain)
        self.file_path = file_path
        self.max_chunk_size = max_chunk_size
        self.convert_lists = convert_lists
        self.splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)

    @staticmethod
    def read_json_file(file_path: str) -> Dict[str, Any]:
        """Read a JSON file and return the data as a dictionary."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def split_json_content(self, json_data: Dict[str, Any]) -> List[str]:
        """
        Split JSON data into smaller chunks using LangChain's RecursiveJsonSplitter.
        
        Args:
            json_data: The JSON data as a dictionary
            
        Returns:
            List of JSON formatted strings representing the chunks
        """
        try:
            # Use RecursiveJsonSplitter to split the JSON data
            text_chunks = self.splitter.split_text(
                json_data=json_data, 
                convert_lists=self.convert_lists,
                ensure_ascii=False
            )
            return text_chunks
        except Exception as e:
            # Fallback to simple JSON string if splitting fails
            return [json.dumps(json_data, indent=2, ensure_ascii=False)]

    def format_chunks_as_markdown(self, chunks: List[str]) -> str:
        """
        Format JSON chunks as markdown with proper code blocks.
        
        Args:
            chunks: List of JSON string chunks
            
        Returns:
            Formatted markdown string
        """
        if len(chunks) == 1:
            # Single chunk, format as one code block
            return f"```json\n{chunks[0]}\n```"
        
        # Multiple chunks, format with headers
        markdown_content = []
        for i, chunk in enumerate(chunks, 1):
            markdown_content.append(f"## JSON Chunk {i}\n\n```json\n{chunk}\n```\n")
        
        return "\n".join(markdown_content)

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            # 1) Start processing: DATA_PROCESSING
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing with RecursiveJsonSplitter",
            )

            # 2) Core parsing: Read and split JSON using RecursiveJsonSplitter
            json_data = self.read_json_file(file_path)
            
            # Split JSON into manageable chunks
            json_chunks = self.split_json_content(json_data)
            
            # Format chunks as markdown
            content = self.format_chunks_as_markdown(json_chunks)
            
            # Add metadata about the splitting process
            chunk_info = f"\n\n---\n\n**JSON Processing Summary:**\n"
            chunk_info += f"- Total chunks: {len(json_chunks)}\n"
            chunk_info += f"- Max chunk size: {self.max_chunk_size} characters\n"
            chunk_info += f"- Convert lists to dicts: {self.convert_lists}\n"
            
            if len(json_chunks) > 1:
                chunk_sizes = [len(chunk) for chunk in json_chunks]
                chunk_info += f"- Chunk sizes: {chunk_sizes}\n"
            
            content += chunk_info

            # 3) End processing: DATA_PROCESSED or DATA_PROCESS_FAILED
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing with RecursiveJsonSplitter",
            )

            # 4) Package output and add these two lifecycle records
            output_vo = MarkdownOutputVo(extension, content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception as e:
            # Record lifecycle for processing failure
            lc_failed = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESS_FAILED,
                usage_purpose="Parsing with RecursiveJsonSplitter - Error occurred",
            )
            
            # Create error output
            error_content = f"# JSON Parsing Error\n\nError occurred while parsing JSON file: {str(e)}\n"
            output_vo = MarkdownOutputVo(self.get_file_extension(file_path), error_content)
            output_vo.add_lifecycle(lc_failed)
            
            return output_vo.to_dict()
