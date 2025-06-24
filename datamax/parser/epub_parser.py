import ebooklib
from typing import Union
from bs4 import BeautifulSoup
from ebooklib import epub
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo
import os

class EpubParser(BaseLife):
    def __init__(self, file_path: Union[str, list]):
        super().__init__()
        self.file_path = file_path

    @staticmethod
    def read_epub_file(file_path: str) -> str:
        try:
            book = epub.read_epub(file_path)
            content = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(chapter_content, 'html.parser')
                    text = soup.get_text()
                    text = text.replace('\u3000', ' ')
                    content += text
            return content
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            extension = self.get_file_extension(file_path)
            content = self.read_epub_file(file_path=file_path)
            mk_content = content
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e
