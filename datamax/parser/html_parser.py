from typing import Union
import pathlib
import sys

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo
from bs4 import BeautifulSoup


class HtmlParser(BaseLife):
    def __init__(self, file_path: Union[str, list]):
        super().__init__()
        self.file_path = file_path

    @staticmethod
    def read_html_file(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
                soup = BeautifulSoup(data, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
        except Exception:
            raise

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            title = self.get_file_extension(file_path)
            content = self.read_html_file(file_path=file_path)
            mk_content = content
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception:
            raise