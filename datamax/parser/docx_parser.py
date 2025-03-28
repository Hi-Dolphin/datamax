import os
import docx2markdown
from docx import Document
from typing import Union
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo


class DocxParser(BaseLife):
    def __init__(self, file_path: Union[str, list], to_markdown: bool = False):
        super().__init__()
        self.file_path = file_path
        self.to_markdown = to_markdown

    @staticmethod
    def read_docx_file(file_path: str) -> str:
        try:
            doc = Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            return '\n'.join(full_text)
        except Exception as e:
            raise e

    def parse(self, file_path: str):
        try:
            title = self.get_file_extension(file_path)
            if self.to_markdown:
                output_md_dir = f'./output/{os.path.basename(file_path).replace(".docx", ".md")}'
                output_dir = os.path.dirname(output_md_dir)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                docx2markdown.docx_to_markdown(file_path, output_md_dir)
                mk_content = open(output_md_dir, 'r', encoding='utf-8').read()
            else:
                content = self.read_docx_file(file_path=file_path)
                mk_content = content
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e
