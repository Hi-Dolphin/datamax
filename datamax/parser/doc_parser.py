import os
import shutil
import subprocess
import tempfile
import chardet
import docx2markdown
from pathlib import Path
from typing import Union
from docx import Document
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo


class DocParser(BaseLife):
    def __init__(self, file_path: Union[str, list], to_markdown: bool = False):
        super().__init__()
        self.file_path = file_path
        self.to_markdown = to_markdown

    def doc_to_docx(self, doc_path: str, dir_path: str) -> str:
        cmd = f'soffice --headless --convert-to docx "{doc_path}" --outdir "{dir_path}"'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        if exit_code == 0:
            pass
        else:
            encoding = chardet.detect(stderr)['encoding']
            if encoding is None:
                encoding = 'utf-8'
            raise Exception(f"Error Output (detected encoding: {encoding}):", stderr.decode(encoding, errors='replace'))
        fname = str(Path(doc_path).stem)
        docx_path = os.path.join(os.path.dirname(doc_path), f'{fname}.docx')
        if not os.path.exists(docx_path):
            raise Exception(f"> !!! File conversion failed {doc_path} ==> {docx_path}")
        else:
            return docx_path

    def read_docx_file(self, doc_path: str, to_mk: bool) -> str:
        try:
            with tempfile.TemporaryDirectory() as temp_path:
                temp_dir = Path(temp_path)
                media_dir = temp_dir / "media"
                media_dir.mkdir()
                file_path = temp_dir / "tmp.doc"
                shutil.copy(doc_path, file_path)
                docx_file_path = self.doc_to_docx(str(file_path), str(temp_path))
                doc = Document(docx_file_path)
                full_text = [para.text for para in doc.paragraphs]
                if to_mk:
                    output_md_dir = f'./output/{os.path.basename(docx_file_path).replace(".docx", ".md")}'
                    output_dir = os.path.dirname(output_md_dir)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    docx2markdown.docx_to_markdown(docx_file_path, output_md_dir)
                    mk_content = open(output_md_dir, 'r', encoding='utf-8').read()
                    return mk_content
                else:
                    return '\n'.join(full_text)
        except Exception as e:
            raise e

    def parse(self, file_path: str):
        try:
            title = self.get_file_extension(file_path)
            if self.to_markdown:
                mk_content = self.read_docx_file(doc_path=file_path, to_mk=True)
            else:
                content = self.read_docx_file(doc_path=file_path, to_mk=False)
                mk_content = content
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e