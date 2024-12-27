import os
import pathlib
import sys
import docx2markdown
from typing import Union

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo
from langchain_community.document_loaders import PyMuPDFLoader
from datamax.utils import clean_original_text
from datamax.utils import setup_environment


class PdfParser(BaseLife):

    def __init__(self, file_path: Union[str, list], use_ocr: bool = False, use_gpu: bool = False, gpu_id: int = 6):
        super().__init__()
        self.file_path = file_path
        self.use_ocr = use_ocr
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        if use_gpu:
            setup_environment(use_gpu=True)
        else:
            setup_environment(use_gpu=False)

    @staticmethod
    def read_pdf_file(file_path) -> str:
        try:
            pdf_loader = PyMuPDFLoader(file_path)
            pdf_documents = pdf_loader.load()
            result_text = ''
            for page in pdf_documents:
                result_text += page.page_content
            return result_text
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        from datamax.utils.paddleocr_pdf_operator import use_paddleocr
        try:
            title = self.get_file_extension(file_path)
            if self.use_ocr:
                output_docx_dir = f'./output/{os.path.basename(file_path).replace(".pdf", "_ocr.docx")}'
                if os.path.exists(output_docx_dir):
                    pass
                else:
                    use_paddleocr(file_path, './output', self.use_gpu, self.gpu_id)
                output_md_dir = f'./output/{os.path.basename(file_path).replace(".pdf", "_ocr.md")}'
                docx2markdown.docx_to_markdown(output_docx_dir, output_md_dir)
                mk_content = open(output_md_dir, 'r', encoding='utf-8').read()
                token_count = self.tk_client.get_tokenizer(content=mk_content)
            else:
                content = self.read_pdf_file(file_path=file_path)
                # clean_text = clean_original_text(content)
                mk_content = content
                token_count = self.tk_client.get_tokenizer(content=mk_content)

            lifecycle = self.generate_lifecycle(source_file=file_path, token_count=token_count, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception:
            raise
