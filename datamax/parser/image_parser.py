import os
import pathlib
import sys
from datamax.utils import setup_environment

setup_environment(use_gpu=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from datamax.parser.base import MarkdownOutputVo

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife
from datamax.parser.pdf_parser import PdfParser
from PIL import Image

class ImageParser(BaseLife):
    def __init__(self,file_path: str):
        super().__init__()
        self.file_path = file_path

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            title = self.get_file_extension(file_path)
            output_pdf_path = f'{os.path.basename(file_path).strip(title)}.pdf'
            image = Image.open(file_path)
            image.save(output_pdf_path, 'PDF', resolution=100.0)
            pdf_parser = PdfParser(output_pdf_path, use_mineru=True)
            output_vo = pdf_parser.parse(output_pdf_path)
            if os.path.exists(output_pdf_path):
                # shutil.rmtree(f'uploaded_files/markdown')
                os.remove(output_pdf_path)
            return output_vo
        except Exception as e:
            raise e
