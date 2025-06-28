import os
import pathlib
import sys
from datamax.utils import setup_environment

setup_environment(use_gpu=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife
from datamax.parser.pdf_parser import PdfParser
from PIL import Image

class ImageParser(BaseLife):
    def __init__(self,file_path: str):
        super().__init__()
        self.file_path = file_path

    def parse(self, file_path: str):
        try:
            # 【1】Use pathlib.Path.stem to get the "base name"
            base_name = pathlib.Path(file_path).stem
            output_pdf_path = f"{base_name}.pdf"

            # Convert image to PDF
            img = Image.open(file_path)
            img.save(output_pdf_path, "PDF", resolution=100.0)

            # Delegate parsing to PdfParser, the extension is handled internally by PdfParser
            pdf_parser = PdfParser(output_pdf_path, use_mineru=True)
            result = pdf_parser.parse(output_pdf_path)

            # Clean up temporary files
            if os.path.exists(output_pdf_path):
                os.remove(output_pdf_path)

            return result

        except Exception:
            raise