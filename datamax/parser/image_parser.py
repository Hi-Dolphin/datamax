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
            # 【1】改用 pathlib.Path.stem 获取“基础名”
            base_name = pathlib.Path(file_path).stem
            output_pdf_path = f"{base_name}.pdf"

            # 转换图片为 PDF
            img = Image.open(file_path)
            img.save(output_pdf_path, "PDF", resolution=100.0)

            # 委托 PdfParser 解析，传入扩展名已由 PdfParser 内部获取
            pdf_parser = PdfParser(output_pdf_path, use_mineru=True)
            result = pdf_parser.parse(output_pdf_path)

            # 清理临时文件
            if os.path.exists(output_pdf_path):
                os.remove(output_pdf_path)

            return result

        except Exception:
            raise