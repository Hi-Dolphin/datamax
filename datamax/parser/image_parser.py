import os
import pathlib
import sys

from datamax.utils import setup_environment

setup_environment(use_gpu=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from PIL import Image

from datamax.parser.base import BaseLife
from datamax.parser.pdf_parser import PdfParser
from datamax.utils.lifecycle_types import LifeType


class ImageParser(BaseLife):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def parse(self, file_path: str):
        try:
            # Start data processing
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )
            # Use pathlib.Path.stem to get the "base name"
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
            # 2) 处理结束：根据内容是否非空生成 DATA_PROCESSED 或 DATA_PROCESS_FAILED
            content = result.get("content", "")
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
                life_type=(
                    LifeType.DATA_PROCESSED
                    if content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )

            # 3) 合并生命周期：先插入 start，再追加 end
            lifecycle = result.get("lifecycle", [])
            lifecycle.insert(0, lc_start.to_dict())
            lifecycle.append(lc_end.to_dict())
            result["lifecycle"] = lifecycle

            return result

        except Exception:
            raise
