import os
import pathlib
import sys
from datamax.utils import setup_environment
setup_environment(use_gpu=True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from paddleocr import PaddleOCR
from datamax.parser.base import MarkdownOutputVo
ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife


class ImageParser(BaseLife):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            title = self.get_file_extension(file_path)
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            result = ocr.ocr(file_path, cls=True)
            recognized_texts = [l[1][0] for line in result for l in line]
            mk_content = '\n'.join(recognized_texts)
            token_count = self.tk_client.get_tokenizer(content=mk_content)
            lifecycle = self.generate_lifecycle(source_file=file_path, token_count=token_count, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e

