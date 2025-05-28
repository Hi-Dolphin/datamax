import chardet
from typing import Union
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo


class TxtParser(BaseLife):
    def __init__(self, file_path: Union[str, list]):
        super().__init__()
        self.file_path = file_path

    @staticmethod
    def detect_encoding(file_path: str):
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
                return result['encoding']
        except Exception as e:
            raise e

    @staticmethod
    def read_txt_file(file_path: str) -> str:
        """
        Reads the Txt file in the specified path and returns its contents.
        :param file_path: indicates the path of the Txt file to be read.
        :return: str: Txt file contents.
        """
        try:
            encoding = TxtParser.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            title = self.get_file_extension(file_path)
            content = self.read_txt_file(file_path=file_path)  # 真实数据是从load加载
            mk_content = content
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e