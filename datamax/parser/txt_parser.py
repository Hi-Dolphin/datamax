from typing import Union

import chardet

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class TxtParser(BaseLife):
    def __init__(self, file_path: Union[str, list], domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    @staticmethod
    def detect_encoding(file_path: str):
        try:
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read())
                return result["encoding"]
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
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            extension = self.get_file_extension(file_path)

            # 1) 开始处理
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSING,
            )

            # 2) 读取文件内容
            content = self.read_txt_file(file_path=file_path)
            mk_content = content

            # 3) 构造输出对象并加上开始生命周期
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)

            # 4) 处理完成
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            output_vo.add_lifecycle(lc_end)

            return output_vo.to_dict()

        except Exception as e:
            # 5) 处理失败
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            # （可选）如果希望在失败时也返回 VO，可在这里构造空 content 的 VO 并加入 lc_fail
            raise
