
import chardet

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class TxtParser(BaseLife):
    def __init__(self, file_path: str | list, domain: str = "Technology"):
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
            with open(file_path, encoding=encoding) as file:
                return file.read()
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            extension = self.get_file_extension(file_path)

            # 1) Start processing
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSING,
            )

            # 2) Read file content
            content = self.read_txt_file(file_path=file_path)
            mk_content = content

            # 3) Construct output object and add start lifecycle
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)

            # 4) Processing completed
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            output_vo.add_lifecycle(lc_end)

            return output_vo.to_dict()

        except Exception as e:
            # 5) Processing failed
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            # (Optional) If you want to return VO even on failure, you can construct VO with empty content here and add lc_fail
            raise
