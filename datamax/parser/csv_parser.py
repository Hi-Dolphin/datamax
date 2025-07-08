import pandas as pd

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class CsvParser(BaseLife):

    def __init__(self, file_path, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    @staticmethod
    def read_csv_file(file_path: str) -> pd.DataFrame:
        """Read a CSV file into a pandas DataFrame."""
        return pd.read_csv(file_path)

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            # 1) 处理开始
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )

            # 2) 核心解析
            df = self.read_csv_file(file_path)
            mk_content = df.to_markdown(index=False)

            # 3) 处理结束或失败
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if mk_content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )

            # 4) 封装输出并添加生命周期
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()
        except Exception as e:
            raise e
