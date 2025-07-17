import warnings

import pandas as pd

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


warnings.filterwarnings("ignore")


class XlsParser(BaseLife):
    """xlsx or xls table use markitdown from Microsoft  so magic for table!"""

    def __init__(self, file_path, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            # 🏷️ Parsing started
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSING,
            )

            # 📊 Read Excel and generate Markdown
            df = pd.read_excel(file_path)
            mk_content = df.to_markdown(index=False)

            # 🏷️ Parsing completed
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )

            output_vo = MarkdownOutputVo(self.get_file_extension(file_path), mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception as e:
            # ❌ Parsing failed
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            # Don't return empty VO here, throw directly, framework can catch and report
            raise e
