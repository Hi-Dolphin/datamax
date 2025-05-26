from datamax.parser.base import MarkdownOutputVo
from datamax.parser.base import BaseLife
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class XlsParser(BaseLife):
    """xlsx or xls table use markitdown from Microsoft  so magic for table!"""

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            df = pd.read_excel(file_path)
            mk_content = df.to_markdown(index=False)
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            output_vo = MarkdownOutputVo(self.get_file_extension(file_path), mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e
