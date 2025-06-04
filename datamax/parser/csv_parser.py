import pandas as pd

from datamax.parser.base import BaseLife, MarkdownOutputVo


class CsvParser(BaseLife):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    @staticmethod
    def read_csv_file(file_path: str) -> pd.DataFrame:
        """Read a CSV file into a pandas DataFrame."""
        return pd.read_csv(file_path)

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            df = self.read_csv_file(file_path)
            mk_content = df.to_markdown(index=False)
            lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
                usage_purpose="Documentation",
                life_type="LLM_ORIGIN",
            )
            output_vo = MarkdownOutputVo(self.get_file_extension(file_path), mk_content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e
