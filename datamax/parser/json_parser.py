import json

from datamax.parser.base import BaseLife, MarkdownOutputVo


class JsonParser(BaseLife):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    @staticmethod
    def read_json_file(file_path: str) -> str:
        """Read and pretty print a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            content = self.read_json_file(file_path)
            lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
                usage_purpose="Documentation",
                life_type="LLM_ORIGIN",
            )
            output_vo = MarkdownOutputVo(self.get_file_extension(file_path), content)
            output_vo.add_lifecycle(lifecycle)
            return output_vo.to_dict()
        except Exception as e:
            raise e
