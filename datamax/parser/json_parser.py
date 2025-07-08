import json

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class JsonParser(BaseLife):

    def __init__(self, file_path, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    @staticmethod
    def read_json_file(file_path: str) -> str:
        """Read and pretty print a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            # 1) 处理开始：DATA_PROCESSING
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )

            # 2) 核心解析：读取并格式化 JSON
            content = self.read_json_file(file_path)

            # 3) 处理结束：DATA_PROCESSED 或 DATA_PROCESS_FAILED
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )

            # 4) 封装输出并添加这两条生命周期
            output_vo = MarkdownOutputVo(extension, content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception as e:
            raise e
