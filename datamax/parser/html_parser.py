from typing import Union

from bs4 import BeautifulSoup

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class HtmlParser(BaseLife):
    def __init__(self, file_path: Union[str, list], domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    @staticmethod
    def read_html_file(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
                soup = BeautifulSoup(data, "html.parser")
                return soup.get_text(separator="\n", strip=True)
        except Exception:
            raise

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            # 1) 提取扩展名并生成“处理开始”事件
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )

            # 2) 核心解析
            content = self.read_html_file(file_path=file_path)
            mk_content = content

            # 3) 根据内容生成“处理完成”或“处理失败”事件
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

        except Exception:
            raise
