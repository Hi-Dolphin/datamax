import os
from typing import Union

import ebooklib
import loguru
from bs4 import BeautifulSoup
from ebooklib import epub

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class EpubParser(BaseLife):
    def __init__(self, file_path: Union[str, list], domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    @staticmethod
    def read_epub_file(file_path: str) -> str:
        try:
            book = epub.read_epub(file_path)
            content = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_content = item.get_content().decode("utf-8")
                    soup = BeautifulSoup(chapter_content, "html.parser")
                    text = soup.get_text()
                    text = text.replace("\u3000", " ")
                    content += text
            return content
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        try:
            extension = self.get_file_extension(file_path)

            # 1) 开始处理
            start_lc = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSING,
            )

            # 2) 读取EPUB内容
            content = self.read_epub_file(file_path=file_path)
            mk_content = content

            # 3) 创建输出 VO 并添加开始事件
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(start_lc)

            # 4) 处理完成
            end_lc = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            output_vo.add_lifecycle(end_lc)

            return output_vo.to_dict()

        except Exception as e:
            loguru.logger.error(f"Failed to parse epub file {file_path}: {e}")
            # 失败时记录一次失败生命周期（可选）
            fail_lc = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            # 若需返回 VO：
            # output_vo = MarkdownOutputVo(self.get_file_extension(file_path), "")
            # output_vo.add_lifecycle(fail_lc)
            raise
