from datamax.parser.base import MarkdownOutputVo


class Parser:

    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self) -> MarkdownOutputVo:
        pass