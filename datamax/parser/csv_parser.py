from datamax.parser.base import MarkdownOutputVo


class CsvParser:

    def __init__(self, filename):
        self.filename = filename

    def parse(self) -> MarkdownOutputVo:
        pass