import os
import importlib
from typing import List, Union

from datamax.utils import data_cleaner


class ParserFactory:
    @staticmethod
    def create_parser(file_path: str, use_ocr: bool = False, use_gpu: bool = False, gpu_id: int = 6,
                      to_markdown: bool = False):
        """
        Create a parser instance based on the file extension.

        :param file_path: The path to the file to be parsed.
        :param use_ocr: Flag to indicate whether OCR should be used.
        :param use_gpu: Flag to indicate whether GPU should be used.
        :param gpu_id: The ID of the GPU to use.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
                    (only supported files in .doc or .docx format)
        :return: An instance of the parser class corresponding to the file extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        parser_class_name = {
            '.docx': 'DocxParser',
            '.doc': 'DocParser',
            '.epub': 'EpubParser',
            '.html': 'HtmlParser',
            '.txt': 'TxtParser',
            '.pptx': 'PPtxParser',
            '.ppt': 'PPtParser',
            '.pdf': 'PdfParser',
            '.jpg': 'ImageParser',
            '.png': 'ImageParser',
            '.svg': 'ImageParser',
            '.webp': 'ImageParser',
        }.get(file_extension)

        if not parser_class_name:
            return None

        if file_extension in ['.jpg', '.png', '.svg', '.webp']:
            module_name = f'datamax.parser.image_parser'
        else:
            # Dynamically determine the module name based on the file extension
            module_name = f'datamax.parser.{file_extension[1:]}_parser'

        try:
            # Dynamically import the module and get the class
            module = importlib.import_module(module_name)
            parser_class = getattr(module, parser_class_name)

            # Special handling for PdfParser arguments
            if parser_class_name == 'PdfParser':
                return parser_class(file_path, use_ocr, use_gpu, gpu_id)
            elif parser_class_name == 'DocxParser' or parser_class_name == 'DocParser':
                return parser_class(file_path, to_markdown)
            else:
                return parser_class(file_path)

        except (ImportError, AttributeError) as e:
            raise e


class DataMaxParser:
    def __init__(self, file_path: Union[str, list] = '', use_ocr: bool = False, use_gpu: bool = False, gpu_id: int = 6,
                 to_markdown: bool = False):
        """
        Initialize the DataMaxParser with file path and parsing options.

        :param file_path: The path to the file or directory to be parsed.
        :param use_ocr: Flag to indicate whether OCR should be used.
        :param use_gpu: Flag to indicate whether GPU should be used.
        :param gpu_id: The ID of the GPU to use.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
        """
        self.file_path = file_path
        self.use_ocr = use_ocr
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.to_markdown = to_markdown
        self.parsed_data = None

    def get_data(self):
        """
        Parse the file or directory specified in the file path and return the data.

        :return: A list of parsed data if the file path is a directory, otherwise a single parsed data.
        """
        try:
            if isinstance(self.file_path, list):
                parsed_data = [self._parse_file(f) for f in self.file_path]
                self.parsed_data = parsed_data
                return parsed_data

            elif isinstance(self.file_path, str) and os.path.isfile(self.file_path):
                parsed_data = self._parse_file(self.file_path)
                self.parsed_data = parsed_data
                return parsed_data

            elif isinstance(self.file_path, str) and os.path.isdir(self.file_path):
                file_list = [os.path.join(self.file_path, file) for file in os.listdir(self.file_path)]
                parsed_data = [self._parse_file(f) for f in file_list if os.path.isfile(f)]
                self.parsed_data = parsed_data
                return parsed_data
            else:
                raise ValueError("Invalid file path.")

        except Exception as e:
            raise e

    def clean_data(self, method_list: List[str], text: str = None):
        """
        Clean data

        methods include AbnormalCleaner， TextFilter， PrivacyDesensitization which is 1 2 3

        :return:
        """
        if text:
            cleaned_text = text
        elif self.parsed_data:
            cleaned_text = self.parsed_data.get('content')
        else:
            raise ValueError("No data to clean.")

        for method in method_list:
            if method == 'abnormal':
                cleaned_text = data_cleaner.AbnormalCleaner(cleaned_text).to_clean().get("text")
            elif method == 'filter':
                cleaned_text = data_cleaner.TextFilter(cleaned_text).to_filter()
                cleaned_text = cleaned_text.get("text") if cleaned_text else ''
            elif method == 'private':
                cleaned_text = data_cleaner.PrivacyDesensitization(cleaned_text).to_private().get("text")

        if self.parsed_data:
            origin_dict = self.parsed_data
            origin_dict['content'] = cleaned_text
            self.parsed_data = None
            return origin_dict
        else:
            return cleaned_text

    def _parse_file(self, file_path):
        """
        Create a parser instance using ParserFactory and parse the file.

        :param file_path: The path to the file to be parsed.
        :return: The parsed data.
        """
        parser = ParserFactory.create_parser(file_path, self.use_ocr, self.use_gpu, self.gpu_id, self.to_markdown)
        if parser:
            return parser.parse(file_path)
