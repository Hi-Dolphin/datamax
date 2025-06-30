import importlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from openai import OpenAI

from datamax.utils import data_cleaner
from datamax.utils.qa_generator import generate_qa_from_content


class ModelInvoker:
    def __init__(self):
        self.client = None

    def invoke_model(self, api_key, base_url, model_name, messages):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        json_data = completion.model_dump()
        return json_data.get("choices")[0].get("message").get("content", "")


class ParserFactory:
    @staticmethod
    def create_parser(
        file_path: str,
        use_mineru: bool = False,
        to_markdown: bool = False,
    ):
        """
        Create a parser instance based on the file extension.
        :param file_path: The path to the file to be parsed.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
                    (only supported files in .doc or .docx format)
        :param use_mineru: Flag to indicate whether MinerU should be used. (only supported files in .pdf format)
        :return: An instance of the parser class corresponding to the file extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        parser_class_name = {
            ".md": "MarkdownParser",
            ".docx": "DocxParser",
            ".doc": "DocParser",
            ".wps": "WpsParser",
            ".epub": "EpubParser",
            ".html": "HtmlParser",
            ".txt": "TxtParser",
            ".pptx": "PPtxParser",
            ".ppt": "PPtParser",
            ".pdf": "PdfParser",
            ".jpg": "ImageParser",
            ".jpeg": "ImageParser",
            ".png": "ImageParser",
            ".webp": "ImageParser",
            ".xlsx": "XlsxParser",
            ".xls": "XlsParser",
        }.get(file_extension)

        if not parser_class_name:
            return None

        if file_extension in [".jpg", "jpeg", ".png", ".webp"]:
            module_name = f"datamax.parser.image_parser"
        else:
            # Dynamically determine the module name based on the file extension
            module_name = f"datamax.parser.{file_extension[1:]}_parser"

        try:
            # Dynamically import the module and get the class
            module = importlib.import_module(module_name)
            parser_class = getattr(module, parser_class_name)

            # Special handling for PdfParser arguments
            if parser_class_name == "PdfParser":
                return parser_class(
                    file_path=file_path,
                    use_mineru=use_mineru,
                )
            elif parser_class_name == "DocxParser" or parser_class_name == "DocParser" or parser_class_name == "WpsParser":
                return parser_class(
                    file_path=file_path, to_markdown=to_markdown, use_uno=True
                )
            elif parser_class_name == "XlsxParser":
                return parser_class(file_path=file_path)
            else:
                return parser_class(file_path=file_path)

        except (ImportError, AttributeError) as e:
            raise e


class DataMax:
    def __init__(
        self,
        file_path: Union[str, list] = "",
        use_mineru: bool = False,
        to_markdown: bool = False,
        ttl: int = 3600,
    ):
        """
        Initialize the DataMaxParser with file path and parsing options.

        :param file_path: The path to the file or directory to be parsed.
        :param use_mineru: Flag to indicate whether MinerU should be used.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
        :param ttl: Time to live for the cache.
        """
        self.file_path = file_path
        self.use_mineru = use_mineru
        self.to_markdown = to_markdown
        self.parsed_data = None
        self.model_invoker = ModelInvoker()
        self._cache = {}
        self.ttl = ttl

    def set_data(self, file_name, parsed_data):
        """
        Set cached data
        :param file_name: File name as cache key
        :param parsed_data: Parsed data as value
        """
        logger.info(f"cache ttl is {self.ttl}s")
        if self.ttl > 0:
            self._cache[file_name] = {
                "data": parsed_data,
                "ttl": time.time() + self.ttl,
            }
            logger.info(
                f"✅ [Cache Updated] Cached data for {file_name}, ttl: {self._cache[file_name]['ttl']}"
            )

    def get_data(self):
        """
        Parse the file or directory specified in the file path and return the data.

        :return: A list of parsed data if the file path is a directory, otherwise a single parsed data.
        """
        try:
            if isinstance(self.file_path, list):
                parsed_data = []
                for f in self.file_path:
                    file_name = os.path.basename(f)
                    if (
                        file_name in self._cache
                        and self._cache[file_name]["ttl"] > time.time()
                    ):
                        logger.info(f"✅ [Cache Hit] Using cached data for {file_name}")
                        parsed_data.append(self._cache[file_name]["data"])
                    else:
                        logger.info(
                            f"⏳ [Cache Miss] No cached data for {file_name}, parsing..."
                        )
                        self._cache = {
                            k: v
                            for k, v in self._cache.items()
                            if v["ttl"] > time.time()
                        }
                        res_data = self._parse_file(f)
                        parsed_data.append(res_data)
                        self.set_data(file_name, res_data)
                return parsed_data

            elif isinstance(self.file_path, str) and os.path.isfile(self.file_path):
                file_name = os.path.basename(self.file_path)
                if (
                    file_name in self._cache
                    and self._cache[file_name]["ttl"] > time.time()
                ):
                    logger.info(f"✅ [Cache Hit] Using cached data for {file_name}")
                    return self._cache[file_name]["data"]
                else:
                    logger.info(
                        f"⏳ [Cache Miss] No cached data for {file_name}, parsing..."
                    )
                    self._cache = {
                        k: v for k, v in self._cache.items() if v["ttl"] > time.time()
                    }
                    parsed_data = self._parse_file(self.file_path)
                    self.parsed_data = parsed_data
                    self.set_data(file_name, parsed_data)
                    return parsed_data

            elif isinstance(self.file_path, str) and os.path.isdir(self.file_path):
                file_list = [
                    str(file) for file in list(Path(self.file_path).rglob("*.*"))
                ]
                parsed_data = []
                for f in file_list:
                    if os.path.isfile(f):
                        file_name = os.path.basename(f)
                        if (
                            file_name in self._cache
                            and self._cache[file_name]["ttl"] > time.time()
                        ):
                            logger.info(
                                f"✅ [Cache Hit] Using cached data for {file_name}"
                            )
                            parsed_data.append(self._cache[file_name]["data"])
                        else:
                            logger.info(
                                f"⏳ [Cache Miss] No cached data for {file_name}, parsing..."
                            )
                            self._cache = {
                                k: v
                                for k, v in self._cache.items()
                                if v["ttl"] > time.time()
                            }
                            res_data = self._parse_file(f)
                            parsed_data.append(res_data)
                            self.set_data(file_name, res_data)
                return parsed_data
            else:
                raise ValueError("Invalid file path.")

        except Exception as e:
            raise e

    def clean_data(self, method_list: List[str], text: str = None):
        """
        Clean data

        methods include AbnormalCleaner, TextFilter, PrivacyDesensitization which are 1, 2, 3

        :return: Cleaned data
        """
        if text:
            cleaned_text = text
        elif self.parsed_data:
            cleaned_text = self.parsed_data.get("content")
        else:
            raise ValueError("No data to clean.")

        for method in method_list:
            if method == "abnormal":
                cleaned_text = (
                    data_cleaner.AbnormalCleaner(cleaned_text).to_clean().get("text")
                )
            elif method == "filter":
                cleaned_text = data_cleaner.TextFilter(cleaned_text).to_filter()
                cleaned_text = cleaned_text.get("text") if cleaned_text else ""
            elif method == "private":
                cleaned_text = (
                    data_cleaner.PrivacyDesensitization(cleaned_text)
                    .to_private()
                    .get("text")
                )

        if self.parsed_data:
            origin_dict = self.parsed_data
            origin_dict["content"] = cleaned_text
            self.parsed_data = None
            return origin_dict
        else:
            return cleaned_text

    def get_pre_label(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        question_number: int = 5,
        max_workers: int = 5,
        language: str = "zh",
        messages: List[Dict[str, str]] = None,
    ):
        """
        Generate pre-labeling data based on processed document content instead of file path

        :param api_key: API key
        :param base_url: API base URL
        :param model_name: Model name
        :param chunk_size: Chunk size
        :param chunk_overlap: Overlap length
        :param question_number: Number of questions generated per chunk
        :param max_workers: Number of concurrent workers
        :param language: Language for QA generation ("zh" for Chinese, "en" for English)
        :param messages: Custom messages
        :return: List of QA pairs
        """
        # First get the processed data
        processed_data = self.get_data()

        # If it's a list (multiple files), merge all content
        if isinstance(processed_data, list):
            content_list = []
            for data in processed_data:
                if isinstance(data, dict) and "content" in data:
                    content_list.append(data["content"])
                elif isinstance(data, str):
                    content_list.append(data)
            content = "\n\n".join(content_list)
        # If it's a dictionary for a single file
        elif isinstance(processed_data, dict) and "content" in processed_data:
            content = processed_data["content"]
        # If it's a string
        elif isinstance(processed_data, str):
            content = processed_data
        else:
            raise ValueError("Unable to extract content field from processed data")

        # Generate QA pairs using content instead of reading files
        return generate_qa_from_content(
            content=content,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            language=language,
            max_workers=max_workers,
            message=messages,
        )

    def save_label_data(self, label_data: list, save_file_name: str = None):
        """
        Save label data to file.
        :param label_data: Label data to be saved.
        :param save_file_name: File name to save the label data.
        """
        if not label_data:
            raise ValueError("No data to save.")
        if not save_file_name:
            if isinstance(self.file_path, str):
                save_file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            else:
                save_file_name = "label_data"
        if isinstance(label_data, list):
            with open(save_file_name + ".jsonl", "w", encoding="utf-8") as f:
                for qa_entry in label_data:
                    f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
            logger.info(
                f"✅ [Label Data Saved] Label data saved to {save_file_name}.jsonl"
            )

    @staticmethod
    def split_text_into_paragraphs(
        text: str, max_length: int = 500, chunk_overlap: int = 100
    ):
        """
        Split text into paragraphs by sentence boundaries, each paragraph not exceeding max_length characters.
        Paragraphs will have chunk_overlap characters of overlap between them.
        """
        import re

        # Split sentences using Chinese punctuation marks
        sentences = re.split("(?<=[。！？])", text)
        paragraphs = []
        current_paragraph = ""
        overlap_buffer = ""

        for sentence in sentences:
            # If current paragraph plus new sentence doesn't exceed max length
            if len(current_paragraph) + len(sentence) <= max_length:
                current_paragraph += sentence
            else:
                if current_paragraph:
                    # Add current paragraph to results
                    paragraphs.append(current_paragraph)
                    # Save overlap portion
                    overlap_buffer = (
                        current_paragraph[-chunk_overlap:] if chunk_overlap > 0 else ""
                    )
                # Start new paragraph with overlap
                current_paragraph = overlap_buffer + sentence
                overlap_buffer = ""

                # Handle overly long sentences
                while len(current_paragraph) > max_length:
                    # Split long paragraph
                    split_point = max_length - len(overlap_buffer)
                    paragraphs.append(current_paragraph[:split_point])
                    # Update overlap buffer
                    overlap_buffer = (
                        current_paragraph[split_point - chunk_overlap : split_point]
                        if chunk_overlap > 0
                        else ""
                    )
                    current_paragraph = overlap_buffer + current_paragraph[split_point:]
                    overlap_buffer = ""

        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)

        return paragraphs

    @staticmethod
    def split_with_langchain(
        text: str, chunk_size: int = 500, chunk_overlap: int = 100
    ):
        """
        Split text using LangChain's intelligent text splitting

        :param text: Text to be split
        :param chunk_size: Maximum length of each chunk
        :param chunk_overlap: Number of overlapping characters between chunks
        :return: List of split text
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_text(text)

    def split_data(
        self,
        parsed_data: Union[str, dict] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_langchain: bool = False,
    ):
        """
        Improved splitting method with LangChain option

        :param use_langchain: Whether to use LangChain for splitting
        :param parsed_data: Data to be split, either string or dict
        :param chunk_size: Maximum length of each chunk
        :param chunk_overlap: Number of overlapping characters between chunks
        :return: List or dict of split text
        """
        if parsed_data:
            self.parsed_data = parsed_data
        if not self.parsed_data:
            raise ValueError("No data to split.")

        if use_langchain:
            if isinstance(self.parsed_data, str):
                return self.split_with_langchain(
                    self.parsed_data, chunk_size, chunk_overlap
                )
            elif isinstance(self.parsed_data, dict):
                if "content" not in self.parsed_data:
                    raise ValueError("Input dict must contain 'content' key")
                chunks = self.split_with_langchain(
                    self.parsed_data["content"], chunk_size, chunk_overlap
                )
                result = self.parsed_data.copy()
                result["content"] = chunks
                return result

        # Handle string input
        if isinstance(self.parsed_data, str):
            return self.split_text_into_paragraphs(
                self.parsed_data, chunk_size, chunk_overlap
            )

        # Handle dict input
        elif isinstance(self.parsed_data, dict):
            if "content" not in self.parsed_data:
                raise ValueError("Input dict must contain 'content' key")

            content = self.parsed_data["content"]
            chunks = self.split_text_into_paragraphs(content, chunk_size, chunk_overlap)

            result = self.parsed_data.copy()
            result["content"] = chunks
            return result
        else:
            raise ValueError("Unsupported input type")

    def _parse_file(self, file_path):
        """
        Create a parser instance using ParserFactory and parse the file.

        :param file_path: The path to the file to be parsed.
        :return: The parsed data.
        """
        try:
            parser = ParserFactory.create_parser(
                use_mineru=self.use_mineru,
                file_path=file_path,
                to_markdown=self.to_markdown,
            )
            if parser:
                return parser.parse(file_path=file_path)
        except Exception as e:
            raise e


if __name__ == "__main__":
    pass
