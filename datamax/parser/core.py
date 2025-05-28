import os
import importlib
from typing import List, Union, Dict
from openai import OpenAI
from datamax.utils import data_cleaner
from datamax.utils.qa_generator import generatr_qa_pairs


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
            timeout: int = 1200
    ):
        """
        Create a parser instance based on the file extension.
        :param file_path: The path to the file to be parsed.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
                    (only supported files in .doc or .docx format)
        :param use_mineru: Flag to indicate whether MinerU should be used. (only supported files in .pdf format)
        :param timeout: Timeout for the request .(only supported files in .xlsx format)
        :return: An instance of the parser class corresponding to the file extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        parser_class_name = {
            '.md': 'MarkdownParser',
            '.docx': 'DocxParser',
            '.doc': 'DocParser',
            '.epub': 'EpubParser',
            '.html': 'HtmlParser',
            '.txt': 'TxtParser',
            '.pptx': 'PPtxParser',
            '.ppt': 'PPtParser',
            '.pdf': 'PdfParser',
            '.jpg': 'ImageParser',
            '.jpeg': 'ImageParser',
            '.png': 'ImageParser',
            '.webp': 'ImageParser',
            '.xlsx': 'XlsxParser',
            '.xls': 'XlsParser'
        }.get(file_extension)

        if not parser_class_name:
            return None

        if file_extension in ['.jpg', 'jpeg', '.png', '.webp']:
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
                return parser_class(
                    file_path=file_path,
                    use_mineru=use_mineru,
                )
            elif parser_class_name == 'DocxParser' or parser_class_name == 'DocParser':
                return parser_class(
                    file_path=file_path, to_markdown=to_markdown
                )
            elif parser_class_name == 'XlsxParser':
                return parser_class(
                    file_path=file_path,
                    timeout=timeout
                )
            else:
                return parser_class(
                    file_path=file_path
                )

        except (ImportError, AttributeError) as e:
            raise e


class DataMax:
    def __init__(self,
                 file_path: Union[str, list] = '',
                 use_mineru: bool = False,
                 to_markdown: bool = False,
                 timeout: int = 1200
                 ):
        """
        Initialize the DataMaxParser with file path and parsing options.

        # <Abandon>
        # :param use_paddle_ocr: Flag to indicate whether PaddleOCR should be used.
        # :param use_paddle_gpu: Flag to indicate whether PaddleOCR-GPU should be used.
        # :param use_got_ocr: Flag to indicate whether GOT-OCR should be used.
        # :param got_weights_path: GOT-OCR Weights Path.
        # :param gpu_id: The ID of the GPU to use.

        :param file_path: The path to the file or directory to be parsed.
        :param use_mineru: Flag to indicate whether MinerU should be used.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
        """
        self.file_path = file_path
        self.use_mineru = use_mineru
        self.to_markdown = to_markdown
        self.parsed_data = None
        self.model_invoker = ModelInvoker()
        self.timeout = timeout

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

    def get_pre_label(self,
                      api_key: str,
                      base_url: str,
                      model_name: str,
                      chunk_size: int = 500,
                      chunk_overlap: int = 100,
                      question_number: int = 5,
                      max_workers: int = 5,
                      messages: List[Dict[str, str]] = None):
        return generatr_qa_pairs(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            max_workers=max_workers,
            message=messages,
            file_path=self.file_path
        )

    ## <Abandon>
    # def enhance_with_model(self, api_key: str, base_url: str, model_name: str, iteration: int = 1,
    #                        messages: List[Dict[str, str]] = None):
    #     """
    #     Enhance the parsed content using a large language model.
    #
    #     :param api_key: API key for the large model service.
    #     :param base_url: Base URL for the large model service.
    #     :param model_name: Name of the model to use.
    #     :param iteration: Number of iterations
    #     :param messages: Custom messages list [{"role": "system", "content": "..."}, ...]
    #     :return: Enhanced text.
    #     """
    #     if not messages:
    #         # If no custom message is provided, the default message structure is used, but only if there is parsed data
    #         if self.parsed_data:
    #             system_prompt = get_system_prompt(self.parsed_data)
    #             default_message_user = {"role": "user", "content": "按照json格式给出问答对"}
    #             messages = [
    #                 {"role": "system", "content": system_prompt},
    #                 default_message_user
    #             ]
    #         else:
    #             raise ValueError("No data to enhance and no custom messages provided.")
    #     try:
    #         if isinstance(iteration, int) and iteration >= 1:
    #             results = []
    #             current_messages = messages.copy()  # Avoid modifying the original message during iteration
    #
    #             for _ in range(iteration):
    #                 enhanced_text = self.model_invoker.invoke_model(
    #                     api_key=api_key,
    #                     base_url=base_url,
    #                     model_name=model_name,
    #                     messages=current_messages
    #                 )
    #
    #                 # Append the generated content to the conversation history in multiple iterations
    #                 if iteration > 1:
    #                     current_messages.append({"role": "assistant", "content": enhanced_text})
    #                     current_messages.append(
    #                         {"role": "user", "content": "请继续生成, 生成要求不变, 结果是jsonlist, 且长度不超过5"})
    #
    #                 # If there is parsed data, update the contents and return a copy of the original dictionary; Otherwise, return the enhanced text directly
    #                 if self.parsed_data:
    #                     origin_dict = self.parsed_data.copy()
    #                     origin_dict['content'] = enhanced_text
    #                     results.append(origin_dict)
    #                 else:
    #                     results.append(enhanced_text)
    #
    #             return results if iteration > 1 else results[0]
    #         else:
    #             raise ValueError("Invalid iteration parameter.")
    #     except Exception as e:
    #         raise Exception(f"An error occurred while enhancing with the model: {e}")

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
                timeout=self.timeout
            )
            if parser:
                return parser.parse(file_path=file_path)
        except Exception as e:
            raise e


if __name__ == '__main__':
    pass
