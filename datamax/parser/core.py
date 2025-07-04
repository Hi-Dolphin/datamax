import os
import json
import time
import importlib
from loguru import logger
from typing import List, Union, Dict
from openai import OpenAI
from pathlib import Path
from datamax.utils import data_cleaner
from datamax.utils.qa_generator import generatr_qa_pairs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datamax.utils.domain_tree import DomainTree


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
                 timeout: int = 1200,
                 ttl: int = 3600
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
        :param timeout: Timeout for the request.
        :param ttl: Time to live for the cache.
        """
        self.file_path = file_path
        self.use_mineru = use_mineru
        self.to_markdown = to_markdown
        self.parsed_data = None
        self.model_invoker = ModelInvoker()
        self.timeout = timeout
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
            self._cache[file_name] = {'data': parsed_data, 'ttl': time.time() + self.ttl}
            logger.info(f"✅ [Cache Updated] Cached data for {file_name}, ttl: {self._cache[file_name]['ttl']}s")

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
                    if file_name in self._cache and self._cache[file_name]['ttl'] > time.time():
                        logger.info(f"✅ [Cache Hit] Using cached data for {file_name}")
                        parsed_data.append(self._cache[file_name]['data'])
                    else:
                        logger.info(f"⏳ [Cache Miss] No cached data for {file_name}, parsing...")
                        self._cache = {k: v for k, v in self._cache.items() if v['ttl'] > time.time()}
                        res_data = self._parse_file(f)
                        parsed_data.append(res_data)
                        self.set_data(file_name, res_data)
                return parsed_data

            elif isinstance(self.file_path, str) and os.path.isfile(self.file_path):
                file_name = os.path.basename(self.file_path)
                if file_name in self._cache and self._cache[file_name]['ttl'] > time.time():
                    logger.info(f"✅ [Cache Hit] Using cached data for {file_name}")
                    return self._cache[file_name]['data']
                else:
                    logger.info(f"⏳ [Cache Miss] No cached data for {file_name}, parsing...")
                    self._cache = {k: v for k, v in self._cache.items() if v['ttl'] > time.time()}
                    parsed_data = self._parse_file(self.file_path)
                    self.parsed_data = parsed_data
                    self.set_data(file_name, parsed_data)
                    return parsed_data

            elif isinstance(self.file_path, str) and os.path.isdir(self.file_path):
                file_list = [str(file) for file in list(Path(self.file_path).rglob('*.*'))]
                parsed_data = []
                for f in file_list:
                    if os.path.isfile(f):
                        file_name = os.path.basename(f)
                        if file_name in self._cache and self._cache[file_name]['ttl'] > time.time():
                            logger.info(f"✅ [Cache Hit] Using cached data for {file_name}")
                            parsed_data.append(self._cache[file_name]['data'])
                        else:
                            logger.info(f"⏳ [Cache Miss] No cached data for {file_name}, parsing...")
                            self._cache = {k: v for k, v in self._cache.items() if v['ttl'] > time.time()}
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

        methods include AbnormalCleaner， TextFilter， PrivacyDesensitization which is 1 2 3

        :return:
        """
        if text:
            cleaned_text = text
        elif self.parsed_data:
            if isinstance(self.parsed_data, dict):
                cleaned_text = self.parsed_data.get('content', '')
            else:
                cleaned_text = str(self.parsed_data)
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

    def generate_qa(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        question_number: int = 5,
        max_workers: int = 5,
        messages: list = None,
        use_tree_label: bool = True
    ) -> list:
        """
        Generate QA pairs from document with optional domain tree labeling
        :param api_key: Api key read from .env
        :param base_url: Base url read from .env
        :param model_name: Model name user chooses
        :param chunk_size: Maximum length of each chunk
        :param chunk_overlap: Number of overlapping characters between chunks
        :param question_number: Question number wish to generate
        :param max_workers: Max workers for multi-threading
        :param messages: Messages for model
        :param use_tree_label: Whether to use domain tree label for generating questions
        :return: List of QA pairs
        """
        from datamax.utils.qa_generator import (
            load_and_split_markdown,
            load_and_split_text,
            process_domain_tree,
            process_questions,
            process_match_tags,
            generatr_qa_pairs
        )
        import uuid
        
        # 检查文件路径类型
        if isinstance(self.file_path, list):
            if len(self.file_path) == 0:
                raise ValueError("文件路径列表为空")
            file_path = self.file_path[0]  # 取第一个文件
        else:
            file_path = self.file_path
            
        # 检查文件扩展名
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # 1. split - 根据文件类型选择不同的处理方法
        if file_extension == '.md':
            page_content = load_and_split_markdown(
                md_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # 对于非markdown格式，使用通用文本处理函数
            page_content = load_and_split_text(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
        if not page_content:
            raise ValueError("文档切分失败或内容为空")

        # 2. generate domain tree (only if use_tree_label is True)
        domain_tree = None
        if use_tree_label:
            domain_tree = process_domain_tree(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                text="\n".join(page_content),
                temperature=0.7,
                top_p=0.9,
            )
            
            # visualization
            if domain_tree and domain_tree.tree:
                print("\n" + "="*60)
                print("🌳 生成的领域树结构:")
                print("="*60)
                print(domain_tree.visualize())
                print("="*60)
                
                # interactive tree modification
                domain_tree = self._interactive_tree_modification(domain_tree)
        
        # 3. generate questions
        question_info = process_questions(
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            page_content=page_content,
            question_number=question_number,
            max_workers=max_workers,
            message=messages,
        )
        # add qid
        for question_item in question_info:
            if "qid" not in question_item:
                question_item["qid"] = str(uuid.uuid4())

        # 4. match tags (only if use_tree_label is True)
        if use_tree_label and domain_tree and hasattr(domain_tree, 'to_json') and domain_tree.to_json():
            q_match_list = process_match_tags(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                tags_json=domain_tree.to_json(),
                questions=[q["question"] for q in question_info],
                max_workers=max_workers
            )
            label_map = {item["question"]: item.get("label", "") for item in q_match_list}
            for question_item in question_info:
                question_item["label"] = label_map.get(question_item["question"], "")
        else:
            # If not using tree label, set empty labels
            for question_item in question_info:
                question_item["label"] = ""

        # 5. generate answers
        qa_list = generatr_qa_pairs(
            question_info=question_info,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            question_number=question_number,
            max_workers=max_workers,
            domain_tree=domain_tree if use_tree_label else None
        )
        return qa_list

    def _interactive_tree_modification(self, domain_tree:DomainTree):
        """
        interactively customize domain tree
        
        :param domain_tree: origin DomainTree instance      
        :return customized DomainTree instance
        """
        print("\n 是否需要进行树修改？")
        print("支持的操作:")
        print("1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）")
        print("2. 增加节点：xxx；父节点：xxx；子节点：xxx")
        print("3. 删除节点：xxx")
        print("4. 更新节点：新名称；原先节点：旧名称")
        print("5. 结束树操作")
        print("注意，节点的格式通常为：x.xx xxxx,如：‘1.1 货物运输组织与路径规划’或‘1 运输系统组织’")
        print("\n请输入操作指令（输入'结束树操作'退出）:")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input == "结束树操作":
                    print("✅ 树操作结束，继续QA对生成...")
                    break
                
                elif user_input.startswith("增加节点："):
                    # parse add node instruction
                    parts = user_input.split("；")
                    if len(parts) >= 2:
                        node_name = parts[0].replace("增加节点：", "").strip()
                        parent_name = parts[1].replace("父节点：", "").strip()
                        if not parent_name:
                            # 父节点为空，作为根节点添加
                            if domain_tree.add_node(node_name):
                                print(f"✅ 成功将节点 '{node_name}' 作为根节点添加")
                            else:
                                print(f"❌ 添加失败：未知错误")
                        elif len(parts) == 2:
                            if domain_tree.add_node(node_name, parent_name):
                                print(f"✅ 成功添加节点 '{node_name}' 到父节点 '{parent_name}' 下")
                            else:
                                print(f"❌ 添加失败：未找到父节点 '{parent_name}'")
                        elif len(parts) == 3:
                            # case 2: insert between parent and child node
                            child_name = parts[2].replace("子节点：", "").strip()
                            if domain_tree.insert_node_between(node_name, parent_name, child_name):
                                print(f"✅ 成功插入节点 '{node_name}' 到 '{parent_name}' 和 '{child_name}' 之间")
                            else:
                                print(f"❌ 插入失败：请检查父节点和子节点的关系")
                        else:
                            print("❌ 格式错误：请使用正确的格式")
                    else:
                        print("❌ 格式错误：请使用正确的格式")
                
                elif user_input.startswith("删除节点："):
                    # parse delete node instruction
                    node_name = user_input.replace("删除节点：", "").strip()
                    if domain_tree.remove_node(node_name):
                        print(f"✅ 成功删除节点 '{node_name}' 及其所有子孙节点")
                    else:
                        print(f"❌ 删除失败：未找到节点 '{node_name}'")
                
                elif user_input.startswith("更新节点："):
                    parts = user_input.split("；")
                    if len(parts) == 2:
                        new_name = parts[0].replace("更新节点：", "").strip()
                        old_name = parts[1].replace("原先节点：", "").strip()
                        if domain_tree.update_node(old_name, new_name):
                            print(f"✅ 成功将节点 '{old_name}' 更新为 '{new_name}'")
                        else:
                            print(f"❌ 更新失败：未找到节点 '{old_name}'")
                    else:
                        print("❌ 格式错误：请使用正确的格式，如：更新节点：新名称；原先节点：旧名称")
                
                else:
                    print("❌ 未知操作，请使用正确的格式")
                    
                # show modified tree structure
                print("\n📝 当前树结构:")
                print(domain_tree.visualize())
                print("\n请输入下一个操作指令:")
                print("支持的操作:")
                print("1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）")
                print("2. 增加节点：xxx；父节点：xxx；子节点：xxx")
                print("3. 删除节点：xxx")
                print("4. 更新节点：新名称；原先节点：旧名称")
                print("5. 结束树操作")
                print("注意，节点的格式通常为：x.xx xxxx,如：‘1.1 货物运输组织与路径规划’或‘1 运输系统组织’")

                
            except KeyboardInterrupt:
                print("\n\n⚠️⚠️操作被中断⚠️⚠️，继续QA对生成...")
                break
            except Exception as e:
                print(f"❌ 操作出错：{e}")
                print("请重新输入操作指令:")
        return domain_tree

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
                save_file_name = 'label_data'
        if isinstance(label_data, list):
            with open(save_file_name + '.jsonl', 'w', encoding='utf-8') as f:
                for qa_entry in label_data:
                    f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
            logger.info(f"✅ [Label Data Saved] Label data saved to {save_file_name}.jsonl")


    @staticmethod 
    def split_text_into_paragraphs(text: str, max_length:int = 500, chunk_overlap: int = 100):
        """
        Split text into paragraphs by sentence boundaries, each paragraph not exceeding max_length characters.
        Paragraphs will have chunk_overlap characters of overlap between them.
        """
        import re 

        # Split sentences using Chinese punctuation marks
        sentences = re.split('(?<=[。！？])', text)
        paragraphs = []
        current_paragraph = ''
        overlap_buffer = ''

        for sentence in sentences:
            # If current paragraph plus new sentence doesn't exceed max length
            if len(current_paragraph) + len(sentence) <= max_length:
                current_paragraph += sentence
            else:
                if current_paragraph:
                    # Add current paragraph to results
                    paragraphs.append(current_paragraph)
                    # Save overlap portion
                    overlap_buffer = current_paragraph[-chunk_overlap:] if chunk_overlap > 0 else ''
                # Start new paragraph with overlap
                current_paragraph = overlap_buffer + sentence
                overlap_buffer = ''
                
                # Handle overly long sentences
                while len(current_paragraph) > max_length:
                    # Split long paragraph
                    split_point = max_length - len(overlap_buffer)
                    paragraphs.append(current_paragraph[:split_point])
                    # Update overlap buffer
                    overlap_buffer = current_paragraph[split_point - chunk_overlap:split_point] if chunk_overlap > 0 else ''
                    current_paragraph = overlap_buffer + current_paragraph[split_point:]
                    overlap_buffer = ''

        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)

        return paragraphs

    @staticmethod
    def split_with_langchain(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
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
            parsed_data: Union[str, dict, None] = None,
            chunk_size: int = 500,
            chunk_overlap: int = 100,
            use_langchain: bool = False):
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
                return self.split_with_langchain(self.parsed_data, chunk_size, chunk_overlap)
            elif isinstance(self.parsed_data, dict):
                if 'content' not in self.parsed_data:
                    raise ValueError("Input dict must contain 'content' key")
                chunks = self.split_with_langchain(self.parsed_data['content'], chunk_size, chunk_overlap)
                result = self.parsed_data.copy()
                result['content'] = chunks
                return result
        
        # Handle string input
        if isinstance(self.parsed_data, str):
            return self.split_text_into_paragraphs(self.parsed_data, chunk_size, chunk_overlap)
        
        # Handle dict input
        elif isinstance(self.parsed_data, dict):
            if 'content' not in self.parsed_data:
                raise ValueError("Input dict must contain 'content' key")
                
            content = self.parsed_data['content']
            chunks = self.split_text_into_paragraphs(content, chunk_size, chunk_overlap)
                
            result = self.parsed_data.copy()
            result['content'] = chunks
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
                timeout=self.timeout
            )
            if parser:
                return parser.parse(file_path=file_path)
        except Exception as e:
            raise e


if __name__ == "__main__":
   pass