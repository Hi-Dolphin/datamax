import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

import datamax.generator.qa_generator as qa_generator
from datamax.cleaner import data_cleaner
from datamax.parser.base import BaseLife
from datamax.utils.lifecycle_types import LifeType
from datamax.utils.debug_logger import DebugContext


class ParserFactory:
    @staticmethod
    def create_parser(
        file_path: str,
        mllm_system_prompt: str,
        use_mineru: bool = False,
        use_qwen_vl_ocr: bool = False,
        use_mllm: bool = False,
        to_markdown: bool = False,
        domain: str = "Technology",
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
    ):
        """
        Create a parser instance based on the file extension.
        :param file_path: The path to the file to be parsed.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
                    (only supported files in .doc or .docx format)
        :param use_mineru: Flag to indicate whether MinerU should be used. (only supported files in .pdf format)
        :param use_qwen_vl_ocr: Flag to indicate whether Qwen-VL OCR should be used. (only supported files in .pdf format)
        :param use_mllm: Flag to indicate whether MLLM should be used. (only supported files in .jpg, .jpeg, .png, .webp format)
        :param mllm_system_prompt: System prompt for MLLM.
        :param api_key: API key for OCR service (required when use_qwen_vl_ocr=True).
        :param base_url: Base URL for OCR service (required when use_qwen_vl_ocr=True).
        :param model_name: Model name for OCR service (required when use_qwen_vl_ocr=True).
        :return: An instance of the parser class corresponding to the file extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        # Define extension groups
        image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        code_extensions = [
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".cpp",
            ".cc",
            ".cxx",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".cs",
            ".swift",
            ".kt",
            ".scala",
        ]

        # Mapping of extensions to (class_name, module_name)
        parser_map = {}
        for ext in image_extensions:
            parser_map[ext] = ("ImageParser", "datamax.parser.image_parser")
        for ext in code_extensions:
            parser_map[ext] = ("CodeParser", "datamax.parser.code_parser")

        # Add other parsers
        document_parsers = {
            ".md": "MarkdownParser",
            ".docx": "DocxParser",
            ".doc": "DocParser",
            ".wps": "WpsParser",
            ".epub": "EpubParser",
            ".html": "HtmlParser",
            ".txt": "TxtParser",
            ".pptx": "PptxParser",
            ".ppt": "PptParser",
            ".pdf": "PdfParser",
            ".xlsx": "XlsxParser",
            ".xls": "XlsParser",
            ".csv": "CsvParser",
            ".json": "JsonParser"
        }
        for ext, class_name in document_parsers.items():
            module_name = f"datamax.parser.{ext[1:]}_parser"
            parser_map[ext] = (class_name, module_name)

        mapping = parser_map.get(file_extension)
        if not mapping:
            return None

        parser_class_name, module_name = mapping

        try:
            if use_mineru and use_mllm:
                raise ValueError(
                    "You must choose between the Mineru and MLLM solutions - they cannot be used at the same time!"
                )
            # use_mineru & use_qwen_vl_ocr can't be used at the same time
            if use_mineru and use_qwen_vl_ocr:
                raise ValueError(
                    "You must choose between the Mineru and Qwen-VL-OCR solutions - they cannot be used at the same time!"
                )

            if mllm_system_prompt and use_mllm and parser_class_name != "ImageParser":
                raise ValueError(
                    "MLLM can only be used with Image type temporarily, try to use Mineru or Qwen-VL-OCR instead, ``use_mineru=True`` or ``use_qwen_vl_ocr=True``"
                )

            # Dynamically import the module and get the class
            module = importlib.import_module(module_name)
            parser_class = getattr(module, parser_class_name)

            # Instantiate based on parser type
            common_kwargs = {"file_path": file_path, "domain": domain}
            if parser_class_name == "PdfParser":
                return parser_class(
                    use_mineru=use_mineru,
                    use_qwen_vl_ocr=use_qwen_vl_ocr,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    **common_kwargs,
                )
            elif parser_class_name in ["DocxParser", "DocParser", "WpsParser"]:
                return parser_class(
                    to_markdown=to_markdown, use_uno=True, **common_kwargs
                )
            elif parser_class_name == "ImageParser":
                return parser_class(
                    use_mllm=use_mllm,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    system_prompt=mllm_system_prompt,
                    **common_kwargs,
                )
            else:
                return parser_class(**common_kwargs)

        except (ImportError, AttributeError) as e:
            raise e


class DataMax(BaseLife):
    def __init__(
        self,
        file_path: str | list = "",
        use_mineru: bool = False,
        use_qwen_vl_ocr: bool = False,
        use_mllm: bool = False,
        mllm_system_prompt: str = "描述图片内容，包括图片中的文字、图片中的对象、图片中的场景等。输出一份专业的中文markdown报告",
        to_markdown: bool = False,
        ttl: int = 3600,
        domain: str = "Technology",
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
    ):
        """
        Initialize the DataMaxParser with file path and parsing options.

        :param file_path: The path to the file or directory to be parsed.
        :param use_mineru: Flag to indicate whether MinerU should be used for PDF or image parsing.
        :param use_qwen_vl_ocr: Flag to indicate whether Qwen-VL OCR should be used for only PDF parsing.
        :param use_mllm: Flag to indicate whether MLLM should be used for only image parsing.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
        :param ttl: Time to live for the cache.
        :param api_key: API key for OCR service (required when use_qwen_vl_ocr=True).
        :param base_url: Base URL for OCR service (required when use_qwen_vl_ocr=True).
        :param model_name: Model name for OCR service (required when use_qwen_vl_ocr=True).
        """
        super().__init__(domain=domain)
        self.file_path = file_path
        self.use_mineru = use_mineru
        self.use_qwen_vl_ocr = use_qwen_vl_ocr
        self.use_mllm = use_mllm
        self.mllm_system_prompt = mllm_system_prompt
        self.to_markdown = to_markdown
        self.parsed_data = None
        self._cache = {}
        self.ttl = ttl
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

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
                    self.parsed_data = self._cache[file_name]["data"]
                    return self.parsed_data
                else:
                    logger.info(
                        f"⏳ [Cache Miss] No cached data for {self.file_path}, parsing..."
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

    def clean_data(self, method_list: list[str], text: str = None):
        """
        Clean data

        methods include AbnormalCleaner, TextFilter, PrivacyDesensitization which are 1, 2, 3

        :return: Cleaned data
        """
        # 1) Prepare original content
        if text:
            cleaned_text = text
        elif self.parsed_data:
            cleaned_text = self.parsed_data.get("content")
        else:
            raise ValueError("No data to clean.")
        # 2) Trigger "cleaning start"
        lc_start = self.generate_lifecycle(
            source_file=self.file_path,
            domain=self.domain,
            life_type=LifeType.DATA_CLEANING,
            usage_purpose="Data Cleaning",
        ).to_dict()

        try:
            # 3) Execute cleaning steps
            for method in method_list:
                if method == "abnormal":
                    cleaned_text = (
                        data_cleaner.AbnormalCleaner(cleaned_text)
                        .to_clean()
                        .get("text")
                    )
                elif method == "filter":
                    cleaned_text = (
                        data_cleaner.TextFilter(cleaned_text)
                        .to_filter()
                        .get("text", "")
                    )
                elif method == "private":
                    cleaned_text = (
                        data_cleaner.PrivacyDesensitization(cleaned_text)
                        .to_private()
                        .get("text")
                    )

            # 4) Cleaning successful, trigger "cleaning completed"
            lc_end = self.generate_lifecycle(
                source_file=self.file_path,
                domain=self.domain,
                life_type=LifeType.DATA_CLEANED,
                usage_purpose="Data Cleaning",
            ).to_dict()

        except Exception as e:
            # 5) Cleaning failed, trigger "cleaning failed"
            lc_fail = self.generate_lifecycle(
                source_file=self.file_path,
                domain=self.domain,
                life_type=LifeType.DATA_CLEAN_FAILED,
                usage_purpose="Data Cleaning",
            ).to_dict()
            # Add failure event to parsed_data before raising
            if self.parsed_data and isinstance(self.parsed_data, dict):
                self.parsed_data.setdefault("lifecycle", []).append(lc_start)
                self.parsed_data["lifecycle"].append(lc_fail)
            raise

        # 6) Update content and merge lifecycles
        if self.parsed_data and isinstance(self.parsed_data, dict):
            origin = self.parsed_data
            origin["content"] = cleaned_text
            origin.setdefault("lifecycle", []).extend([lc_start, lc_end])
            # Reset parsed_data to avoid secondary contamination
            self.parsed_data = None
            return origin
        else:
            # When returning plain text, also return lifecycle information
            return cleaned_text

    def complete_api_url(self, base_url):
        """
        Automatically complete the API URL path for the website

        rules:
            1. /chat/completions as default endpoint
            2. Only add version if not already present in path
        """
        from datamax.generator.qa_generator import complete_api_url
        return complete_api_url(base_url)

    def get_pre_label(
        self,
        *,
        content: str = None,
        use_mllm: bool = False,
        api_key: str,
        base_url: str,
        model_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        question_number: int = 5,
        max_workers: int = 5,
        language: str = "zh",
        use_tree_label: bool = False,
        messages: list = None,
        interactive_tree: bool = False,
        custom_domain_tree: list[dict[str, Any]] | None = None,
        debug: bool = False,
        structured_data: bool = False,
        auto_self_review_mode: bool = False,
        checkpoint_path: str | None = None,
        resume_from_checkpoint: bool = True,
    ):
        """
        Generate pre-labeling data based on processed document content instead of file path

        :param content: Processed document content
        :param use_mllm: Whether to use mllm model
        :param api_key: API key
        :param base_url: API base URL
        :param model_name: Model name
        :param chunk_size: Chunk size
        :param chunk_overlap: Overlap length
        :param question_number: Number of questions generated per chunk
        :param max_workers: Number of concurrent workers
        :param language: Language for QA generation ("zh" for Chinese, "en" for English)
        :param use_tree_label: Whether to use domain tree label for generating questions
        :param messages: Custom messages
        :param interactive_tree: Whether to allow interactive tree modification
        :param custom_domain_tree: Custom domain tree structure in the format:
            [
                {
                    "label": "1 一级领域标签",
                    "child": [
                        {"label": "1.1 二级领域标签1"},
                        {"label": "1.2 二级领域标签2"}
                    ]
                },
                {
                    "label": "2 一级领域标签(无子标签)"
                }
            ]
        :param debug: Enable debug logging
        :param structured_data: Whether to use structured data format
        :param auto_self_review_mode: Whether to activate review mode. When True, generated QA pairs will be 
                           sent to LLM for review, and only pairs with scores >= 4 will be kept.
        :param checkpoint_path: Optional JSONL file path to persist QA generation progress.
        :param resume_from_checkpoint: Whether to reuse existing checkpoint data when restarting.
        :return: List of QA pairs
        """
        import datamax.generator.qa_generator as qa_generator

        # Initialize debug context
        dbg = DebugContext(enabled=debug, context_name="get_pre_label")
        
        # Log input parameters
        dbg.log_params(
            content_provided=content is not None,
            content_length=len(content) if content else 0,
            use_mllm=use_mllm,
            api_key='***' if api_key else None,
            base_url=base_url,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            max_workers=max_workers,
            language=language,
            use_tree_label=use_tree_label,
            messages_provided=messages is not None,
            interactive_tree=interactive_tree,
            custom_domain_tree_provided=custom_domain_tree is not None,
            file_path=self.file_path,
            use_mineru=self.use_mineru,
            domain=self.domain,
            checkpoint_path_provided=checkpoint_path is not None,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # Prepare content
        with dbg.section("Content Preparation"):
            if content is not None:
                text = content
                dbg.log(f"Using external content, length: {len(text)}")
            else:
                dbg.log("Fetching content via get_data()")
                processed = self.get_data()
                dbg.log_data_structure(processed, "processed_data")

                # Convert to text
                if isinstance(processed, list):
                    parts = [d["content"] if isinstance(d, dict) else d for d in processed]
                    text = "\n\n".join(parts)
                    dbg.log(f"Merged {len(parts)} parts, total length: {len(text)}")
                elif isinstance(processed, dict):
                    text = processed.get("content", "")
                    dbg.log(f"Extracted content from dict, length: {len(text)}")
                else:
                    text = processed
                    dbg.log(f"Using content as-is, length: {len(text)}")
                
                print(text)

        # Add lifecycle marker
        if self.parsed_data is not None and isinstance(self.parsed_data, dict):
            dbg.log("Adding DATA_LABELLING lifecycle entry")
            self.parsed_data.setdefault("lifecycle", []).append(
                self.generate_lifecycle(
                    source_file=self.file_path,
                    domain=self.domain,
                    life_type=LifeType.DATA_LABELLING,
                    usage_purpose="Labeling",
                ).to_dict()
            )

        try:
            # Complete API URL
            base_url = qa_generator.complete_api_url(base_url)
            dbg.log(f"Completed API URL: {base_url}")

            # Generate QA pairs
            with dbg.section("QA Generation"):
                if use_mllm and self.use_mineru:
                    logger.info("Using multimodal QA generator...")
                    
                    # Prepare file paths
                    if isinstance(self.file_path, list):
                        file_names = [os.path.basename(f).replace(".pdf", ".md") for f in self.file_path]
                    elif isinstance(self.file_path, str) and os.path.isfile(self.file_path):
                        file_names = [os.path.basename(self.file_path).replace(".pdf", ".md")]
                    elif isinstance(self.file_path, str) and os.path.isdir(self.file_path):
                        file_names = [
                            os.path.basename(file).replace(".pdf", ".md")
                            for file in list(Path(self.file_path).rglob("*.*"))
                        ]
                    
                    dbg.log(f"Generated {len(file_names)} file names")
                    
                    file_names = [
                        os.path.join(
                            Path(__file__).parent.parent.parent.resolve(),
                            "__temp__",
                            "markdown",
                            f,
                        )
                        for f in file_names
                    ]

                    from datamax.utils import multimodal_qa_generator as generator_module

                    multimodal_file_path = os.path.join(
                        "__temp__",
                        "markdown",
                        os.path.basename(self.file_path).replace(".pdf", ".md"),
                    )
                    dbg.log(f"Multimodal file path: {multimodal_file_path}")

                    data = generator_module.generatr_qa_pairs(
                        file_path=multimodal_file_path,
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        question_number=question_number,
                        max_workers=max_workers,
                    )
                else:
                    logger.info("Using standard QA generator...")
                    dbg.log(f"Text length: {len(text)}")
                    dbg.log_params(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        question_number=question_number,
                        max_workers=max_workers,
                        use_tree_label=use_tree_label
                    )

                    data = qa_generator.full_qa_labeling_process(
                        content=text,
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        question_number=question_number,
                        max_workers=max_workers,
                        use_tree_label=use_tree_label,
                        messages=messages,
                        interactive_tree=interactive_tree,
                        custom_domain_tree=custom_domain_tree,
                        use_mineru=self.use_mineru,
                        debug=debug,
                        structured_data=structured_data,
                        checkpoint_path=checkpoint_path,
                        resume_from_checkpoint=resume_from_checkpoint,
                    )
                
                dbg.log_data_structure(data, "generated_data")

            # Mark success
            if self.parsed_data is not None and isinstance(self.parsed_data, dict):
                dbg.log("Adding DATA_LABELLED lifecycle entry")
                self.parsed_data["lifecycle"].append(
                    self.generate_lifecycle(
                        source_file=self.file_path,
                        domain=self.domain,
                        life_type=LifeType.DATA_LABELLED,
                        usage_purpose="Labeling",
                    ).to_dict()
                )

            # Review mode processing
            if auto_self_review_mode:
                logger.info("🔍 Activating review mode: QA pairs will be reviewed by LLM")
                dbg.log("Starting QA pair review process")
                
                # Extract QA pairs from data structure
                qa_pairs = []
                if isinstance(data, dict) and "qa_pairs" in data:
                    qa_pairs = data["qa_pairs"]
                elif isinstance(data, list):
                    qa_pairs = data
                else:
                    logger.warning("Unexpected data format for review mode, skipping review")
                    return data
                
                # Import review prompt template
                from datamax.generator.prompt_templates import get_system_prompt_for_review
                
                reviewed_qa_pairs = []
                rejected_count = 0
                
                for i, qa_pair in enumerate(qa_pairs):
                    try:
                        # Convert QA pair to JSON string for review
                        qa_pair_json = json.dumps(qa_pair, ensure_ascii=False, indent=2)
                        
                        # Create review messages
                        review_prompt = get_system_prompt_for_review(text, qa_pair_json)
                        review_messages = [
                            {"role": "system", "content": review_prompt},
                            {"role": "user", "content": "请进行评分"}
                        ]
                        
                        # Get review result from LLM using llm_generator
                        from datamax.generator.qa_generator import llm_generator
                        review_result_list = llm_generator(
                            api_key=api_key,
                            model=model_name,
                            base_url=base_url,
                            type="review",
                            message=review_messages,
                            debug=debug
                        )
                        review_result = review_result_list[0] if review_result_list else ""

                        # Parse review result
                        try:
                            review_json = json.loads(review_result)
                            score = review_json.get("score", 0)
                            reason = review_json.get("reason", "No reason provided")
                            
                            if score >= 4:
                                reviewed_qa_pairs.append(qa_pair)
                                dbg.log(f"QA pair {i+1} passed review (score: {score}): {reason}")
                            else:
                                rejected_count += 1
                                dbg.log(f"QA pair {i+1} rejected (score: {score}): {reason}")
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse review result for QA pair {i+1}, rejecting as low quality")
                            rejected_count += 1
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error reviewing QA pair {i+1}: {e}")
                        rejected_count += 1
                        continue
                
                logger.info(f"✅ Review completed: {len(reviewed_qa_pairs)} passed, {rejected_count} rejected")
                dbg.log(f"Review results: {len(reviewed_qa_pairs)} passed, {rejected_count} rejected")
                
                # Update data structure with reviewed QA pairs
                if isinstance(data, dict) and "qa_pairs" in data:
                    data["qa_pairs"] = reviewed_qa_pairs
                else:
                    data = reviewed_qa_pairs
                
                # Preview reviewed QA pairs
                dbg.preview_qa_pairs(data, max_preview=10)
                
                dbg.log("Returning reviewed data")
                return data
            
            # Preview QA pairs
            dbg.preview_qa_pairs(data, max_preview=10)
            
            dbg.log("Returning generated data")
            return data

        except ImportError as e:
            logger.error(f"Cannot import generator module: {e}")
            dbg.log(f"ImportError: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error occurred while generating pre-labeled data: {e}")
            dbg.log(f"Exception: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Mark failure
            if self.parsed_data is not None and isinstance(self.parsed_data, dict):
                dbg.log("Adding DATA_LABEL_FAILED lifecycle entry")
                self.parsed_data["lifecycle"].append(
                    self.generate_lifecycle(
                        source_file=self.file_path,
                        domain=self.domain,
                        life_type=LifeType.DATA_LABEL_FAILED,
                        usage_purpose="Labeling",
                    ).to_dict()
                )
            raise

    def save_label_data(
        self, label_data: list | dict, save_file_name: str = "qa_pairs"
    ):
        """
        Save label data to file.
        :param label_data: Label data to be saved (list or dict).
        :param save_file_name: File name to save the label data.
        """
        if not label_data:
            raise ValueError("No data to save. Please check if label_data is empty.")
        if not save_file_name:
            if isinstance(self.file_path, str):
                save_file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            else:
                save_file_name = "label_data"

        # Handle list type data
        if isinstance(label_data, list):
            with open(save_file_name + ".jsonl", "w", encoding="utf-8") as f:
                for qa_entry in label_data:
                    f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
            logger.info(
                f"✅ [Label Data Saved] Label data saved to {save_file_name}.jsonl"
            )
        # Handle dict type data
        elif isinstance(label_data, dict):
            # Extract QA pairs from dict structure
            qa_pairs = []
            if "qa_pairs" in label_data:
                qa_pairs = label_data["qa_pairs"]
            elif "data" in label_data:
                qa_pairs = label_data["data"]
            else:
                # If dict doesn't contain expected keys, save the entire dict
                qa_pairs = [label_data]

            with open(save_file_name + ".jsonl", "w", encoding="utf-8") as f:
                for qa_entry in qa_pairs:
                    f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
            logger.info(
                f"✅ [Label Data Saved] Label data saved to {save_file_name}.jsonl (extracted {len(qa_pairs)} QA pairs from dict)"
            )
        else:
            raise ValueError(
                f"Unsupported data type: {type(label_data)}. Expected list or dict."
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
        parsed_data: str | dict = None,
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
                use_qwen_vl_ocr=self.use_qwen_vl_ocr,
                use_mllm=self.use_mllm,
                mllm_system_prompt=self.mllm_system_prompt,
                file_path=file_path,
                to_markdown=self.to_markdown,
                domain=self.domain,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
            )
            if parser:
                return parser.parse(file_path=file_path)
        except Exception as e:
            raise e


if __name__ == "__main__":
    pass
