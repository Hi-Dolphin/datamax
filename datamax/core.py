import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal, TypedDict, cast, overload

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

import datamax.generator.qa_generator as qa_generator
from datamax.cleaner import data_cleaner
from datamax.generator import PerformanceMonitor
from datamax.parser.base import BaseLife
from datamax.utils.debug_logger import DebugContext
from datamax.utils.lifecycle_types import LifeType


class LifecycleMetadata(TypedDict):
    storage_size: int
    source_file: str
    domain: str
    usage_purpose: str


class LifecycleRecord(TypedDict):
    update_time: str
    life_type: list[str]
    life_metadata: LifecycleMetadata


ParsedFieldValue = str | list[LifecycleRecord]


class ParsedDocument(TypedDict, total=False):
    extension: str
    content: str
    lifecycle: list[LifecycleRecord]


class ParsedDataList(list[ParsedDocument]):
    """List wrapper that exposes dict-like get access for tooling."""

    def get(
        self,
        key: str,
        default: ParsedFieldValue | None = None,
    ) -> ParsedFieldValue | None:
        if not self:
            return default

        first = self[0]
        if key not in first:
            return default

        value = first[key]
        return cast(ParsedFieldValue, value)


ParsedData = ParsedDocument
ParsedDataResult = ParsedDocument | ParsedDataList


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
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
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
            ".json": "JsonParser",
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
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
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
        self.performance_monitor: PerformanceMonitor | None = None
        self.llm_call_records: list[dict] = []
        self.llm_call_qa_pairs: list[dict] = []
        self.last_performance_report: dict = {}

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

    def get_data(self) -> ParsedDataResult:
        """
        Parse the file or directory specified in the file path and return the data.

        :return: A list of parsed data if the file path is a directory, otherwise a single parsed data.
        """
        try:
            if isinstance(self.file_path, list):
                parsed_items: list[ParsedData] = []
                for f in self.file_path:
                    file_name = os.path.basename(f)
                    if (
                        file_name in self._cache
                        and self._cache[file_name]["ttl"] > time.time()
                    ):
                        logger.info(f"✅ [Cache Hit] Using cached data for {file_name}")
                        parsed_items.append(self._cache[file_name]["data"])
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
                        if res_data is None:
                            logger.warning(
                                f"[Parse Warning] Parser returned no data for {file_name}, skipping."
                            )
                            continue
                        parsed_value = cast(ParsedData, res_data)
                        parsed_items.append(parsed_value)
                        self.set_data(file_name, parsed_value)
                parsed_list = ParsedDataList(parsed_items)
                self.parsed_data = parsed_list
                return parsed_list

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
                    if parsed_data is None:
                        raise ValueError(
                            f"Parser returned no data for {self.file_path}."
                        )
                    if isinstance(parsed_data, list):
                        parsed_list = ParsedDataList(parsed_data)
                        self.parsed_data = parsed_list
                        self.set_data(file_name, parsed_list)
                        return parsed_list
                    parsed_item = cast(ParsedData, parsed_data)
                    self.parsed_data = parsed_item
                    self.set_data(file_name, parsed_item)
                    return parsed_item

            elif isinstance(self.file_path, str) and os.path.isdir(self.file_path):
                file_list = [
                    str(file) for file in list(Path(self.file_path).rglob("*.*"))
                ]
                parsed_items: list[ParsedData] = []
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
                            parsed_items.append(self._cache[file_name]["data"])
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
                            if res_data is None:
                                logger.warning(
                                    f"[Parse Warning] Parser returned no data for {file_name}, skipping."
                                )
                                continue
                            parsed_value = cast(ParsedData, res_data)
                            parsed_items.append(parsed_value)
                            self.set_data(file_name, parsed_value)
                parsed_list = ParsedDataList(parsed_items)
                self.parsed_data = parsed_list
                return parsed_list
            else:
                raise ValueError("Invalid file path.")

        except Exception as e:
            raise e

    def clean_data(self, method_list: list[str], text: str | None = None):
        """
        Clean data

        methods include AbnormalCleaner, TextFilter, PrivacyDesensitization which are 1, 2, 3

        :return: Cleaned data
        """

        def _require_str(value: Any, label: str) -> str:
            if isinstance(value, str):
                return value
            if value is None:
                raise ValueError(f"{label} produced no text.")
            raise TypeError(
                f"{label} expected string output, got {type(value).__name__}"
            )

        # 1) Prepare original content
        if text is not None:
            cleaned_text: str = text
        elif isinstance(self.parsed_data, ParsedDataList):
            content = self.parsed_data.get("content")
            cleaned_text = _require_str(content, "Parsed dataset content")
        elif isinstance(self.parsed_data, dict):
            content = cast(ParsedDocument, self.parsed_data).get("content")
            cleaned_text = _require_str(content, "Parsed document content")
        elif isinstance(self.parsed_data, str):
            cleaned_text = self.parsed_data
        elif self.parsed_data is None:
            raise ValueError("No data to clean.")
        else:
            raise TypeError(f"Unsupported parsed data type: {type(self.parsed_data)!r}")
        # 2) Trigger "cleaning start"
        if isinstance(self.file_path, list):
            source_file = ", ".join(map(str, self.file_path))
        else:
            source_file = str(self.file_path)
        lc_start = cast(
            LifecycleRecord,
            self.generate_lifecycle(
                source_file=source_file,
                domain=self.domain,
                life_type=LifeType.DATA_CLEANING,
                usage_purpose="Data Cleaning",
            ).to_dict(),
        )

        try:
            # 3) Execute cleaning steps
            for method in method_list:
                if method == "abnormal":
                    result = data_cleaner.AbnormalCleaner(cleaned_text).to_clean()
                    cleaned_text = _require_str(
                        result.get("text"),
                        "AbnormalCleaner",
                    )
                elif method == "filter":
                    result = data_cleaner.TextFilter(cleaned_text).to_filter()
                    cleaned_text = _require_str(
                        result.get("text", ""),
                        "TextFilter",
                    )
                elif method == "private":
                    result = data_cleaner.PrivacyDesensitization(
                        cleaned_text
                    ).to_private()
                    cleaned_text = _require_str(
                        result.get("text"),
                        "PrivacyDesensitization",
                    )

            # 4) Cleaning successful, trigger "cleaning completed"
            lc_end = cast(
                LifecycleRecord,
                self.generate_lifecycle(
                    source_file=source_file,
                    domain=self.domain,
                    life_type=LifeType.DATA_CLEANED,
                    usage_purpose="Data Cleaning",
                ).to_dict(),
            )

        except Exception:
            # 5) Cleaning failed, trigger "cleaning failed"
            lc_fail = cast(
                LifecycleRecord,
                self.generate_lifecycle(
                    source_file=source_file,
                    domain=self.domain,
                    life_type=LifeType.DATA_CLEAN_FAILED,
                    usage_purpose="Data Cleaning",
                ).to_dict(),
            )
            # Add failure event to parsed_data before raising
            if self.parsed_data and isinstance(self.parsed_data, dict):
                lifecycle_list = cast(
                    list[LifecycleRecord], self.parsed_data.setdefault("lifecycle", [])
                )
                lifecycle_list.append(lc_start)
                lifecycle_list.append(lc_fail)
            raise

        # 6) Update content and merge lifecycles
        if lc_end is None:
            raise RuntimeError("Lifecycle end event was not generated.")
        if self.parsed_data and isinstance(self.parsed_data, dict):
            origin = cast(ParsedDocument, self.parsed_data)
            origin["content"] = cleaned_text
            lifecycle_list = cast(
                list[LifecycleRecord], origin.setdefault("lifecycle", [])
            )
            lifecycle_list.extend([lc_start, lc_end])
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
        content: str | None = None,
        use_mllm: bool = False,
        api_key: str,
        base_url: str,
        model_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        question_number: int = 5,
        max_qps: float = 5.0,
        language: str = "zh",
        use_tree_label: bool = False,
        messages: list | None = None,
        interactive_tree: bool = False,
        custom_domain_tree: list[dict[str, Any]] | None = None,
        debug: bool = False,
        structured_data: bool = False,
        auto_self_review_mode: bool = False,
        review_max_qps: float = 5.0,
        review_max_retries: int = 10,
        review_score_threshold: int = 4,
        review_user_prompt: str = "请进行评分",
        review_progress_desc: str = "Reviewing QA pairs",
        checkpoint_path: str | None = None,
        resume_from_checkpoint: bool = True,
        agent_mode: dict[str, Any] | str | bool | None = None,
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
        :param max_qps: Maximum requests per second budget
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
        :param review_max_qps: Maximum QPS for review requests.
        :param review_max_retries: Maximum retry attempts per review request.
        :param review_score_threshold: Minimum score required to keep a QA pair.
        :param review_user_prompt: User message sent to the reviewer LLM.
        :param review_progress_desc: Progress bar description for review mode.
        :param checkpoint_path: Optional JSONL file path to persist QA generation progress.
        :param resume_from_checkpoint: Whether to reuse existing checkpoint data when restarting.
        :param agent_mode: Optional agent configuration. When enabled, agent-style training data will be generated
                           instead of standard QA pairs. Accepts a dict with keys like `enabled`, `agent_backend`,
                           or a shorthand string backend ("langgraph", "openai").
        :return: List of QA pairs
        """
        # Initialize debug context
        dbg = DebugContext(enabled=debug, context_name="get_pre_label")
        perf_monitor = PerformanceMonitor()
        self.performance_monitor = perf_monitor

        def _round_metric(value, digits: int = 3):
            return round(value, digits) if isinstance(value, (int, float)) else value

        def _log_performance_report(report: dict):
            if not dbg.enabled:
                return
            for stage_name, metrics in report.get("stages", {}).items():
                dbg.log(
                    f"Performance[{stage_name}]",
                    items=metrics.get("items"),
                    qpm=_round_metric(metrics.get("effective_qpm")),
                    tokens_per_second=_round_metric(metrics.get("tokens_per_second")),
                    request_count=metrics.get("request_count"),
                )
            totals = report.get("totals", {})
            dbg.log(
                "PerformanceTotals",
                total_tokens=totals.get("total_tokens"),
                overall_qpm=_round_metric(totals.get("overall_qpm")),
                tokens_per_second=_round_metric(totals.get("tokens_per_second")),
            )

        def _format_performance_table(report: dict) -> str:
            stages = report.get("stages", {})
            if not stages:
                return "┌──────────────────────┐\n│ No performance data │\n└──────────────────────┘"

            headers = [
                "Stage",
                "Runs",
                "Items",
                "Duration(s)",
                "Reqs",
                "QPM",
                "Total Tokens",
                "Tokens/s",
                "Tokens/req",
            ]
            rows = []

            for stage_name, metrics in stages.items():
                label = stage_name.replace("_", " ").title()
                rows.append(
                    [
                        label,
                        str(metrics.get("runs", 0)),
                        str(metrics.get("items", 0)),
                        f"{metrics.get('stage_duration_seconds', 0.0):.2f}",
                        str(metrics.get("request_count", 0)),
                        f"{metrics.get('effective_qpm', 0.0):.2f}",
                        str(metrics.get("total_tokens", 0)),
                        f"{metrics.get('tokens_per_second', 0.0):.2f}",
                        f"{metrics.get('tokens_per_request', 0.0):.2f}",
                    ]
                )

            widths = [len(header) for header in headers]
            for row in rows:
                for idx, cell in enumerate(row):
                    widths[idx] = max(widths[idx], len(cell))

            def _make_border(left: str, fill: str, junction: str, right: str) -> str:
                segments = [fill * (w + 2) for w in widths]
                return left + junction.join(segments) + right

            top = _make_border("┌", "─", "┬", "┐")
            mid = _make_border("├", "─", "┼", "┤")
            bottom = _make_border("└", "─", "┴", "┘")

            header_cells = [
                f" {headers[idx].ljust(widths[idx])} " for idx in range(len(headers))
            ]
            header_line = "│" + "│".join(header_cells) + "│"

            row_lines = []
            for row in rows:
                cells = [
                    f" {row[idx].ljust(widths[idx])} " for idx in range(len(headers))
                ]
                row_lines.append("│" + "│".join(cells) + "│")

            totals = report.get("totals", {})
            totals_line = (
                "Totals: tokens="
                f"{totals.get('total_tokens', 0)} | prompt="
                f"{totals.get('prompt_tokens', 0)} | completion="
                f"{totals.get('completion_tokens', 0)} | "
                f"overall_qpm={totals.get('overall_qpm', 0.0):.2f} | "
                f"tokens/s={totals.get('tokens_per_second', 0.0):.2f} | "
                f"duration={totals.get('workflow_duration_seconds', 0.0):.2f}s"
            )

            table_lines = [top, header_line, mid, *row_lines, bottom, totals_line]
            return "\n".join(table_lines)

        def _record_performance(target):
            if (
                isinstance(target, dict)
                and "performance" in target
                and target["performance"]
            ):
                report = target["performance"]
                if isinstance(report, dict):
                    self.last_performance_report = report
                    call_records = target.get("llm_call_records", [])
                    call_qa_pairs = target.get("llm_call_qa_pairs", [])
                    self.llm_call_records = call_records
                    self.llm_call_qa_pairs = call_qa_pairs
                    _log_performance_report(report)
                    logger.info(
                        "\n📊 Performance Summary\n{}",
                        _format_performance_table(report),
                    )
                    return report
            report = perf_monitor.build_report()
            self.last_performance_report = report
            call_records = perf_monitor.get_call_records()
            call_qa_pairs = perf_monitor.call_records_as_qa_pairs()
            self.llm_call_records = call_records
            self.llm_call_qa_pairs = call_qa_pairs
            if isinstance(target, dict):
                target["performance"] = report
                target["llm_call_records"] = call_records
                target["llm_call_qa_pairs"] = call_qa_pairs
            _log_performance_report(report)
            logger.info(
                "\n📊 Performance Summary\n{}", _format_performance_table(report)
            )
            return report

        agent_mode_config: dict[str, Any] = {}
        agent_mode_enabled = False
        agent_backend = "openai"
        if isinstance(agent_mode, dict):
            agent_mode_config = dict(agent_mode)
            enabled_flag = agent_mode_config.get("enabled")
            agent_mode_enabled = (
                bool(enabled_flag)
                if enabled_flag is not None
                else bool(agent_mode_config)
            )
            mode_value = agent_mode_config.get("mode") or agent_mode_config.get("type")
            if isinstance(mode_value, str) and mode_value.lower() == "agent":
                agent_mode_enabled = True
            backend_value = (
                agent_mode_config.get("agent_backend")
                or agent_mode_config.get("backend")
                or agent_mode_config.get("strategy")
            )
            if isinstance(backend_value, str) and backend_value.strip():
                agent_backend = backend_value.strip().lower()
        elif isinstance(agent_mode, str):
            normalized_mode = agent_mode.strip().lower()
            if normalized_mode and normalized_mode not in {"qa", "none", "disabled"}:
                agent_mode_enabled = True
                agent_backend = (
                    normalized_mode
                    if normalized_mode in {"langgraph", "openai"}
                    else "langgraph"
                )
                agent_mode_config = {"agent_backend": agent_backend}
        elif agent_mode is True:
            agent_mode_enabled = True
        else:
            agent_mode_config = {}

        if agent_backend not in {"langgraph", "openai"}:
            dbg.log(
                f"Unrecognized agent backend '{agent_backend}', defaulting to 'langgraph'"
            )
            agent_backend = "langgraph"
        if agent_mode_enabled:
            agent_mode_config.setdefault("agent_backend", agent_backend)

        # Log input parameters
        dbg.log_params(
            content_provided=content is not None,
            content_length=len(content) if content else 0,
            use_mllm=use_mllm,
            api_key="***" if api_key else None,
            base_url=base_url,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            max_qps=max_qps,
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
            agent_mode_enabled=agent_mode_enabled,
            agent_backend=agent_backend if agent_mode_enabled else None,
        )

        # Prepare content
        with dbg.section("Content Preparation"):
            if content is not None:
                prepared_text = content
                dbg.log(f"Using external content, length: {len(prepared_text)}")
            else:
                dbg.log("Fetching content via get_data()")
                processed = self.get_data()
                dbg.log_data_structure(processed, "processed_data")

                # Convert to text
                if isinstance(processed, list):
                    parts = [
                        d["content"] if isinstance(d, dict) else d for d in processed
                    ]
                    prepared_text = "\n\n".join(parts)
                    dbg.log(
                        f"Merged {len(parts)} parts, total length: {len(prepared_text)}"
                    )
                elif isinstance(processed, dict):
                    prepared_text = processed.get("content", "")
                    dbg.log(
                        f"Extracted content from dict, length: {len(prepared_text)}"
                    )
                else:
                    prepared_text = processed
                    dbg.log(f"Using content as-is, length: {len(prepared_text)}")

                print(prepared_text)

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
                if agent_mode_enabled:
                    logger.info("Using agent training data generator...")
                    dbg.log("Agent mode enabled; switching to agent training pipeline")
                    from datamax.generator.agent_qa_generator import (
                        AgentGenerationConfig,
                        generate_agent_training_data,
                    )

                    spec_sources: list[Any] = []
                    configured_sources = (
                        agent_mode_config.get("spec_sources")
                        if isinstance(agent_mode_config, dict)
                        else None
                    )
                    if isinstance(configured_sources, list):
                        spec_sources.extend(configured_sources)
                    elif configured_sources:
                        spec_sources.append(configured_sources)

                    if isinstance(self.file_path, list):
                        spec_sources.extend(self.file_path)
                    elif isinstance(self.file_path, str) and self.file_path:
                        spec_sources.append(self.file_path)

                    spec_sources = [src for src in spec_sources if src]

                    if content is not None and not spec_sources:
                        parsed_spec: Any | None = None
                        try:
                            parsed_spec = json.loads(content)
                        except json.JSONDecodeError:
                            try:
                                import yaml  # type: ignore

                                parsed_spec = yaml.safe_load(content)
                            except Exception:
                                parsed_spec = None
                        if parsed_spec:
                            spec_sources.append(parsed_spec)

                    if not spec_sources:
                        raise ValueError(
                            "Agent mode requires at least one API specification file or content."
                        )

                    min_interval = agent_mode_config.get("min_request_interval_seconds")
                    if min_interval is None and max_qps > 0:
                        min_interval = 1.0 / max_qps

                    agent_question_count = agent_mode_config.get(
                        "question_count", question_number
                    )
                    agent_checkpoint_path = agent_mode_config.get(
                        "checkpoint_path", checkpoint_path
                    )
                    agent_resume_from_checkpoint = agent_mode_config.get(
                        "resume_from_checkpoint", resume_from_checkpoint
                    )

                    default_max_retries = AgentGenerationConfig.__dataclass_fields__[
                        "max_retries"
                    ].default
                    max_retries_value = agent_mode_config.get("max_retries")
                    if max_retries_value is None:
                        max_retries_value = default_max_retries
                    default_min_interval = AgentGenerationConfig.__dataclass_fields__[
                        "min_request_interval_seconds"
                    ].default
                    min_interval_value = (
                        float(min_interval)
                        if isinstance(min_interval, (int, float))
                        else default_min_interval
                    )

                    agent_config = AgentGenerationConfig(
                        api_key=api_key,
                        base_url=base_url,
                        agent_question_generate_model=agent_mode_config.get(
                            "agent_question_generate_model", model_name
                        ),
                        classify_model=agent_mode_config.get(
                            "classify_model", model_name
                        ),
                        core_agent_answer_generate_model=agent_mode_config.get(
                            "core_agent_answer_generate_model", model_name
                        ),
                        review_model=agent_mode_config.get("review_model", model_name),
                        question_count=agent_question_count,
                        max_questions_per_context=agent_mode_config.get(
                            "max_questions_per_context", 4
                        ),
                        top_k_tools=agent_mode_config.get("top_k_tools", 5),
                        max_turns=agent_mode_config.get("max_turns", 8),
                        langgraph_retry=agent_mode_config.get("langgraph_retry", 1),
                        checkpoint_path=agent_checkpoint_path,
                        resume_from_checkpoint=agent_resume_from_checkpoint,
                        max_retries=int(max_retries_value),
                        min_request_interval_seconds=min_interval_value,
                        question_temperature=agent_mode_config.get(
                            "question_temperature", 0.5
                        ),
                        classify_temperature=agent_mode_config.get(
                            "classify_temperature", 0.3
                        ),
                        agent_temperature=agent_mode_config.get(
                            "agent_temperature", 0.7
                        ),
                        review_temperature=agent_mode_config.get(
                            "review_temperature", 0.2
                        ),
                        max_workers=agent_mode_config.get("max_workers", 4),
                        debug=debug,
                        agent_backend=agent_backend,
                        auth=agent_mode_config.get("auth"),
                        default_tool_server=agent_mode_config.get(
                            "default_tool_server"
                        ),
                        tool_request_timeout=float(
                            agent_mode_config.get("tool_request_timeout", 30.0)
                        ),
                        require_auth_for_protected_tools=agent_mode_config.get(
                            "require_auth_for_protected_tools", True
                        ),
                    )

                    if isinstance(min_interval, (int, float)):
                        agent_config.min_request_interval_seconds = float(min_interval)
                    if agent_mode_config.get("question_temperature") is not None:
                        agent_config.question_temperature = float(
                            agent_mode_config["question_temperature"]
                        )
                    if agent_mode_config.get("classify_temperature") is not None:
                        agent_config.classify_temperature = float(
                            agent_mode_config["classify_temperature"]
                        )
                    if agent_mode_config.get("agent_temperature") is not None:
                        agent_config.agent_temperature = float(
                            agent_mode_config["agent_temperature"]
                        )
                    if agent_mode_config.get("review_temperature") is not None:
                        agent_config.review_temperature = float(
                            agent_mode_config["review_temperature"]
                        )
                    if agent_mode_config.get("top_k_tools") is not None:
                        agent_config.top_k_tools = int(agent_mode_config["top_k_tools"])
                    if agent_mode_config.get("max_turns") is not None:
                        agent_config.max_turns = int(agent_mode_config["max_turns"])
                    if agent_mode_config.get("max_workers") is not None:
                        agent_config.max_workers = int(agent_mode_config["max_workers"])
                    if agent_mode_config.get("max_questions_per_context") is not None:
                        agent_config.max_questions_per_context = int(
                            agent_mode_config["max_questions_per_context"]
                        )
                    if agent_mode_config.get("langgraph_retry") is not None:
                        agent_config.langgraph_retry = int(
                            agent_mode_config["langgraph_retry"]
                        )
                    if agent_mode_config.get("max_retries") is not None:
                        agent_config.max_retries = int(agent_mode_config["max_retries"])
                    if agent_mode_config.get("tool_request_timeout") is not None:
                        agent_config.tool_request_timeout = float(
                            agent_mode_config["tool_request_timeout"]
                        )
                    if (
                        agent_mode_config.get("require_auth_for_protected_tools")
                        is not None
                    ):
                        agent_config.require_auth_for_protected_tools = bool(
                            agent_mode_config["require_auth_for_protected_tools"]
                        )
                    if agent_mode_config.get("default_tool_server") is not None:
                        agent_config.default_tool_server = str(
                            agent_mode_config["default_tool_server"]
                        )
                    if agent_mode_config.get("auth") is not None:
                        agent_config.auth = agent_mode_config["auth"]

                    dbg.log_params(
                        agent_backend=agent_backend,
                        spec_source_count=len(spec_sources),
                        agent_question_count=agent_config.question_count,
                        agent_top_k_tools=agent_config.top_k_tools,
                        agent_max_turns=agent_config.max_turns,
                    )

                    data = generate_agent_training_data(spec_sources, agent_config)
                    auto_self_review_mode = False
                    if isinstance(data, dict):
                        metadata = data.setdefault("metadata", {})
                        metadata.setdefault("agent_mode", {})
                        metadata["agent_mode"].update(
                            {
                                "enabled": True,
                                "agent_backend": agent_backend,
                            }
                        )
                        if isinstance(agent_mode_config, dict):
                            metadata["agent_mode"]["config"] = agent_mode_config
                elif use_mllm and self.use_mineru:
                    logger.info("Using multimodal QA generator...")

                    # Prepare file paths
                    if isinstance(self.file_path, list):
                        file_names = [
                            os.path.basename(f).replace(".pdf", ".md")
                            for f in self.file_path
                        ]
                    elif isinstance(self.file_path, str) and os.path.isfile(
                        self.file_path
                    ):
                        file_names = [
                            os.path.basename(self.file_path).replace(".pdf", ".md")
                        ]
                    elif isinstance(self.file_path, str) and os.path.isdir(
                        self.file_path
                    ):
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

                    from datamax.utils import (
                        multimodal_qa_generator as generator_module,
                    )

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
                        max_qps=max_qps,
                    )
                else:
                    logger.info("Using standard QA generator...")
                    dbg.log(f"Text length: {len(prepared_text)}")
                    dbg.log_params(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        question_number=question_number,
                        max_qps=max_qps,
                        use_tree_label=use_tree_label,
                    )

                    data = qa_generator.full_qa_labeling_process(
                        content=prepared_text,
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        question_number=question_number,
                        max_qps=max_qps,
                        use_tree_label=use_tree_label,
                        messages=messages,
                        interactive_tree=interactive_tree,
                        custom_domain_tree=custom_domain_tree,
                        use_mineru=self.use_mineru,
                        debug=debug,
                        structured_data=structured_data,
                        checkpoint_path=checkpoint_path,
                        resume_from_checkpoint=resume_from_checkpoint,
                        perf_monitor=perf_monitor,
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
                logger.info(
                    "🔍 Activating review mode: QA pairs will be reviewed by LLM"
                )
                dbg.log("Starting QA pair review process")

                # Extract QA pairs from data structure
                qa_pairs = []
                if isinstance(data, dict) and "qa_pairs" in data:
                    qa_pairs = data["qa_pairs"]
                elif isinstance(data, list):
                    qa_pairs = data
                else:
                    logger.warning(
                        "Unexpected data format for review mode, skipping review"
                    )
                    _record_performance(data)
                    return data

                from datamax.generator.qa_generator import review_qa_pairs

                reviewed_qa_pairs, rejected_count = review_qa_pairs(
                    qa_pairs=qa_pairs,
                    source_text=prepared_text,
                    api_key=api_key,
                    model=model_name,
                    base_url=base_url,
                    score_threshold=review_score_threshold,
                    max_qps=review_max_qps,
                    max_retries=review_max_retries,
                    debug=debug,
                    user_prompt=review_user_prompt,
                    progress_desc=review_progress_desc,
                    dbg=dbg,
                    perf_monitor=perf_monitor,
                )

                logger.info(
                    f"✅ Review completed: {len(reviewed_qa_pairs)} passed, {rejected_count} rejected"
                )
                dbg.log(
                    f"Review results: {len(reviewed_qa_pairs)} passed, {rejected_count} rejected"
                )

                # Update data structure with reviewed QA pairs
                if isinstance(data, dict) and "qa_pairs" in data:
                    data["qa_pairs"] = reviewed_qa_pairs
                else:
                    data = reviewed_qa_pairs

                _record_performance(data)
                # Preview reviewed QA pairs
                dbg.preview_qa_pairs(data, max_preview=10)

                dbg.log("Returning reviewed data")
                return data

            # Preview QA pairs
            _record_performance(data)
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
            qa_pairs: list[dict] = []
            core_pairs = label_data.get("qa_pairs")
            if isinstance(core_pairs, list):
                qa_pairs.extend(core_pairs)
            elif "data" in label_data and isinstance(label_data["data"], list):
                qa_pairs.extend(label_data["data"])
            llm_call_pairs = label_data.get("llm_call_qa_pairs")
            llm_call_pairs_list = (
                llm_call_pairs if isinstance(llm_call_pairs, list) else []
            )
            qa_pairs.extend(llm_call_pairs_list)
            if not qa_pairs:
                # If dict doesn't contain expected keys, save the entire dict
                qa_pairs = [label_data]

            with open(save_file_name + ".jsonl", "w", encoding="utf-8") as f:
                for qa_entry in qa_pairs:
                    f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
            base_count = len(core_pairs) if isinstance(core_pairs, list) else 0
            llm_call_count = len(llm_call_pairs_list)
            logger.info(
                f"✅ [Label Data Saved] Label data saved to {save_file_name}.jsonl "
                f"(extracted {len(qa_pairs)} QA pairs from dict; "
                f"{base_count} final pairs + {llm_call_count} call transcripts)"
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
