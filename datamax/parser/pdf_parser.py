import os
import re
import fitz  # PyMuPDF
import subprocess
import tempfile
import base64
from PIL import Image
from typing import List, Union
from contextlib import suppress
from loguru import logger
from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader

from datamax.parser.base import MarkdownOutputVo, BaseLife
from datamax.utils.lifecycle_types import LifeType


OCR_MODEL_SET = [
    "qwen-vl-ocr",
    "qwen-vl-ocr-latest",
    "qwen-vl-max-latest",
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
]

class PdfLLMOcrProcessor(BaseLife):
    """PDF to Markdown"""

    def __init__(self, api_key: str, base_url: str, model_name: str, domain: str = "Technology"):
        
        if model_name not in OCR_MODEL_SET:
            raise ValueError("ocr_model_name is wrong, only support: " + ", ".join(OCR_MODEL_SET))
        super().__init__(domain=domain)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name if model_name in OCR_MODEL_SET else model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.domain = domain

    def _pdf_to_images(self, file_path: str, dpi: int = 300) -> List[str]:
        logger.info(f"PDF to image: {file_path}")
        temp_image_paths = []
        doc = fitz.open(file_path)
        try:
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=dpi)
                with Image.frombytes("RGB", (pix.width, pix.height), pix.samples) as img:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        img.save(temp_file.name, "JPEG", quality=95)
                        temp_image_paths.append(temp_file.name)
            logger.info(f"PDF to image，total {len(temp_image_paths)} pages: {file_path}")
            return temp_image_paths
        finally:
            doc.close()

    @staticmethod
    def encode_image(image_path):
        return base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

    def _ocr_page_to_markdown(self, image_path: str) -> MarkdownOutputVo:
        logger.info(f"OCR Processing: {image_path}")
        base64_image = self.encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        messages = [
            {
                "role": "system",
                "content": "你是一个Markdown转换专家，请将文档内容转换为标准Markdown格式：\n"
                           "- 表格使用Markdown语法\n"
                           "- 数学公式用$$包裹\n"
                           "- 保留原始段落结构"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 8192
                    },
                    {"type": "text", "text": "请以Markdown格式输出本页所有内容"}
                ]
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048
            )
            raw_text = response.choices[0].message.content or ""
            logger.info(f"OCR Process Done: {image_path}")
            return MarkdownOutputVo(
                extension="md",
                content=self._format_markdown(raw_text)
            )
        except Exception as e:
            logger.error(f"OCR Process Fail: {image_path}, Fail: {e}")
            raise

    def _format_markdown(self, text: str) -> str:
        text = re.sub(r'\|(\s*\-+\s*)\|', r'|:---:|', text)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def parse(self, file_path: Union[str, List[str]]) -> Union[MarkdownOutputVo, List[MarkdownOutputVo]]:
        """
        Single-file or multi-file PDF to Markdown can be converted
        Returns:
            MarkdownOutputVo or MarkdownOutputVo list
        """
        if isinstance(file_path, str):
            logger.info(f"Start Process PDF: {file_path}")
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="PDF to Markdown"
            )
            logger.debug(f"⚙️ LifeCycle: DATA_PROCESSING - {lc_start}")
            combined_md = MarkdownOutputVo(extension="md", content="")
            combined_md.add_lifecycle(lc_start)
            image_paths = self._pdf_to_images(file_path)
            try:
                for i, img_path in enumerate(image_paths):
                    logger.info(f"Processing page {i+1}/{len(image_paths)}: {file_path}")
                    page_md = self._ocr_page_to_markdown(img_path)
                    combined_md.content += f"## Page {i+1}\n\n{page_md.content}\n\n"
                    lc_page = self.generate_lifecycle(
                        source_file=img_path,
                        domain="document_ocr",
                        life_type=LifeType.DATA_PROCESSING,
                        usage_purpose="PDF to Markdown"
                    )
                    logger.debug(f"⚙️ LifeCycle: DATA_PROCESSING - {lc_page}")
                    combined_md.add_lifecycle(lc_page)
                    with suppress(PermissionError):
                        os.unlink(img_path)
                lc_end = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type=LifeType.DATA_PROCESSED,
                    usage_purpose="PDF to Markdown"
                )
                logger.debug(f"⚙️ LifeCycle: DATA_PROCESSED - {lc_end}")
                combined_md.add_lifecycle(lc_end)
                logger.info(f"Processing completed: {file_path}")
                return combined_md
            except Exception as e:
                for p in image_paths:
                    with suppress(PermissionError):
                        os.unlink(p)
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type=LifeType.DATA_PROCESS_FAILED,
                    usage_purpose="PDF to Markdown"
                )
                logger.error(f"Processing failed: {file_path}, Error: {e}, Lifecycle: {lc_fail}")
                combined_md.add_lifecycle(lc_fail)
                combined_md.content += f"\nProcessing failed: {e}"
                return combined_md
        elif isinstance(file_path, list):
            results = []
            for f in file_path:
                try:
                    results.append(self.parse(f))
                except Exception as e:
                    lc_fail = self.generate_lifecycle(
                        source_file=f,
                        domain=self.domain,
                        life_type=LifeType.DATA_PROCESS_FAILED,
                        usage_purpose="PDF to Markdown"
                    )
                    logger.error(f"Batch processing failed: {f}, Error: {e}, Lifecycle: {lc_fail}")
                    vo = MarkdownOutputVo(extension="md", content=f"Processing failed: {e}")
                    vo.add_lifecycle(lc_fail)
                    results.append(vo)
            return results
        else:
            raise ValueError("file_path must be str or list[str]")


class PdfParser(BaseLife):
    def __init__(
        self,
        file_path: str | list,
        use_mineru: bool = False,
        use_qwen_vl_ocr: bool = False,
        domain: str = "Technology",
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
    ):
        super().__init__(domain=domain)
        self.file_path = file_path
        self.use_mineru = use_mineru
        self.use_qwen_vl_ocr = use_qwen_vl_ocr
        self.domain = domain
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name if model_name in OCR_MODEL_SET else model_name

        # Validate OCR parameters
        if self.use_qwen_vl_ocr:
            if not all([self.api_key, self.base_url, self.model_name]):
                raise ValueError("Qwen-VL OCR requires api_key, base_url, and model_name to be provided")


    @staticmethod
    def encode_image(image_path):
        return base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

    def _ocr_page_to_markdown(self, image_path: str) -> MarkdownOutputVo:
        logger.info(f"OCR Process image: {image_path}")
        base64_image = self.encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        messages = [
            {
                "role": "system",
                "content": "你是一个Markdown转换专家，请将文档内容转换为标准Markdown格式：\n"
                           "- 表格使用Markdown语法\n"
                           "- 数学公式用$$包裹\n"
                           "- 保留原始段落结构"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 8192
                    },
                    {"type": "text", "text": "请以Markdown格式输出本页所有内容"}
                ]
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048
            )
            raw_text = response.choices[0].message.content or ""
            logger.info(f"OCR Process Done: {image_path}")
            return MarkdownOutputVo(
                extension="md",
                content=self._format_markdown(raw_text)
            )
        except Exception as e:
            logger.error(f"OCR Process Fail: {image_path}, Fail: {e}")
            raise


    def parse(self, file_path: Union[str, List[str]]) -> Union[MarkdownOutputVo, List[MarkdownOutputVo]]:
        """
        Single-file or multi-file PDF to Markdown can be converted
        Returns:
            MarkdownOutputVo or MarkdownOutputVo list
        """
        if isinstance(file_path, str):
            logger.info(f"Start Process PDF: {file_path}")
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="PDF to Markdown"
            )
            logger.debug(f"⚙️ LifeCycle: DATA_PROCESSING - {lc_start}")
            combined_md = MarkdownOutputVo(extension="md", content="")
            combined_md.add_lifecycle(lc_start)
            image_paths = self._pdf_to_images(file_path)
            try:
                for i, img_path in enumerate(image_paths):
                    logger.info(f"Processing page {i+1}/{len(image_paths)}: {file_path}")
                    page_md = self._ocr_page_to_markdown(img_path)
                    combined_md.content += f"## Page {i+1}\n\n{page_md.content}\n\n"
                    lc_page = self.generate_lifecycle(
                        source_file=img_path,
                        domain="document_ocr",
                        life_type=LifeType.DATA_PROCESSING,
                        usage_purpose="PDF to Markdown"
                    )
                    logger.debug(f"⚙️ LifeCycle: DATA_PROCESSING - {lc_page}")
                    combined_md.add_lifecycle(lc_page)
                    with suppress(PermissionError):
                        os.unlink(img_path)
                lc_end = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type=LifeType.DATA_PROCESSED,
                    usage_purpose="PDF to Markdown"
                )
                logger.debug(f"⚙️ LifeCycle: DATA_PROCESSED - {lc_end}")
                combined_md.add_lifecycle(lc_end)
                logger.info(f"Process Done: {file_path}")
                return combined_md
            except Exception as e:
                for p in image_paths:
                    with suppress(PermissionError):
                        os.unlink(p)
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type=LifeType.DATA_PROCESS_FAILED,
                    usage_purpose="PDF to Markdown"
                )
                logger.error(f"Process Fail: {file_path}, Fail: {e}, LifeCycle: {lc_fail}")
                combined_md.add_lifecycle(lc_fail)
                combined_md.content += f"\nProcess Fail: {e}"
                return combined_md
        elif isinstance(file_path, list):
            results = []
            for f in file_path:
                try:
                    results.append(self.parse(f))
                except Exception as e:
                    lc_fail = self.generate_lifecycle(
                        source_file=f,
                        domain=self.domain,
                        life_type=LifeType.DATA_PROCESS_FAILED,
                        usage_purpose="PDF to Markdown"
                    )
                    logger.error(f"Process Fail: {f}, Fail: {e}, LifeCycle: {lc_fail}")
                    vo = MarkdownOutputVo(extension="md", content=f"Process Fail: {e}")
                    vo.add_lifecycle(lc_fail)
                    results.append(vo)
            return results
        else:
            raise ValueError("file_path must be str or list[str]")


    @staticmethod
    def read_pdf_file(file_path) -> str:
        try:
            pdf_loader = PyMuPDFLoader(file_path)
            pdf_documents = pdf_loader.load()
            result_text = ""
            for page in pdf_documents:
                result_text += page.page_content
            return result_text
        except Exception as e:
            raise e

    def parse(self, file_path: str) -> MarkdownOutputVo:
        lc_start = self.generate_lifecycle(
            source_file=file_path,
            domain=self.domain,
            usage_purpose="Documentation",
            life_type=LifeType.DATA_PROCESSING,
        )
        logger.debug("⚙️ DATA_PROCESSING lifecycle generated")
        try:
            extension = self.get_file_extension(file_path)

            if self.use_qwen_vl_ocr:
                # Qwen-VL OCR Process PDF
                ocr_processor = PdfLLMOcrProcessor(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model_name=self.model_name,
                    domain=self.domain
                )
                result = ocr_processor.parse(file_path)
                # avoid IndexError
                if len(result.lifecycle) >= 2:
                    life_cycle_obj = result.lifecycle[1: -1]
                else:
                    life_cycle_obj = result.lifecycle

                if isinstance(result, dict):
                    mk_content = result.get("content", "")
                elif hasattr(result, 'content'):
                    mk_content = result.content
                else:
                    mk_content = str(result)
                
                # save to markdown file
                output_dir = "__temp__"
                output_folder_name = os.path.basename(file_path).replace(".pdf", "")
                output_markdown = f"{output_dir}/markdown/{output_folder_name}.md"
                os.makedirs(os.path.dirname(output_markdown), exist_ok=True)
                with open(output_markdown, "w", encoding="utf-8") as f:
                    f.write(mk_content)
            elif self.use_mineru:
                ## cancel cache
                # output_dir = "__temp__"
                # output_folder_name = os.path.basename(file_path).replace(".pdf", "")
                # output_mineru = f"{output_dir}/markdown/{output_folder_name}.md"

                # if os.path.exists(output_mineru):
                #     mk_content = open(output_mineru, encoding="utf-8").read()
                # else:

                # Lazy import
                from datamax.utils.mineru_operator import pdf_processor
                mk_content = pdf_processor.process_pdf(file_path)
            else:
                content = self.read_pdf_file(file_path=file_path)
                mk_content = content

            # —— Lifecycle: Processing completed —— #
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            logger.debug("⚙️ DATA_PROCESSED lifecycle generated")

            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            # ocr process per page. add lifecycle for each page
            if self.use_qwen_vl_ocr:
                _ = [output_vo.add_lifecycle(ext) for ext in life_cycle_obj]
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception as e:
            # —— Lifecycle: Processing failed —— #
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            logger.debug("⚙️ DATA_PROCESS_FAILED lifecycle generated")

            raise Exception(
                {
                    "error": str(e),
                    "file_path": file_path,
                    "lifecycle": [lc_fail.to_dict()],
                }
            )
