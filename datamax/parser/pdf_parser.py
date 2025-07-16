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
from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType
from datamax.utils.mineru_operator import pdf_processor

class PdfOcrProcessor(BaseLife):
    """PDF转Markdown"""

    def __init__(self, api_key: str, base_url: str, model_name: str, domain: str = "Technology"):
        # OCR模型可选列表
        OCR_MODEL_MAP = {
            "qwen-vl-ocr": "qwen-vl-ocr",
            "qwen-vl-ocr-latest": "qwen-vl-ocr-latest",
            "qwen-vl-max-latest": "qwen-vl-max-latest",
            "qwen-vl-max": "qwen-vl-max",
            "qwen-vl-plus": "qwen-vl-plus",
            "qwen-vl-plus-latest": "qwen-vl-plus-latest",
        }
        if model_name not in OCR_MODEL_MAP and model_name not in OCR_MODEL_MAP.values():
            raise ValueError("ocr_model_name参数错误，只能为：1(qwen-vl-ocr)")
        super().__init__(domain=domain)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = OCR_MODEL_MAP.get(model_name, model_name)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.domain = domain

    def _pdf_to_images(self, file_path: str, dpi: int = 300) -> List[str]:
        logger.info(f"PDF转图片开始: {file_path}")
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
            logger.info(f"PDF转图片完成，共 {len(temp_image_paths)} 页: {file_path}")
            return temp_image_paths
        finally:
            doc.close()

    @staticmethod
    def encode_image(image_path):
        return base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

    def _ocr_page_to_markdown(self, image_path: str) -> MarkdownOutputVo:
        logger.info(f"OCR识别图片: {image_path}")
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
            logger.info(f"OCR识别完成: {image_path}")
            return MarkdownOutputVo(
                extension="md",
                content=self._format_markdown(raw_text)
            )
        except Exception as e:
            logger.error(f"OCR识别失败: {image_path}, 错误: {e}")
            raise

    def _format_markdown(self, text: str) -> str:
        text = re.sub(r'\|(\s*\-+\s*)\|', r'|:---:|', text)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def parse(self, file_path: Union[str, List[str]]) -> Union[MarkdownOutputVo, List[MarkdownOutputVo]]:
        """
        支持单文件或多文件PDF转Markdown
        Returns:
            MarkdownOutputVo 或 MarkdownOutputVo 列表
        """
        if isinstance(file_path, str):
            logger.info(f"开始处理PDF: {file_path}")
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type="DATA_PROCESSING",
                usage_purpose="PDF转Markdown"
            )
            logger.debug(f"⚙️ DATA_PROCESSING 生命周期已生成: {lc_start}")
            combined_md = MarkdownOutputVo(extension="md", content="")
            combined_md.add_lifecycle(lc_start)
            image_paths = self._pdf_to_images(file_path)
            try:
                for i, img_path in enumerate(image_paths):
                    logger.info(f"Processing page {i+1}/{len(image_paths)}: {file_path}")
                    page_md = self._ocr_page_to_markdown(img_path)
                    combined_md.content += f"## 第 {i+1} 页\n\n{page_md.content}\n\n"
                    lc_page = self.generate_lifecycle(
                        source_file=img_path,
                        domain="document_ocr",
                        life_type="text_extraction",
                        usage_purpose="PDF转Markdown"
                    )
                    logger.debug(f"⚙️ text_extraction 生命周期已生成: {lc_page}")
                    combined_md.add_lifecycle(lc_page)
                    with suppress(PermissionError):
                        os.unlink(img_path)
                lc_end = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type="DATA_PROCESSED",
                    usage_purpose="PDF转Markdown"
                )
                logger.debug(f"⚙️ DATA_PROCESSED 生命周期已生成: {lc_end}")
                combined_md.add_lifecycle(lc_end)
                logger.info(f"处理完成: {file_path}")
                return combined_md
            except Exception as e:
                for p in image_paths:
                    with suppress(PermissionError):
                        os.unlink(p)
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type="DATA_PROCESS_FAILED",
                    usage_purpose="PDF转Markdown"
                )
                logger.error(f"处理失败: {file_path}, 错误: {e}, 生命周期: {lc_fail}")
                combined_md.add_lifecycle(lc_fail)
                combined_md.content += f"\n处理失败: {e}"
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
                        life_type="DATA_PROCESS_FAILED",
                        usage_purpose="PDF转Markdown"
                    )
                    logger.error(f"批量处理失败: {f}, 错误: {e}, 生命周期: {lc_fail}")
                    vo = MarkdownOutputVo(extension="md", content=f"处理失败: {e}")
                    vo.add_lifecycle(lc_fail)
                    results.append(vo)
            return results
        else:
            raise ValueError("file_path 必须为 str 或 list[str]")


class PdfParser(BaseLife):

    def __init__(
        self,
        file_path: Union[str, list],
        use_mineru: bool = False,
        use_qwen_vl_ocr: bool = False,
        domain: str = "Technology",
        ocr_api_key: str = None,
        ocr_base_url: str = None,
        ocr_model_name: str = None,
    ):
        super().__init__(domain=domain)

        self.file_path = file_path
        self.use_mineru = use_mineru
        self.use_qwen_vl_ocr = use_qwen_vl_ocr
        self.ocr_api_key = ocr_api_key
        self.ocr_base_url = ocr_base_url
        self.ocr_model_name = ocr_model_name
        
        # 验证OCR参数
        if self.use_qwen_vl_ocr:
            if not all([self.ocr_api_key, self.ocr_base_url, self.ocr_model_name]):
                raise ValueError("Qwen-VL OCR requires api_key, base_url, and model_name to be provided")

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
        logger.debug("⚙️ DATA_PROCESSING 生命周期已生成")
        try:
            extension = self.get_file_extension(file_path)

            if self.use_qwen_vl_ocr:
                # 使用Qwen-VL OCR处理PDF
                ocr_processor = PdfOcrProcessor(
                    api_key=self.ocr_api_key,
                    base_url=self.ocr_base_url,
                    model_name=self.ocr_model_name,
                    domain=self.domain
                )
                result = ocr_processor.parse(file_path)
                life_cycle_obj = result.lifecycle[1: -1] # 去除首尾ocr生命周期

                if isinstance(result, dict):
                    mk_content = result.get("content", "")
                elif hasattr(result, 'content'):
                    mk_content = result.content
                else:
                    mk_content = str(result)
                
                # 保存为markdown文件
                output_dir = "uploaded_files"
                output_folder_name = os.path.basename(file_path).replace(".pdf", "")
                output_markdown = f"{output_dir}/markdown/{output_folder_name}.md"
                os.makedirs(os.path.dirname(output_markdown), exist_ok=True)
                with open(output_markdown, "w", encoding="utf-8") as f:
                    f.write(mk_content)
            elif self.use_mineru:
                output_dir = "uploaded_files"
                output_folder_name = os.path.basename(file_path).replace(".pdf", "")
                output_mineru = f"{output_dir}/markdown/{output_folder_name}.md"

                if os.path.exists(output_mineru):
                    mk_content = open(output_mineru, "r", encoding="utf-8").read()
                else:
                    mk_content = pdf_processor.process_pdf(file_path)
            else:
                content = self.read_pdf_file(file_path=file_path)
                mk_content = content

            # —— 生命周期：处理完成 —— #
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            logger.debug("⚙️ DATA_PROCESSED 生命周期已生成")

            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            # ocr处理单独页生命周期
            if self.use_qwen_vl_ocr:
                _ = [output_vo.add_lifecycle(ext) for ext in life_cycle_obj]
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception as e:
            # —— 生命周期：处理失败 —— #
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            logger.debug("⚙️ DATA_PROCESS_FAILED 生命周期已生成")

            raise Exception(
                {
                    "error": str(e),
                    "file_path": file_path,
                    "lifecycle": [lc_fail.to_dict()],
                }
            )

if __name__ == "__main__":
    # 简单演示：单文件或多文件 PDF 转 Markdown，只打印最终内容和生命周期
    processor = PdfOcrProcessor(
        api_key="sk-xxx",
        base_url="your_base_url_here",
        model_name="your_model_here"
    )
    # 单文件
    vo = processor.parse("test.pdf")
    print(vo)
    # 多文件
    vos = processor.parse(["test1.pdf", "test2.pdf"])
    if isinstance(vos, list):
        for idx, vo in enumerate(vos):
            print(f"文件{idx+1}结果:")
            print(vo)
            print("="*40)
    else:
        print(vos)
