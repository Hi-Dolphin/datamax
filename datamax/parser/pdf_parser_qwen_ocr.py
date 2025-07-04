import os
import fitz  # PyMuPDF
from PIL import Image
import tempfile
import dashscope
from typing import List
from contextlib import suppress
from datamax.parser.base import MarkdownOutputVo, BaseLife
import re

class PdfOcrProcessor(BaseLife):  # 继承BaseLife以复用生命周期管理
    """PDF转Markdown处理器（集成生命周期跟踪）"""
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "qwen-vl-ocr-latest"
    ):
        """
        Args:
            api_key: DashScope API密钥
            model_name: 模型名称 (默认qwen-vl-ocr-latest)
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        dashscope.api_key = self.api_key

    def _pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """PDF转临时图片文件（自动资源清理）"""
        temp_image_paths = []
        doc = fitz.open(pdf_path)
        
        try:
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=dpi)
                
                with Image.frombytes("RGB", (pix.width, pix.height), pix.samples) as img:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        img.save(temp_file.name, "JPEG", quality=95)
                        temp_image_paths.append(temp_file.name)
            
            return temp_image_paths
        finally:
            doc.close()

    def _ocr_page_to_markdown(self, image_path: str) -> MarkdownOutputVo:
        """
        单页OCR识别（返回Markdown对象）
        Args:
            image_path: 图片路径
        Returns:
            MarkdownOutputVo对象（含内容和元数据）
        """
        messages = [{
            "role": "system",
            "content": "你是一个Markdown转换专家，请将文档内容转换为标准Markdown格式：\n"
                       "- 表格使用Markdown语法\n"
                       "- 数学公式用$$包裹\n"
                       "- 保留原始段落结构"
        }, {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": "请以Markdown格式输出本页所有内容"}
            ]
        }]

        response = dashscope.MultiModalConversation.call(
            model=self.model_name,
            messages=messages,
            parameters={"result_format": "markdown"}
        )

        # 构造Markdown输出对象
        raw_text = response.output.choices[0].message.content[0]["text"]
        return MarkdownOutputVo(
            extension="md",
            content=self._format_markdown(raw_text)
        )

    def _format_markdown(self, text: str) -> str:
        """Markdown后处理"""
        # 表格对齐优化
        text = re.sub(r'\|(\s*\-+\s*)\|', r'|:---:|', text)
        # 清理多余空行
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def process_pdf(self, pdf_path: str) -> MarkdownOutputVo:
        """
        处理PDF文档（全流程）
        Args:
            pdf_path: PDF文件路径
        Returns:
            包含所有页面内容和生命周期的MarkdownOutputVo
        """
        image_paths = self._pdf_to_images(pdf_path)
        combined_md = MarkdownOutputVo(extension="md", content="")

        try:
            for i, img_path in enumerate(image_paths):
                print(f"Processing page {i+1}/{len(image_paths)}")
                page_md = self._ocr_page_to_markdown(img_path)
                
                # 合并内容
                combined_md.content += f"## 第 {i+1} 页\n\n{page_md.content}\n\n"
                
                # 记录生命周期
                combined_md.add_lifecycle(
                    self.generate_lifecycle(
                        source_file=img_path,
                        domain="document_ocr",
                        life_type="text_extraction",
                        usage_purpose="PDF转Markdown"
                    )
                )
                
                # 安全删除临时文件
                with suppress(PermissionError):
                    os.unlink(img_path)
            
            return combined_md
        except Exception as e:
            # 异常时清理文件
            for p in image_paths:
                with suppress(PermissionError):
                    os.unlink(p)
            raise