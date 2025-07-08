import os
import pathlib
import sys

from datamax.utils import setup_environment
import dashscope
from typing import Optional

# Initialize environment
setup_environment(use_gpu=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from PIL import Image

from datamax.parser.base import BaseLife
from datamax.parser.pdf_parser import PdfParser
from datamax.utils.lifecycle_types import LifeType


class ImageParser(BaseLife):
    """ImageParser class for parsing images using Qwen model or traditional PDF conversion method.
    
    ## 使用Qwen模型
    ```python
    parser = ImageParser(
        "image.jpg",
        api_key="your_api_key",
        use_mllm=True,
        model_name="qwen-vl-plus",
        system_prompt="Describe the image in detail, focusing on objects, colors, and spatial relationships."
    )
    result = parser.parse("image.jpg", "What is in this image?")
    ```
    ## 使用传统方法
    ```python
    parser = ImageParser("image.jpg")
    result = parser.parse("image.jpg")
    ```
    """
    def __init__(
        self,
        file_path: str,
        use_gpu: bool = False,
        domain: str = "Technology",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = "qwen-vl-plus",
        system_prompt: Optional[str] = "You are a helpful assistant that accurately describes images in detail.",
        use_mllm: bool = False
    ):
        # 初始化 BaseLife，记录 domain
        super().__init__(domain=domain)
        self.domain = domain
        # 可选的 GPU 环境设置
        if use_gpu:
            setup_environment(use_gpu=True)
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        """
        Initialize the ImageParser with optional Qwen model configuration.
        """
        self.file_path = file_path
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.use_mllm = use_mllm
        
        if self.use_mllm:
            if not self.api_key:
                raise ValueError("API key is required when use_mllm is True")
            dashscope.api_key = self.api_key
            if self.base_url:
                dashscope.base_url = self.base_url

    def _parse_with_qwen(self, query: Optional[str] = None) -> str:
        """
        Parse image using Qwen model.
        """
        if query is None:
            query = (
                """
                Describe this image in detail, focusing on objects, and spatial relationships.
                your output should be in the markdown format.
                every object is described in a separate paragraph, with spatial relationships between objects and its possible functions described in the same paragraph.
                """
            )
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                {'image': self.file_path},
                {'text': query}
            ]}
        ]
        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            result_format="message"
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            print(f"HTTP返回码：{response.status_code}")
            print(f"错误码：{response.code}")
            print(f"错误信息：{response.message}")

    def parse(
        self,
        file_path: Optional[str] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Parse the image file using either Qwen model or traditional method.
        
        Args:
            file_path: 新的图片路径（可选），如果提供则覆盖初始化时的 self.file_path
            query:     Qwen 模型的提问内容（可选，仅在 use_mllm=True 时使用）
            
        Returns:
            Qwen 模型返回的文本（use_mllm=True）或传统 OCR 结果（use_mllm=False）
        """
        # 1) 覆盖 self.file_path，以便后续各分支都用同一个路径
        if file_path is not None:
            self.file_path = file_path

        # 2) 必须有图片路径，且文件存在
        if not self.file_path or not os.path.exists(self.file_path):
            raise ValueError(f"未找到图片文件：{self.file_path}")

        try:
            # 3) Qwen 分支：仅使用 query 作为提问内容
            if self.use_mllm:
                return self._parse_with_qwen(query)

            # 4) 传统分支：先将图片转 PDF，再用 PdfParser 解析
            base_name = pathlib.Path(self.file_path).stem
            extension = self.get_file_extension(self.file_path)
            lc_start = self.generate_lifecycle(
                source_file=self.file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )

            output_pdf_path = f"{base_name}.pdf"

            img = Image.open(self.file_path)
            img.save(output_pdf_path, "PDF", resolution=100.0)

            pdf_parser = PdfParser(output_pdf_path, use_mineru=True, domain=self.domain)
            result = pdf_parser.parse(output_pdf_path)

            if os.path.exists(output_pdf_path):
                os.remove(output_pdf_path)

            # 处理结束：生成生命周期结束事件
            content = result.get("content", "")
            lc_end = self.generate_lifecycle(
                source_file=self.file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )

            # 合并生命周期
            lifecycle = result.get("lifecycle", [])
            lifecycle.insert(0, lc_start.to_dict())
            lifecycle.append(lc_end.to_dict())
            result["lifecycle"] = lifecycle

            return result

        except Exception:
            raise


if __name__ == "__main__":
    ip = ImageParser(
        file_path="picture.png",
        use_mllm=True,
        api_key="sk-xxxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-vl-max-latest",
    )
    print(ip.parse())
