import os
import pathlib

from datamax.utils import setup_environment
import openai
import base64
from mimetypes import guess_type
from typing import Optional

from PIL import Image

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.parser.pdf_parser import PdfParser
from datamax.utils.lifecycle_types import LifeType

from loguru import logger

class ImageParser(BaseLife):
    """ImageParser class for parsing images using Vision model or traditional PDF conversion method.

        ## Using Vision Model
        ```python
        parser = ImageParser(
            "image.jpg",
            api_key="your_api_key",
            use_mllm=True,
            model_name="gpt-4o",
            system_prompt="Describe the image in detail, focusing on objects, colors, and spatial relationships."
        )
        result = parser.parse("image.jpg", "What is in this image?")
        ```
        ## Using Traditional Method
        ```python
        parser = ImageParser("image.jpg")
        result = parser.parse("image.jpg")
        ```
    """
    def __init__(
        self,
        file_path: str,
        system_prompt: Optional[str],
        use_gpu: bool = False,
        domain: str = "Technology",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = "gpt-4o",
        use_mllm: bool = False
    ):
        # Initialize BaseLife, record domain
        super().__init__(domain=domain)

        # Optional GPU environment setup
        if use_gpu:
            setup_environment(use_gpu=True)
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        """
        Initialize the ImageParser with optional Vision model configuration.

        Args:
            file_path: Path to the image file
            api_key: API key for OpenAI service (default: None)
            base_url: Base URL for OpenAI API (default: None)
            model_name: Vision model name (default: "gpt-4o")
            system_prompt: System prompt for the model (default: descriptive prompt)
            use_mllm: Whether to use Vision model for image parsing (default: False)
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
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url if self.base_url else None)

    def _parse_with_mllm(self, query: str) -> str:
        """
        Parse image using Vision model.

        Args:
            image_path: Path to the image file
            query: The question/prompt for the image (default: "Describe this image in detail.")

        Returns:
            The model's response as a string
        """
        logger.success(f"⏳ Using Vision model to parse image: {self.file_path}")
        logger.debug(f"system_prompt: {self.system_prompt}")

        if query is None:
            query = """
            Describe this image in detail, focusing on objects, and spatial relationships.
            your output should be in the markdown format.
            every object is described in a separate paragraph, with spatial relationships between objects and its possible functions described in the same paragraph.
            """

        with open(self.file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        mime_type, _ = guess_type(self.file_path)
        if not mime_type:
            mime_type = 'image/jpeg'  # default

        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': query},
                    {'type': 'image_url', 'image_url': {'url': f"data:{mime_type};base64,{encoded_string}"}}
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return response.choices[0].message.content

    def parse(self, file_path: str, query: Optional[str] = None) -> str:
        """
        Parse the image file using either Vision model or traditional PDF conversion method.

        Args:
            file_path: Path to the image file
            query: Optional query/prompt for Vision model (default: None)

        Returns:
            Parsed text content from the image
        """
        try:
            if self.use_mllm:
                # 1) Processing start: generate DATA_PROCESSING event
                extension = self.get_file_extension(file_path)

                lc_start = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type=LifeType.DATA_PROCESSING,
                    usage_purpose="Parsing",
                )
                llm_res = self._parse_with_mllm(query)
                output_vo = MarkdownOutputVo(extension, llm_res)

                # 2) Processing end: generate DATA_PROCESSED event
                lc_end = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    life_type=(
                        LifeType.DATA_PROCESSED
                        if llm_res.strip()
                        else LifeType.DATA_PROCESS_FAILED
                    ),
                    usage_purpose="Parsing",
                )
                output_vo.add_lifecycle(lc_start)
                output_vo.add_lifecycle(lc_end)
                return output_vo.to_dict()
            
            # Fall back to traditional method if not using Vision
            base_name = pathlib.Path(file_path).stem

            # 1) Processing start: generate DATA_PROCESSING event
            extension = self.get_file_extension(file_path)
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )

            output_pdf_path = f"{base_name}.pdf"

            img = Image.open(file_path)
            img.save(output_pdf_path, "PDF", resolution=100.0)

            pdf_parser = PdfParser(output_pdf_path, use_mineru=True)
            result = pdf_parser.parse(output_pdf_path)

            if os.path.exists(output_pdf_path):
                os.remove(output_pdf_path)
            # 2) Processing end: generate DATA_PROCESSED or DATA_PROCESS_FAILED based on whether content is non-empty
            content = result.get("content", "")
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )

            # 3) Merge lifecycle: insert start first, then append end
            lifecycle = result.get("lifecycle", [])
            lifecycle.insert(0, lc_start.to_dict())
            lifecycle.append(lc_end.to_dict())
            result["lifecycle"] = lifecycle

            return result

        except Exception:
            raise
