import base64
import mimetypes
import os
import pathlib
import sys
from openai import OpenAI
from datamax.utils import setup_environment
from typing import Optional



ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from PIL import Image

from datamax.parser.base import BaseLife
from datamax.parser.pdf_parser import PdfParser
from datamax.utils.lifecycle_types import LifeType


class ImageParser(BaseLife):
    """ImageParser class for parsing images using Qwen model or traditional PDF conversion method.
    
        ## Using Qwen Model
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
        ## Using Traditional Method
        ```python
        parser = ImageParser("image.jpg", use_mllm=False)
        result = parser.parse("image.jpg")
        ```
    """
    def __init__(
        self,
        file_path: str,
        use_mllm: bool = False,
        domain: str = "Technology",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = "qwen-vl-plus",
        system_prompt: Optional[str] = "You are a helpful assistant that accurately describes images in detail.",
        use_gpu: bool = False
    ):
        # Initialize BaseLife, record domain
        super().__init__(domain=domain)

        # Optional GPU environment setup
        if use_gpu:
            setup_environment(use_gpu=True)
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        """
        Initialize the ImageParser with optional Qwen model configuration.

        Args:
            file_path: Path to the image file
            api_key: API key for Qwen service (default: None)
            base_url: Base URL for Qwen API (default: None)
            model_name: Qwen model name (default: "qwen-vl-plus")
            system_prompt: System prompt for the model (default: descriptive prompt)
            use_mllm: Whether to use the professional parser (MLLM for images).
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
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _encode_image_to_base64(self, file_path: str) -> str:
        """
        Encodes an image file to a Base64 data URI.

        Args:
            file_path: The path to the image file.

        Returns:
            A Base64 encoded data URI string.
        """
        # Infer the MIME type of the image from the file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # Default to JPEG if the MIME type cannot be determined
            mime_type = "image/jpeg"

        # Read the image file in binary mode
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Format as a data URI
        return f"data:{mime_type};base64,{encoded_string}"

    def _parse_with_mllm(self, prompt: str) -> str:
        """
        Parse image using Qwen model.
        
        Args:
            prompt: The question/prompt for the image.

        Returns:
            The model's response as a string.
        """
        if prompt is None:
            prompt = f"""
            Describe this image in detail, focusing on objects, and spatial relationships.
            your output should be in the markdown format.
            every object is described in a separate paragraph, with spatial relationships between objects and its possible functions described in the same paragraph.
            """

        # Encode the local image to a Base64 data URI
        base64_image = self._encode_image_to_base64(self.file_path)

        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': [
                    # Use the Base64 data URI instead of a file path
                    {'type': 'image_url', 'image_url': {'url': base64_image}},
                    {'type': 'text', 'text': prompt}
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""


    def parse(self, file_path: str, system_prompt: Optional[str] = None) -> str:
        """
        Parse the image file using either Qwen model or traditional PDF conversion method.

        Args:
            file_path: Path to the image file
            prompt: Optional prompt/prompt for Qwen model (default: None)

        Returns:
            Parsed text content from the image
        """
        try:
            if self.use_mllm:
                return self._parse_with_mllm(system_prompt)

            # Fall back to traditional method if not using pro parser
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

if __name__ == "__main__":
    ip = ImageParser(
        file_path="picture.png",
        use_mllm=True,
        api_key="sk-xxxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-vl-max-latest",
        )
    print(ip.parse())