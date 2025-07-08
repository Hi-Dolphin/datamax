import os
import pathlib
import sys
import subprocess
from typing import Union, Optional, Dict

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType
from langchain_community.document_loaders import PyMuPDFLoader
from loguru import logger
from datamax.utils.mineru_operator import pdf_processor
import os


class PdfParser(BaseLife):
    def __init__(
        self,
        file_path: Union[str, list],
        use_mineru: bool = False,
        use_got_ocr: bool = False,
        got_ocr_model_path: str = "./GOT_weights/",
        gpu_id: int = 0,
        extract_images: bool = False,
        image_output_dir: str = "extracted_images",
        temp_dir: str = "__temp__",
        domain: str = "Technology",
        **kwargs
    ):
        """
        初始化 PdfParser。

        Args:
            file_path: PDF 或 PPT 文件路径，或路径列表
            use_mineru: 是否使用 minerU 处理
            use_got_ocr: 是否使用 GOT-OCR 处理
            got_ocr_model_path: GOT-OCR 模型权重路径
            gpu_id: GPU ID
            extract_images: 是否提取文档中的图片
            image_output_dir: 提取图片的保存目录
            temp_dir: 临时文件夹
            domain: 生命周期域（默认 Technology）
            **kwargs: 其他可选参数（留给后续扩展）
        """
        # 把 domain 传给父类，才能生成正确的 lifecycle
        super().__init__(domain=domain)

        self.file_path = file_path
        self.use_mineru = use_mineru
        self.use_got_ocr = use_got_ocr
        self.got_ocr_model_path = got_ocr_model_path
        self.gpu_id = gpu_id
        self.extract_images = extract_images
        self.image_output_dir = image_output_dir
        self.temp_dir = temp_dir
        self.domain = domain

        # 确保临时目录存在
        os.makedirs(self.temp_dir, exist_ok=True)

    def mineru_process(self, input_pdf_filename, output_dir):
        """Original mineru processing method"""
        proc = None
        try:
            logger.info(f"mineru is working...\n input_pdf_filename: {input_pdf_filename} | output_dir: ./{output_dir}. plz waiting!")
            command = ['magic-pdf', '-p', input_pdf_filename, '-o', output_dir]
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise Exception(f"mineru failed with return code {proc.returncode}: {stderr.decode()}")

            logger.info(f"Markdown saved in {output_dir}, input file is {input_pdf_filename}")

        except Exception as e:
            logger.error(f"Error: {e}")
            if proc is not None:
                proc.kill()
                proc.wait()
                logger.info("The process was terminated due to an error.")
            raise

        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
                logger.info("The process was terminated due to timeout or completion.")

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
        try:
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSING,
            )
            logger.debug("⚙️ DATA_PROCESSING 生命周期已生成")

            extension = self.get_file_extension(file_path)
            
            # Choose processing method based on configuration
            if self.use_got_ocr:
                mk_content = self._process_with_got_ocr(file_path)
            elif self.use_mineru:
                output_dir = self.temp_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_folder_name = os.path.basename(file_path).replace(".pdf", "")
                output_mineru = f'{output_dir}/markdown/{output_folder_name}.md'

                if os.path.exists(output_mineru):
                    mk_content = open(output_mineru, 'r', encoding='utf-8').read()
                else:
                    mk_content = pdf_processor.process_pdf(file_path, output_dir=output_dir)
            else:
                # Fall back to basic text extraction
                pdf_loader = PyMuPDFLoader(file_path)
                pdf_documents = pdf_loader.load()
                mk_content = ''.join(page.page_content for page in pdf_documents)
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise

        try:
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
    import torch
    print(torch.cuda.is_available())
    file_path = "D:/Datamax_hkh/examples/testfiles/图纸/图、文/3.VALVE SPINDLE GRINDER.pdf"
    parser = PdfParser(file_path, use_mineru=True)
    result = parser.parse(file_path)
    print(result)