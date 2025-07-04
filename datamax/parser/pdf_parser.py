import os
import subprocess
from typing import Union

from langchain_community.document_loaders import PyMuPDFLoader
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType
from datamax.utils.mineru_operator import pdf_processor


class PdfParser(BaseLife):

    def __init__(
        self,
        file_path: Union[str, list],
        use_mineru: bool = False,
        use_qwen_ocr: bool = False,
        api_key: str = None,
        model_name: str = "qwen-vl-max"
    ):
        super().__init__()

        self.file_path = file_path
        self.use_mineru = use_mineru
        self.use_qwen_ocr = use_qwen_ocr
        self.api_key = api_key
        self.model_name = model_name

    def mineru_process(self, input_pdf_filename, output_dir):
        proc = None
        try:
            logger.info(
                f"mineru is working...\n input_pdf_filename: {input_pdf_filename} | output_dir: ./{output_dir}. plz waiting!"
            )
            command = ["magic-pdf", "-p", input_pdf_filename, "-o", output_dir]
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # 等待命令执行完成
            stdout, stderr = proc.communicate()
            # 检查命令是否成功执行
            if proc.returncode != 0:
                raise Exception(
                    f"mineru failed with return code {proc.returncode}: {stderr.decode()}"
                )

            logger.info(
                f"Markdown saved in {output_dir}, input file is {input_pdf_filename}"
            )

        except Exception as e:
            logger.error(f"Error: {e}")
            if proc is not None:
                proc.kill()
                proc.wait()
                logger.info("The process was terminated due to an error.")
            raise  # Re-raise the exception to let the caller handle it

        finally:
            # 确保子进程已经结束
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                    logger.info(
                        "The process was terminated due to timeout or completion."
                    )

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
            domain="Technology",
            usage_purpose="Documentation",
            life_type=LifeType.DATA_PROCESSING,
        )
        logger.debug("⚙️ DATA_PROCESSING 生命周期已生成")
        try:
            if self.use_qwen_ocr:
                from datamax.parser.pdf_parser_qwen_ocr import PdfOcrProcessor
                processor = PdfOcrProcessor(api_key=self.api_key, model_name=self.model_name)
                content = processor.process_pdf(file_path)
                
                # 修正文件名：移除原后缀，添加 .md
                base_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取无后缀的文件名
                output_md_path = f"{base_name}.md"  # 生成 pdftest.md
                
                with open(output_md_path, "w", encoding="utf-8") as f:
                    f.write(content.content if hasattr(content, "content") else str(content))
                
                mk_content = content.content if hasattr(content, "content") else str(content)
                
            elif self.use_mineru:
                output_dir = "uploaded_files"
                output_folder_name = os.path.basename(file_path).replace(".pdf", "")
                # output_mineru = f'{output_dir}/{output_folder_name}/auto/{output_folder_name}.md'
                # if os.path.exists(output_mineru):
                #     pass
                # else:
                # self.mineru_process(input_pdf_filename=file_path, output_dir=output_dir)
                # mk_content = open(output_mineru, 'r', encoding='utf-8').read()

                # todo: 是否有必要跟api的默认保存路径保持一致
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
                domain="Technology",
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            logger.debug("⚙️ DATA_PROCESSED 生命周期已生成")

            output_vo = MarkdownOutputVo(self.get_file_extension(file_path), mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            return output_vo.to_dict()

        except Exception as e:
            # —— 生命周期：处理失败 —— #
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
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
