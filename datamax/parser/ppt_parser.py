import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import chardet
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType
from datamax.utils.ppt_extract import PPtExtractor

# 尝试导入UNO处理器
try:
    from datamax.utils.uno_handler import HAS_UNO, convert_with_uno
except ImportError:
    HAS_UNO = False


class PptParser(BaseLife):
    def __init__(self, file_path: Union[str, list], use_uno: bool = None, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

        # 自动检测是否使用UNO（如果未指定）
        if use_uno is None:
            self.use_uno = HAS_UNO
        else:
            self.use_uno = use_uno and HAS_UNO

    def ppt_to_pptx(self, ppt_path: str, dir_path: str) -> str:
        if self.use_uno:
            # 使用UNO API进行转换
            try:
                pptx_path = convert_with_uno(ppt_path, "pptx", dir_path)

                if not os.path.exists(pptx_path):
                    raise Exception(
                        f"> !!! File conversion failed {ppt_path} ==> {pptx_path}"
                    )
                else:
                    return pptx_path

            except Exception as e:
                if (
                    hasattr(self, "_fallback_to_subprocess")
                    and self._fallback_to_subprocess
                ):
                    return self._ppt_to_pptx_subprocess(ppt_path, dir_path)
                raise
        else:
            # 使用传统的subprocess方式
            return self._ppt_to_pptx_subprocess(ppt_path, dir_path)

    def _ppt_to_pptx_subprocess(self, ppt_path: str, dir_path: str) -> str:
        """使用subprocess将.ppt文件转换为.pptx文件（传统方式）"""
        cmd = f'soffice --headless --convert-to pptx "{ppt_path}" --outdir "{dir_path}"'
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        if exit_code == 0:
            pass
        else:
            encoding = chardet.detect(stderr)["encoding"]
            if encoding is None:
                encoding = "utf-8"
            raise Exception(
                f"Error Output (detected encoding: {encoding}):",
                stderr.decode(encoding, errors="replace"),
            )
        fname = str(Path(ppt_path).stem)
        pptx_path = os.path.join(os.path.dirname(ppt_path), f"{fname}.pptx")
        if not os.path.exists(pptx_path):
            raise Exception(f"> !!! File conversion failed {ppt_path} ==> {pptx_path}")
        else:
            return pptx_path

    def read_ppt_file(self, file_path: str):
        try:
            with tempfile.TemporaryDirectory() as temp_path:
                temp_dir = Path(temp_path).resolve()
                media_dir = temp_dir / "media"
                media_dir.mkdir()
                tmp_file_path = temp_dir / "tmp.ppt"
                shutil.copy(file_path, tmp_file_path)
                pptx_file_path = self.ppt_to_pptx(
                    ppt_path=str(tmp_file_path), dir_path=temp_path
                )
                pptx_extractor = PPtExtractor()
                pages_list = pptx_extractor.extract(
                    Path(pptx_file_path), "tmp", temp_dir, media_dir, True
                )
                contents = ""
                for index, page in enumerate(pages_list):
                    page_content_list = page["content_list"]
                    for content in page_content_list:
                        if content["type"] == "image":
                            pass
                        elif content["type"] == "text":
                            data = content["data"]
                            contents += data
                return contents
        except Exception:
            raise

    def parse(self, file_path: str) -> MarkdownOutputVo:
        # —— 生命周期：开始处理 PPT —— #
        lc_start = self.generate_lifecycle(
            source_file=file_path,
            domain=self.domain,
            usage_purpose="Documentation",
            life_type=LifeType.DATA_PROCESSING,
        )
        logger.debug("⚙️ DATA_PROCESSING 生命周期已生成")

        try:
            extension = self.get_file_extension(file_path)
            content = self.read_ppt_file(file_path=file_path)
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

            # 返回包含失败生命周期的异常信息
            raise Exception(
                {
                    "error": str(e),
                    "file_path": file_path,
                    "lifecycle": [lc_fail.to_dict()],
                }
            )
