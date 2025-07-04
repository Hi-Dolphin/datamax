import multiprocessing
import os
import time
import warnings
from multiprocessing import Queue

import pandas as pd
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType

warnings.filterwarnings("ignore")


class XlsxParser(BaseLife):
    """XLSX解析器 - 使用pandas读取并转换为markdown，支持多进程处理"""

    def __init__(self, file_path, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path
        logger.info(f"🚀 XlsxParser初始化完成 - 文件路径: {file_path}")

    def _parse_with_pandas(self, file_path: str) -> str:
        """使用pandas读取Excel并转换为markdown"""
        logger.info(f"🐼 开始使用pandas读取Excel文件: {file_path}")

        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                logger.error(f"🚫 Excel文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 验证文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 文件大小: {file_size} 字节")

            if file_size == 0:
                logger.warning(f"⚠️ 文件大小为0字节: {file_path}")
                return "*文件为空*"

            # 使用pandas读取Excel文件
            logger.debug("📊 正在读取Excel数据...")
            df = pd.read_excel(file_path, sheet_name=None)  # 读取所有工作表

            markdown_content = ""

            if isinstance(df, dict):
                # 多个工作表
                logger.info(f"📑 检测到多个工作表，共 {len(df)} 个")
                for sheet_name, sheet_df in df.items():
                    logger.debug(f"📋 处理工作表: {sheet_name}, 形状: {sheet_df.shape}")
                    markdown_content += f"## 工作表: {sheet_name}\n\n"

                    if not sheet_df.empty:
                        # 清理数据：移除完全为空的行和列
                        sheet_df = sheet_df.dropna(how="all").dropna(axis=1, how="all")

                        if not sheet_df.empty:
                            sheet_markdown = sheet_df.to_markdown(index=False)
                            markdown_content += sheet_markdown + "\n\n"
                            logger.debug(
                                f"✅ 工作表 {sheet_name} 转换完成，有效数据形状: {sheet_df.shape}"
                            )
                        else:
                            markdown_content += "*该工作表无有效数据*\n\n"
                            logger.warning(f"⚠️ 工作表 {sheet_name} 清理后无有效数据")
                    else:
                        markdown_content += "*该工作表为空*\n\n"
                        logger.warning(f"⚠️ 工作表 {sheet_name} 为空")
            else:
                # 单个工作表
                logger.info(f"📄 单个工作表，形状: {df.shape}")
                if not df.empty:
                    # 清理数据：移除完全为空的行和列
                    df = df.dropna(how="all").dropna(axis=1, how="all")

                    if not df.empty:
                        markdown_content = df.to_markdown(index=False)
                        logger.info(f"✅ 工作表转换完成，有效数据形状: {df.shape}")
                    else:
                        markdown_content = "*工作表无有效数据*"
                        logger.warning("⚠️ 工作表清理后无有效数据")
                else:
                    markdown_content = "*工作表为空*"
                    logger.warning("⚠️ 工作表为空")

            logger.info(
                f"🎊 pandas转换完成，markdown内容长度: {len(markdown_content)} 字符"
            )
            logger.debug(f"👀 前200字符预览: {markdown_content[:200]}...")

            return markdown_content

        except FileNotFoundError as e:
            logger.error(f"🚫 文件未找到: {str(e)}")
            raise
        except PermissionError as e:
            logger.error(f"🔒 文件权限错误: {str(e)}")
            raise Exception(f"无权限访问文件: {file_path}")
        except pd.errors.EmptyDataError as e:
            logger.error(f"📭 Excel文件为空: {str(e)}")
            raise Exception(f"Excel文件为空或无法读取: {file_path}")
        except Exception as e:
            logger.error(f"💥 pandas读取Excel失败: {str(e)}")
            raise

    def _parse(self, file_path: str, result_queue: Queue) -> dict:
        """解析Excel文件的核心方法"""
        logger.info(f"🎬 开始解析Excel文件: {file_path}")

        # —— 生命周期：开始处理 —— #
        lc_start = self.generate_lifecycle(
            source_file=file_path,
            domain=self.domain,
            usage_purpose="Documentation",
            life_type=LifeType.DATA_PROCESSING,
        )
        logger.debug("⚙️ DATA_PROCESSING 生命周期已生成")

        try:
            # 使用pandas解析Excel
            logger.info("🐼 使用pandas模式解析Excel")
            mk_content = self._parse_with_pandas(file_path)

            # 检查内容是否为空
            if not mk_content.strip():
                logger.warning(f"⚠️ 解析出的内容为空: {file_path}")
                mk_content = "*无法解析文件内容*"

            logger.info(f"🎊 文件内容解析完成，最终内容长度: {len(mk_content)} 字符")

            # —— 生命周期：处理完成 —— #
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            logger.debug("⚙️ DATA_PROCESSED 生命周期已生成")

            # 创建输出对象并添加两个生命周期
            extension = self.get_file_extension(file_path)
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)

            result = output_vo.to_dict()
            result_queue.put(result)
            logger.info(f"🏆 Excel文件解析完成: {file_path}")
            logger.debug(f"🔑 返回结果键: {list(result.keys())}")

            time.sleep(0.5)  # 给队列一点时间
            return result

        except Exception as e:
            # —— 生命周期：处理失败 —— #
            try:
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    usage_purpose="Documentation",
                    life_type=LifeType.DATA_PROCESS_FAILED,
                )
                logger.debug("⚙️ DATA_PROCESS_FAILED 生命周期已生成")
                # 如果需要，也可以把它加到 error_result 里：
                # error_result = {"error": str(e), "file_path": file_path, "lifecycle":[lc_fail.to_dict()]}
            except Exception:
                pass

            # —— 生命周期：处理失败 —— #
            try:
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    usage_purpose="Documentation",
                    life_type=LifeType.DATA_PROCESS_FAILED,
                )
                logger.debug("⚙️ DATA_PROCESS_FAILED 生命周期已生成")
            except Exception:
                pass

            logger.error(f"💀 解析Excel文件失败: {file_path}, 错误: {str(e)}")
            # 将错误也放入队列
            error_result = {
                "error": str(e),
                "file_path": file_path,
                # 额外把失败的 lifecycle 也一起返回，测试中可选校验
                "lifecycle": [lc_fail.to_dict()] if "lc_fail" in locals() else [],
            }
            result_queue.put(error_result)
            raise

    def parse(self, file_path: str) -> dict:
        """解析Excel文件 - 支持多进程和超时控制"""
        logger.info(f"🚀 启动Excel解析进程 - 文件: {file_path}")

        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                logger.error(f"🚫 文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 验证文件扩展名
            if not file_path.lower().endswith((".xlsx", ".xls")):
                logger.warning(f"⚠️ 文件扩展名不是Excel格式: {file_path}")

            result_queue = Queue()
            process = multiprocessing.Process(
                target=self._parse, args=(file_path, result_queue)
            )
            process.start()
            logger.debug(f"⚡ 启动子进程，PID: {process.pid}")

        except Exception as e:
            logger.error(
                f"💀 Excel解析失败: {file_path}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}"
            )
            raise
