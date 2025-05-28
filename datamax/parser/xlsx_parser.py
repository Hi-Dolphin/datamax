import multiprocessing
import time
import logging
import os
from multiprocessing import Queue
from datamax.parser.base import MarkdownOutputVo
from datamax.parser.base import BaseLife
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 配置日志
logger = logging.getLogger(__name__)


class XlsxParser(BaseLife):
    """XLSX解析器 - 使用pandas读取并转换为markdown，支持多进程处理"""

    def __init__(self, file_path, timeout):
        super().__init__()
        self.file_path = file_path
        self.timeout = timeout
        logger.info(f"🚀 XlsxParser初始化完成 - 文件路径: {file_path}, 超时: {timeout}s")

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
                        sheet_df = sheet_df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not sheet_df.empty:
                            sheet_markdown = sheet_df.to_markdown(index=False)
                            markdown_content += sheet_markdown + "\n\n"
                            logger.debug(f"✅ 工作表 {sheet_name} 转换完成，有效数据形状: {sheet_df.shape}")
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
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if not df.empty:
                        markdown_content = df.to_markdown(index=False)
                        logger.info(f"✅ 工作表转换完成，有效数据形状: {df.shape}")
                    else:
                        markdown_content = "*工作表无有效数据*"
                        logger.warning("⚠️ 工作表清理后无有效数据")
                else:
                    markdown_content = "*工作表为空*"
                    logger.warning("⚠️ 工作表为空")
            
            logger.info(f"🎊 pandas转换完成，markdown内容长度: {len(markdown_content)} 字符")
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
        
        try:
            # 使用pandas解析Excel
            logger.info("🐼 使用pandas模式解析Excel")
            mk_content = self._parse_with_pandas(file_path)
            
            # 检查内容是否为空
            if not mk_content.strip():
                logger.warning(f"⚠️ 解析出的内容为空: {file_path}")
                mk_content = "*无法解析文件内容*"
            
            logger.info(f"🎊 文件内容解析完成，最终内容长度: {len(mk_content)} 字符")
            
            # 生成lifecycle信息
            lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain="Technology",
                usage_purpose="Documentation",
                life_type="LLM_ORIGIN"
            )
            logger.debug("⚙️ 生成lifecycle信息完成")
            
            # 创建输出对象
            title = self.get_file_extension(file_path)
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            
            result = output_vo.to_dict()
            result_queue.put(result)
            logger.info(f"🏆 Excel文件解析完成: {file_path}")
            logger.debug(f"🔑 返回结果键: {list(result.keys())}")
            
            time.sleep(0.5)  # 给队列一点时间
            return result
            
        except Exception as e:
            logger.error(f"💀 解析Excel文件失败: {file_path}, 错误: {str(e)}")
            # 将错误也放入队列
            error_result = {"error": str(e), "file_path": file_path}
            result_queue.put(error_result)
            raise

    def parse(self, file_path: str) -> dict:
        """解析Excel文件 - 支持多进程和超时控制"""
        logger.info(f"🚀 启动Excel解析进程 - 文件: {file_path}, 超时: {self.timeout}s")
        
        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                logger.error(f"🚫 文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 验证文件扩展名
            if not file_path.lower().endswith(('.xlsx', '.xls')):
                logger.warning(f"⚠️ 文件扩展名不是Excel格式: {file_path}")
            
            result_queue = Queue()
            process = multiprocessing.Process(target=self._parse, args=(file_path, result_queue))
            process.start()
            logger.debug(f"⚡ 启动子进程，PID: {process.pid}")
            
            start_time = time.time()
            
            # 等待解析完成或超时
            while time.time() - start_time < self.timeout:
                elapsed_time = int(time.time() - start_time)
                logger.debug(f"⏱️ 等待解析完成... {elapsed_time}s")
                
                if not process.is_alive():
                    logger.debug("✅ 子进程已完成")
                    break
                    
                if not result_queue.empty():
                    result = result_queue.get()
                    process.join()  # 等待进程正常结束
                    
                    # 检查是否是错误结果
                    if "error" in result:
                        logger.error(f"💥 子进程返回错误: {result['error']}")
                        raise Exception(result["error"])
                    
                    logger.info(f"🎉 Excel解析成功完成，耗时: {elapsed_time}s")
                    return result
                    
                time.sleep(1)
            else:
                # 超时处理
                logger.error(f"⏰ 解析超时 ({self.timeout}s)，终止进程")
                process.terminate()
                process.join(timeout=5)  # 给进程5秒时间优雅退出
                
                if process.is_alive():
                    logger.error("💀 强制杀死进程")
                    process.kill()
                    
                raise TimeoutError(f"Excel解析超时: {file_path}")
                
        except Exception as e:
            logger.error(f"💀 Excel解析失败: {file_path}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}")
            raise