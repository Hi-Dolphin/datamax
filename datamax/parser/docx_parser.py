import os
import shutil
import subprocess
import tempfile
import chardet
import logging
from pathlib import Path
from typing import Union
from datamax.parser.base import BaseLife
from datamax.parser.base import MarkdownOutputVo


# 配置日志
logger = logging.getLogger(__name__)


class DocxParser(BaseLife):
    def __init__(self, file_path: Union[str, list], to_markdown: bool = False):
        super().__init__()
        self.file_path = file_path
        self.to_markdown = to_markdown
        logger.info(f"🚀 DocxParser初始化完成 - 文件路径: {file_path}, 转换为markdown: {to_markdown}")

    def docx_to_txt(self, docx_path: str, dir_path: str) -> str:
        """将.docx文件转换为.txt文件"""
        logger.info(f"🔄 开始转换DOCX文件为TXT - 源文件: {docx_path}, 输出目录: {dir_path}")
        
        try:
            cmd = f'soffice --headless --convert-to txt "{docx_path}" --outdir "{dir_path}"'
            logger.debug(f"⚡ 执行转换命令: {cmd}")
            
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            exit_code = process.returncode
            
            if exit_code == 0:
                logger.info(f"✅ DOCX到TXT转换成功 - 退出码: {exit_code}")
                if stdout:
                    logger.debug(f"📄 转换输出: {stdout.decode('utf-8', errors='replace')}")
            else:
                encoding = chardet.detect(stderr)['encoding']
                if encoding is None:
                    encoding = 'utf-8'
                error_msg = stderr.decode(encoding, errors='replace')
                logger.error(f"❌ DOCX到TXT转换失败 - 退出码: {exit_code}, 错误信息: {error_msg}")
                raise Exception(f"Error Output (detected encoding: {encoding}): {error_msg}")
            
            fname = str(Path(docx_path).stem)
            txt_path = os.path.join(dir_path, f'{fname}.txt')
            
            if not os.path.exists(txt_path):
                logger.error(f"❌ 转换后的TXT文件不存在: {txt_path}")
                raise Exception(f"文件转换失败 {docx_path} ==> {txt_path}")
            else:
                logger.info(f"🎉 TXT文件转换成功，文件路径: {txt_path}")
                return txt_path
                
        except subprocess.SubprocessError as e:
            logger.error(f"💥 subprocess执行失败: {str(e)}")
            raise Exception(f"执行转换命令时发生错误: {str(e)}")
        except Exception as e:
            logger.error(f"💥 DOCX到TXT转换过程中发生未知错误: {str(e)}")
            raise

    def read_txt_file(self, txt_path: str) -> str:
        """读取txt文件内容"""
        logger.info(f"📖 开始读取TXT文件: {txt_path}")
        
        try:
            # 检测文件编码
            with open(txt_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                if encoding is None:
                    encoding = 'utf-8'
                logger.debug(f"🔍 检测到文件编码: {encoding}")
            
            # 读取文件内容
            with open(txt_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            logger.info(f"📄 TXT文件读取完成 - 内容长度: {len(content)} 字符")
            logger.debug(f"👀 前100字符预览: {content[:100]}...")
            
            return content
            
        except FileNotFoundError as e:
            logger.error(f"🚫 TXT文件未找到: {str(e)}")
            raise Exception(f"文件未找到: {txt_path}")
        except Exception as e:
            logger.error(f"💥 读取TXT文件时发生错误: {str(e)}")
            raise

    def read_docx_file(self, docx_path: str) -> str:
        """读取docx文件并转换为文本"""
        logger.info(f"📖 开始读取DOCX文件 - 文件: {docx_path}")
        
        try:
            with tempfile.TemporaryDirectory() as temp_path:
                logger.debug(f"📁 创建临时目录: {temp_path}")
                
                temp_dir = Path(temp_path)
                
                file_path = temp_dir / "tmp.docx"
                shutil.copy(docx_path, file_path)
                logger.debug(f"📋 复制文件到临时目录: {docx_path} -> {file_path}")
                
                # 转换DOCX为TXT
                txt_file_path = self.docx_to_txt(str(file_path), str(temp_path))
                logger.info(f"🎯 DOCX转TXT完成: {txt_file_path}")
                
                # 读取TXT文件内容
                content = self.read_txt_file(txt_file_path)
                logger.info(f"✨ TXT文件内容读取完成，内容长度: {len(content)} 字符")
                
                return content
                    
        except FileNotFoundError as e:
            logger.error(f"🚫 文件未找到: {str(e)}")
            raise Exception(f"文件未找到: {docx_path}")
        except PermissionError as e:
            logger.error(f"🔒 文件权限错误: {str(e)}")
            raise Exception(f"无权限访问文件: {docx_path}")
        except Exception as e:
            logger.error(f"💥 读取DOCX文件时发生错误: {str(e)}")
            raise

    def parse(self, file_path: str):
        """解析DOCX文件"""
        logger.info(f"🎬 开始解析DOCX文件: {file_path}")
        
        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                logger.error(f"🚫 文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 验证文件扩展名
            if not file_path.lower().endswith('.docx'):
                logger.warning(f"⚠️ 文件扩展名不是.docx: {file_path}")
            
            # 验证文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 文件大小: {file_size} 字节")
            
            if file_size == 0:
                logger.warning(f"⚠️ 文件大小为0字节: {file_path}")
            
            title = self.get_file_extension(file_path)
            logger.debug(f"🏷️ 提取文件标题: {title}")
            
            # 使用soffice转换为txt后读取内容
            logger.info("📝 使用soffice转换DOCX为TXT并读取内容")
            content = self.read_docx_file(docx_path=file_path)
            
            # 根据to_markdown参数决定是否保持原格式还是处理为markdown格式
            if self.to_markdown:
                # 简单的文本到markdown转换（保持段落结构）
                mk_content = self.format_as_markdown(content)
                logger.info("🎨 内容已格式化为markdown格式")
            else:
                mk_content = content
                logger.info("📝 保持原始文本格式")
            
            logger.info(f"🎊 文件内容解析完成，最终内容长度: {len(mk_content)} 字符")
            
            # 检查内容是否为空
            if not mk_content.strip():
                logger.warning(f"⚠️ 解析出的内容为空: {file_path}")
            
            lifecycle = self.generate_lifecycle(source_file=file_path, domain="Technology",
                                                usage_purpose="Documentation", life_type="LLM_ORIGIN")
            logger.debug("⚙️ 生成lifecycle信息完成")
            
            output_vo = MarkdownOutputVo(title, mk_content)
            output_vo.add_lifecycle(lifecycle)
            
            result = output_vo.to_dict()
            logger.info(f"🏆 DOCX文件解析完成: {file_path}")
            logger.debug(f"🔑 返回结果键: {list(result.keys())}")
            
            return result
            
        except FileNotFoundError as e:
            logger.error(f"🚫 文件不存在错误: {str(e)}")
            raise
        except PermissionError as e:
            logger.error(f"🔒 文件权限错误: {str(e)}")
            raise Exception(f"无权限访问文件: {file_path}")
        except Exception as e:
            logger.error(f"💀 解析DOCX文件失败: {file_path}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}")
            raise

    def format_as_markdown(self, content: str) -> str:
        """将纯文本格式化为简单的markdown格式"""
        if not content.strip():
            return content
        
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # 简单的markdown格式化规则
            # 可以根据需要扩展更多规则
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
