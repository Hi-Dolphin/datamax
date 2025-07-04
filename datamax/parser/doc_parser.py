import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import chardet
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType

# 尝试导入OLE相关库（用于读取DOC内部结构）
try:
    import olefile

    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False
    logger.warning("⚠️ olefile库未安装，无法进行高级DOC解析")

# 尝试导入UNO处理器
try:
    from datamax.utils.uno_handler import HAS_UNO, convert_with_uno
except ImportError:
    HAS_UNO = False
    logger.error(
        "❌ UNO处理器导入失败！\n"
        "🔧 解决方案：\n"
        "1. 安装LibreOffice和python-uno：\n"
        "   - Ubuntu/Debian: sudo apt-get install libreoffice python3-uno\n"
        "   - CentOS/RHEL: sudo yum install libreoffice python3-uno\n"
        "   - macOS: brew install libreoffice\n"
        "   - Windows: 下载并安装LibreOffice\n"
        "2. 确保Python可以访问uno模块：\n"
        "   - Linux: export PYTHONPATH=/usr/lib/libreoffice/program:$PYTHONPATH\n"
        "   - Windows: 添加LibreOffice\\program到系统PATH\n"
        "3. 验证安装：python -c 'import uno'\n"
        "4. 如果仍有问题，请查看完整文档：\n"
        "   https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
    )


class DocParser(BaseLife):
    def __init__(
        self,
        file_path: Union[str, list],
        to_markdown: bool = False,
        use_uno: bool = True,
        domain: str = "Technology"
    ):
        super().__init__(domain=domain)
        self.file_path = file_path
        self.to_markdown = to_markdown

        # 优先使用UNO（除非明确禁用）
        if use_uno and HAS_UNO:
            self.use_uno = True
            logger.info(f"🚀 DocParser初始化完成 - 使用UNO API进行单线程高效处理")
        else:
            self.use_uno = False
            if use_uno and not HAS_UNO:
                logger.warning(
                    f"⚠️ UNO不可用，回退到传统命令行方式\n"
                    f"💡 提示：UNO转换更快更稳定，强烈建议安装和配置UNO\n"
                    f"📖 请参考上述错误信息中的安装指南"
                )
            else:
                logger.info(f"🚀 DocParser初始化完成 - 使用传统命令行方式")

        logger.info(f"📄 文件路径: {file_path}, 转换为markdown: {to_markdown}")

    def extract_all_content(self, doc_path: str) -> str:
        """
        综合提取DOC文件的所有内容
        支持多种DOC内部格式和存储方式
        """
        logger.info(f"🔍 开始综合内容提取: {doc_path}")

        all_content = []

        try:
            # 1. 尝试使用OLE解析提取内容（如果可用）
            if HAS_OLEFILE:
                ole_content = self._extract_ole_content(doc_path)
                if ole_content:
                    all_content.append(("ole", ole_content))

            # 2. 尝试提取嵌入对象
            embedded_content = self._extract_embedded_objects(doc_path)
            if embedded_content:
                all_content.append(("embedded", embedded_content))

            # 3. 如果上述方法都没有提取到内容，使用传统转换
            if not all_content:
                logger.info("🔄 使用传统转换方式提取内容")
                return ""  # 返回空，让调用者使用传统方式

            # 检查内容质量，特别是对于WPS文件
            for content_type, content in all_content:
                if content and self._check_content_quality(content):
                    logger.info(f"✅ 使用 {content_type} 内容提取成功")
                    return content

            # 如果所有内容质量都不佳，返回空
            logger.warning("⚠️ 所有提取方式的内容质量都不佳")
            return ""

        except Exception as e:
            logger.error(f"💥 综合内容提取失败: {str(e)}")
            return ""

    def _extract_ole_content(self, doc_path: str) -> str:
        """使用OLE解析提取DOC内容"""
        try:
            ole = olefile.OleFileIO(doc_path)
            logger.info(f"📂 成功打开OLE文件: {doc_path}")

            # 列出所有流
            streams = ole.listdir()
            logger.debug(f"📋 可用的OLE流: {streams}")

            # 检查是否是WPS生成的文件
            is_wps = any("WpsCustomData" in str(stream) for stream in streams)
            if is_wps:
                logger.info("📝 检测到WPS DOC文件，建议使用传统转换方式")
                # 对于WPS文件，OLE解析可能不可靠，返回空让其使用传统方式
                ole.close()
                return ""

            all_texts = []

            # 尝试提取WordDocument流
            if ole.exists("WordDocument"):
                try:
                    word_stream = ole.openstream("WordDocument").read()
                    logger.info(f"📄 WordDocument流大小: {len(word_stream)} 字节")
                    text = self._parse_word_stream(word_stream)
                    if text:
                        all_texts.append(text)
                except Exception as e:
                    logger.error(f"💥 解析WordDocument流失败: {str(e)}")

            # 尝试读取其他可能包含文本的流
            text_content = []
            for entry in ole.listdir():
                if any(name in str(entry) for name in ["Text", "Content", "Body"]):
                    try:
                        stream = ole.openstream(entry)
                        data = stream.read()
                        # 尝试解码
                        decoded = self._try_decode_bytes(data)
                        if decoded and len(decoded.strip()) > 10:
                            text_content.append(decoded)
                    except:
                        continue

            if text_content:
                combined = "\n".join(text_content)
                logger.info(f"📄 从OLE流中提取文本: {len(combined)} 字符")
                return self._clean_extracted_text(combined)

            ole.close()

            return ""

        except Exception as e:
            logger.warning(f"⚠️ OLE解析失败: {str(e)}")

        return ""

    def _parse_word_stream(self, data: bytes) -> str:
        """解析WordDocument流中的文本"""
        try:
            # DOC文件格式复杂，这里提供基础的文本提取
            # 查找文本片段
            text_parts = []

            # 尝试多种编码，特别注意中文编码
            for encoding in [
                "utf-16-le",
                "utf-8",
                "gbk",
                "gb18030",
                "gb2312",
                "big5",
                "cp936",
                "cp1252",
            ]:
                try:
                    decoded = data.decode(encoding, errors="ignore")
                    # 检查是否包含合理的中文字符
                    chinese_chars = len(
                        [c for c in decoded if "\u4e00" <= c <= "\u9fff"]
                    )
                    if chinese_chars > 10 or (decoded and len(decoded.strip()) > 50):
                        # 过滤出可打印字符，但保留中文
                        cleaned = self._filter_printable_text(decoded)
                        if cleaned and len(cleaned.strip()) > 20:
                            text_parts.append(cleaned)
                            logger.debug(
                                f"📝 使用编码 {encoding} 成功解码，包含 {chinese_chars} 个中文字符"
                            )
                            break
                except:
                    continue

            return "\n".join(text_parts) if text_parts else ""

        except Exception as e:
            logger.error(f"💥 解析Word流失败: {str(e)}")
            return ""

    def _filter_printable_text(self, text: str) -> str:
        """过滤文本，保留可打印字符和中文"""
        result = []
        for char in text:
            # 保留中文字符
            if "\u4e00" <= char <= "\u9fff":
                result.append(char)
            # 保留日文字符
            elif "\u3040" <= char <= "\u30ff":
                result.append(char)
            # 保留韩文字符
            elif "\uac00" <= char <= "\ud7af":
                result.append(char)
            # 保留ASCII可打印字符和空白字符
            elif char.isprintable() or char.isspace():
                result.append(char)
            # 保留常用标点符号
            elif char in '，。！？；：""' "（）【】《》、·…—":
                result.append(char)

        return "".join(result)

    def _try_decode_bytes(self, data: bytes) -> str:
        """尝试使用多种编码解码字节数据"""
        # 优先尝试中文编码
        encodings = [
            "utf-8",
            "gbk",
            "gb18030",
            "gb2312",
            "big5",
            "utf-16-le",
            "utf-16-be",
            "cp936",
            "cp1252",
            "latin-1",
        ]

        # 首先尝试使用chardet检测编码
        try:
            import chardet

            detected = chardet.detect(data)
            if detected["encoding"] and detected["confidence"] > 0.7:
                encodings.insert(0, detected["encoding"])
                logger.debug(
                    f"🔍 检测到编码: {detected['encoding']} (置信度: {detected['confidence']})"
                )
        except:
            pass

        for encoding in encodings:
            try:
                decoded = data.decode(encoding, errors="ignore")
                # 检查是否包含有意义的文本（包括中文）
                if decoded and (
                    any(c.isalnum() for c in decoded)
                    or any("\u4e00" <= c <= "\u9fff" for c in decoded)
                ):
                    # 进一步清理文本
                    cleaned = self._filter_printable_text(decoded)
                    if cleaned and len(cleaned.strip()) > 10:
                        return cleaned
            except:
                continue

        return ""

    def _extract_embedded_objects(self, doc_path: str) -> str:
        """提取DOC文件中的嵌入对象"""
        try:
            if not HAS_OLEFILE:
                return ""

            embedded_content = []

            with olefile.OleFileIO(doc_path) as ole:
                # 查找嵌入的对象
                for entry in ole.listdir():
                    entry_name = "/".join(entry)

                    # 检查是否是嵌入对象
                    if any(
                        pattern in entry_name.lower()
                        for pattern in ["object", "embed", "package"]
                    ):
                        logger.info(f"📎 找到嵌入对象: {entry_name}")
                        try:
                            stream = ole.openstream(entry)
                            data = stream.read()

                            # 尝试提取文本内容
                            text = self._try_decode_bytes(data)
                            if text and len(text.strip()) > 20:
                                embedded_content.append(text.strip())
                        except:
                            continue

            return "\n\n".join(embedded_content) if embedded_content else ""

        except Exception as e:
            logger.warning(f"⚠️ 提取嵌入对象失败: {str(e)}")
            return ""

    def _clean_extracted_text(self, text: str) -> str:
        """清理提取的文本，彻底移除所有XML标签和控制字符，只保留纯文本"""
        try:
            # 1. 解码HTML/XML实体
            text = html.unescape(text)

            # 2. 移除所有XML/HTML标签
            text = re.sub(r"<[^>]+>", "", text)

            # 3. 移除XML命名空间前缀
            text = re.sub(r"\b\w+:", "", text)

            # 4. 移除NULL字符和其他控制字符
            text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

            # 5. 移除特殊的XML字符序列
            text = re.sub(r"&[a-zA-Z]+;", "", text)
            text = re.sub(r"&#\d+;", "", text)
            text = re.sub(r"&#x[0-9a-fA-F]+;", "", text)

            # 6. 保留有意义的字符，移除其他特殊字符
            # 保留：中文、日文、韩文、英文、数字、常用标点和空白
            allowed_chars = (
                r"\w\s"  # 字母数字和空白
                r"\u4e00-\u9fff"  # 中文
                r"\u3040-\u30ff"  # 日文
                r"\uac00-\ud7af"  # 韩文
                r'，。！？；：""'
                "（）【】《》、·…—"  # 中文标点
                r'.,!?;:()[\]{}"\'`~@#$%^&*+=\-_/\\'  # 英文标点和常用符号
            )

            # 使用更严格的过滤，但保留所有有意义的字符
            cleaned_text = "".join(
                char for char in text if re.match(f"[{allowed_chars}]", char)
            )

            # 7. 移除过长的无意义字符序列（通常是二进制垃圾）
            cleaned_text = re.sub(r"([^\s\u4e00-\u9fff])\1{5,}", r"\1", cleaned_text)

            # 8. 清理多余的空白，但保留段落结构
            cleaned_text = re.sub(
                r"[ \t]+", " ", cleaned_text
            )  # 多个空格/制表符变为单个空格
            cleaned_text = re.sub(
                r"\n\s*\n\s*\n+", "\n\n", cleaned_text
            )  # 多个空行变为双空行
            cleaned_text = re.sub(
                r"^\s+|\s+$", "", cleaned_text, flags=re.MULTILINE
            )  # 移除行首行尾空白

            # 9. 进一步清理：移除独立的标点符号行
            lines = cleaned_text.split("\n")
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line:
                    # 检查行是否主要是有意义的内容
                    # 计算中文、英文字母和数字的比例
                    meaningful_chars = sum(
                        1 for c in line if (c.isalnum() or "\u4e00" <= c <= "\u9fff")
                    )

                    # 如果有意义字符占比超过30%，或者行长度小于5（可能是标题），则保留
                    if len(line) < 5 or (
                        meaningful_chars > 0 and meaningful_chars / len(line) > 0.3
                    ):
                        cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1]:  # 保留段落分隔
                    cleaned_lines.append("")

            result = "\n".join(cleaned_lines).strip()

            # 10. 最终检查
            if len(result) < 10:
                logger.warning("⚠️ 清理后的文本过短，可能存在问题")
                return ""

            # 检查是否还包含XML标签
            if re.search(r"<[^>]+>", result):
                logger.warning("⚠️ 清理后仍包含XML标签，进行二次清理")
                result = re.sub(r"<[^>]+>", "", result)

            return result

        except Exception as e:
            logger.error(f"💥 清理文本失败: {str(e)}")
            return text

    def _combine_extracted_content(self, content_list: list) -> str:
        """合并提取到的各种内容"""
        combined = []

        # 按优先级排序内容
        priority_order = ["ole", "embedded", "converted", "fallback"]

        for content_type in priority_order:
            for item_type, content in content_list:
                if item_type == content_type and content.strip():
                    combined.append(content.strip())

        # 添加其他未分类的内容
        for item_type, content in content_list:
            if item_type not in priority_order and content.strip():
                combined.append(content.strip())

        return "\n\n".join(combined) if combined else ""

    def doc_to_txt(self, doc_path: str, dir_path: str) -> str:
        """将.doc文件转换为.txt文件"""
        logger.info(
            f"🔄 开始转换DOC文件为TXT - 源文件: {doc_path}, 输出目录: {dir_path}"
        )

        if self.use_uno:
            # 使用UNO API进行转换
            try:
                logger.info("🎯 使用UNO API进行文档转换...")
                txt_path = convert_with_uno(doc_path, "txt", dir_path)

                if not os.path.exists(txt_path):
                    logger.error(f"❌ 转换后的TXT文件不存在: {txt_path}")
                    raise Exception(f"文件转换失败 {doc_path} ==> {txt_path}")
                else:
                    logger.info(f"🎉 TXT文件转换成功，文件路径: {txt_path}")
                    return txt_path

            except Exception as e:
                logger.error(
                    f"💥 UNO转换失败: {str(e)}\n"
                    f"🔍 诊断信息：\n"
                    f"   - 错误类型: {type(e).__name__}\n"
                    f"   - LibreOffice是否已安装？尝试运行: soffice --version\n"
                    f"   - Python UNO模块是否可用？尝试: python -c 'import uno'\n"
                    f"   - 是否有其他LibreOffice实例在运行？\n"
                    f"   - 文件权限是否正确？\n"
                    f"🔧 可能的解决方案：\n"
                    f"   1. 确保LibreOffice正确安装\n"
                    f"   2. 关闭所有LibreOffice进程\n"
                    f"   3. 检查文件权限和路径\n"
                    f'   4. 尝试手动运行: soffice --headless --convert-to txt "{doc_path}"'
                )
                logger.warning("⚠️ 自动回退到传统命令行方式...")
                return self._doc_to_txt_subprocess(doc_path, dir_path)
        else:
            # 使用传统的subprocess方式
            return self._doc_to_txt_subprocess(doc_path, dir_path)

    def _doc_to_txt_subprocess(self, doc_path: str, dir_path: str) -> str:
        """使用subprocess将.doc文件转换为.txt文件（传统方式）"""
        try:
            cmd = f'soffice --headless --convert-to txt "{doc_path}" --outdir "{dir_path}"'
            logger.debug(f"⚡ 执行转换命令: {cmd}")

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code == 0:
                logger.info(f"✅ DOC到TXT转换成功 - 退出码: {exit_code}")
                if stdout:
                    logger.debug(
                        f"📄 转换输出: {stdout.decode('utf-8', errors='replace')}"
                    )
            else:
                encoding = chardet.detect(stderr)["encoding"]
                if encoding is None:
                    encoding = "utf-8"
                error_msg = stderr.decode(encoding, errors="replace")
                logger.error(
                    f"❌ DOC到TXT转换失败 - 退出码: {exit_code}, 错误信息: {error_msg}"
                )
                raise Exception(
                    f"Error Output (detected encoding: {encoding}): {error_msg}"
                )

            fname = str(Path(doc_path).stem)
            txt_path = os.path.join(dir_path, f"{fname}.txt")

            if not os.path.exists(txt_path):
                logger.error(f"❌ 转换后的TXT文件不存在: {txt_path}")
                raise Exception(f"文件转换失败 {doc_path} ==> {txt_path}")
            else:
                logger.info(f"🎉 TXT文件转换成功，文件路径: {txt_path}")
                return txt_path

        except subprocess.SubprocessError as e:
            logger.error(f"💥 subprocess执行失败: {str(e)}")
            raise Exception(f"执行转换命令时发生错误: {str(e)}")
        except Exception as e:
            logger.error(f"💥 DOC到TXT转换过程中发生未知错误: {str(e)}")
            raise

    def read_txt_file(self, txt_path: str) -> str:
        """读取txt文件内容"""
        logger.info(f"📖 开始读取TXT文件: {txt_path}")

        try:
            # 检测文件编码
            with open(txt_path, "rb") as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"]
                if encoding is None:
                    encoding = "utf-8"
                logger.debug(f"🔍 检测到文件编码: {encoding}")

            # 读取文件内容
            with open(txt_path, "r", encoding=encoding, errors="replace") as f:
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

    def read_doc_file(self, doc_path: str) -> str:
        """读取doc文件并转换为文本"""
        logger.info(f"📖 开始读取DOC文件 - 文件: {doc_path}")

        try:
            # 首先尝试综合提取（如果有高级解析功能）
            if HAS_OLEFILE:
                comprehensive_content = self.extract_all_content(doc_path)
                if comprehensive_content and comprehensive_content.strip():
                    # 检查内容质量
                    if self._check_content_quality(comprehensive_content):
                        logger.info(
                            f"✨ 使用综合提取方式成功，内容长度: {len(comprehensive_content)} 字符"
                        )
                        return comprehensive_content
                    else:
                        logger.warning("⚠️ 综合提取的内容质量不佳，尝试其他方式")

            # 降级到传统转换方式
            logger.info("🔄 使用传统转换方式")

            with tempfile.TemporaryDirectory() as temp_path:
                logger.debug(f"📁 创建临时目录: {temp_path}")

                temp_dir = Path(temp_path)

                file_path = temp_dir / "tmp.doc"
                shutil.copy(doc_path, file_path)
                logger.debug(f"📋 复制文件到临时目录: {doc_path} -> {file_path}")

                # 转换DOC为TXT
                txt_file_path = self.doc_to_txt(str(file_path), str(temp_path))
                logger.info(f"🎯 DOC转TXT完成: {txt_file_path}")

                # 读取TXT文件内容
                content = self.read_txt_file(txt_file_path)
                logger.info(f"✨ TXT文件内容读取完成，内容长度: {len(content)} 字符")

                return content

        except FileNotFoundError as e:
            logger.error(f"🚫 文件未找到: {str(e)}")
            raise Exception(f"文件未找到: {doc_path}")
        except PermissionError as e:
            logger.error(f"🔒 文件权限错误: {str(e)}")
            raise Exception(f"无权限访问文件: {doc_path}")
        except Exception as e:
            logger.error(f"💥 读取DOC文件时发生错误: {str(e)}")
            raise

    def _check_content_quality(self, content: str) -> bool:
        """检查提取内容的质量"""
        if not content or len(content) < 50:
            return False

        # 计算乱码字符比例
        total_chars = len(content)
        # 可识别字符：ASCII、中文、日文、韩文、常用标点
        recognizable = sum(
            1
            for c in content
            if (
                c.isascii()
                or "\u4e00" <= c <= "\u9fff"  # 中文
                or "\u3040" <= c <= "\u30ff"  # 日文
                or "\uac00" <= c <= "\ud7af"  # 韩文
                or c in '，。！？；：""' "（）【】《》、·…—\n\r\t "
            )
        )

        # 如果可识别字符占比低于70%，认为质量不佳
        if recognizable / total_chars < 0.7:
            logger.warning(
                f"⚠️ 内容质量检查失败：可识别字符比例 {recognizable}/{total_chars} = {recognizable/total_chars:.2%}"
            )
            return False

        return True

    def parse(self, file_path: str):
        """解析DOC文件"""
        logger.info(f"🎬 开始解析DOC文件: {file_path}")

        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                logger.error(f"🚫 文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 验证文件扩展名
            if not file_path.lower().endswith(".doc"):
                logger.warning(f"⚠️ 文件扩展名不是.doc: {file_path}")

            # 验证文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 文件大小: {file_size} 字节")

            if file_size == 0:
                logger.warning(f"⚠️ 文件大小为0字节: {file_path}")
            # 生命周期：Data Processing 开始
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Documentation",
            )

            # 🏷️ 提取文件扩展名
            extension = self.get_file_extension(file_path)
            logger.debug(f"🏷️ 提取文件扩展名: {extension}")

            # 读取文件内容
            logger.info("📝 读取DOC文件内容")
            content = self.read_doc_file(doc_path=file_path)

            # 根据to_markdown参数决定是否保持原格式还是处理为markdown格式
            if self.to_markdown:
                # 简单的文本到markdown转换（保持段落结构）
                mk_content = self.format_as_markdown(content)
                logger.info("🎨 内容已格式化为markdown格式")
            else:
                mk_content = content
                logger.info("📝 保持原始文本格式")
            # 3) 生命周期：Data Processed or Failed
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if mk_content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Documentation",
            )

            logger.info(f"🎊 文件内容解析完成，最终内容长度: {len(mk_content)} 字符")

            # 检查内容是否为空
            if not mk_content.strip():
                logger.warning(f"⚠️ 解析出的内容为空: {file_path}")

            lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type="LLM_ORIGIN",
            )
            logger.debug("⚙️ 生成lifecycle信息完成")

            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            # output_vo.add_lifecycle(lc_origin)

            result = output_vo.to_dict()
            logger.info(f"🏆 DOC文件解析完成: {file_path}")
            logger.debug(f"🔑 返回结果键: {list(result.keys())}")

            return result

        except FileNotFoundError as e:
            logger.error(f"🚫 文件不存在错误: {str(e)}")
            raise
        except PermissionError as e:
            logger.error(f"🔒 文件权限错误: {str(e)}")
            raise Exception(f"无权限访问文件: {file_path}")
        except Exception as e:
            logger.error(
                f"💀 解析DOC文件失败: {file_path}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}"
            )
            raise

    def format_as_markdown(self, content: str) -> str:
        """将纯文本格式化为简单的markdown格式"""
        if not content.strip():
            return content

        lines = content.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue

            # 简单的markdown格式化规则
            # 可以根据需要扩展更多规则
            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _extract_text_from_wps_stream(self, data: bytes) -> str:
        """从WPS的WordDocument流中提取文本（使用更宽松的策略）"""
        try:
            text_parts = []

            # WPS文件可能使用不同的编码和结构
            # 尝试多种策略提取文本

            # 策略1：尝试找到连续的文本块
            # 查找看起来像文本的字节序列
            i = 0
            while i < len(data):
                # 查找可能的文本开始位置
                if i + 2 < len(data):
                    # 检查是否是Unicode文本（小端序）
                    if data[i + 1] == 0 and 32 <= data[i] <= 126:
                        # 可能是ASCII字符的Unicode编码
                        text_block = bytearray()
                        j = i
                        while (
                            j + 1 < len(data)
                            and data[j + 1] == 0
                            and 32 <= data[j] <= 126
                        ):
                            text_block.append(data[j])
                            j += 2
                        if len(text_block) > 10:
                            text_parts.append(
                                text_block.decode("ascii", errors="ignore")
                            )
                        i = j
                    # 检查是否是UTF-8或GBK中文
                    elif 0xE0 <= data[i] <= 0xEF or 0x81 <= data[i] <= 0xFE:
                        # 可能是多字节字符
                        text_block = bytearray()
                        j = i
                        while j < len(data):
                            if data[j] < 32 and data[j] not in [9, 10, 13]:
                                break
                            text_block.append(data[j])
                            j += 1
                        if len(text_block) > 20:
                            # 尝试解码
                            for encoding in ["utf-8", "gbk", "gb18030", "gb2312"]:
                                try:
                                    decoded = text_block.decode(
                                        encoding, errors="ignore"
                                    )
                                    if decoded and len(decoded.strip()) > 10:
                                        text_parts.append(decoded)
                                        break
                                except:
                                    continue
                        i = j
                    else:
                        i += 1
                else:
                    i += 1

            # 合并文本部分
            if text_parts:
                combined = "\n".join(text_parts)
                return self._clean_extracted_text(combined)

            # 如果上述方法失败，回退到原始方法
            return self._parse_word_stream(data)

        except Exception as e:
            logger.error(f"💥 解析WPS流失败: {str(e)}")
            return ""
