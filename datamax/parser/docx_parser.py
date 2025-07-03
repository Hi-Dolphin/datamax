import html
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union

import chardet
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType

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


class DocxParser(BaseLife):
    def __init__(
        self,
        file_path: Union[str, list],
        to_markdown: bool = False,
        use_uno: bool = True,
        domain: str = "Technology",
    ):
        super().__init__(domain=domain)
        self.file_path = file_path
        self.to_markdown = to_markdown

        # 优先使用UNO（除非明确禁用）
        if use_uno and HAS_UNO:
            self.use_uno = True
            logger.info(f"🚀 DocxParser初始化完成 - 使用UNO API进行单线程高效处理")
        else:
            self.use_uno = False
            if use_uno and not HAS_UNO:
                logger.warning(
                    f"⚠️ UNO不可用，回退到传统命令行方式\n"
                    f"💡 提示：UNO转换更快更稳定，强烈建议安装和配置UNO\n"
                    f"📖 请参考上述错误信息中的安装指南"
                )
            else:
                logger.info(f"🚀 DocxParser初始化完成 - 使用传统命令行方式")

        logger.info(f"📄 文件路径: {file_path}, 转换为markdown: {to_markdown}")

    def docx_to_txt(self, docx_path: str, dir_path: str) -> str:
        """将.docx文件转换为.txt文件"""
        logger.info(
            f"🔄 开始转换DOCX文件为TXT - 源文件: {docx_path}, 输出目录: {dir_path}"
        )

        if self.use_uno:
            # 使用UNO API进行转换
            try:
                logger.info("🎯 使用UNO API进行文档转换...")
                txt_path = convert_with_uno(docx_path, "txt", dir_path)

                if not os.path.exists(txt_path):
                    logger.error(f"❌ 转换后的TXT文件不存在: {txt_path}")
                    raise Exception(f"文件转换失败 {docx_path} ==> {txt_path}")
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
                    f'   4. 尝试手动运行: soffice --headless --convert-to txt "{docx_path}"'
                )
                logger.warning("⚠️ 自动回退到传统命令行方式...")
                return self._docx_to_txt_subprocess(docx_path, dir_path)
        else:
            # 使用传统的subprocess方式
            return self._docx_to_txt_subprocess(docx_path, dir_path)

    def _docx_to_txt_subprocess(self, docx_path: str, dir_path: str) -> str:
        """使用subprocess将.docx文件转换为.txt文件（传统方式）"""
        try:
            cmd = f'soffice --headless --convert-to txt "{docx_path}" --outdir "{dir_path}"'
            logger.debug(f"⚡ 执行转换命令: {cmd}")

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code == 0:
                logger.info(f"✅ DOCX到TXT转换成功 - 退出码: {exit_code}")
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
                    f"❌ DOCX到TXT转换失败 - 退出码: {exit_code}, 错误信息: {error_msg}"
                )
                raise Exception(
                    f"Error Output (detected encoding: {encoding}): {error_msg}"
                )

            fname = str(Path(docx_path).stem)
            txt_path = os.path.join(dir_path, f"{fname}.txt")

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

    def extract_all_content(self, docx_path: str) -> str:
        """
        综合提取DOCX文件的所有内容
        支持多种DOCX内部格式和存储方式
        """
        logger.info(f"🔍 开始综合内容提取: {docx_path}")

        all_content = []

        try:
            with zipfile.ZipFile(docx_path, "r") as docx:
                # 1. 检查并提取altChunk内容 (HTML/MHT嵌入)
                altchunk_content = self._extract_altchunk_content_internal(docx)
                if altchunk_content:
                    all_content.append(("altChunk", altchunk_content))

                # 2. 提取标准document.xml内容
                standard_content = self._extract_standard_document_content(docx)
                if standard_content:
                    all_content.append(("standard", standard_content))

                # 3. 提取嵌入对象内容 (embeddings)
                embedded_content = self._extract_embedded_objects(docx)
                if embedded_content:
                    all_content.append(("embedded", embedded_content))

                # 4. 提取头部和脚部内容
                header_footer_content = self._extract_headers_footers(docx)
                if header_footer_content:
                    all_content.append(("header_footer", header_footer_content))

                # 5. 提取注释和批注
                comments_content = self._extract_comments(docx)
                if comments_content:
                    all_content.append(("comments", comments_content))

                # 6. 提取文本框和图形对象中的文本
                textbox_content = self._extract_textbox_content(docx)
                if textbox_content:
                    all_content.append(("textboxes", textbox_content))

        except Exception as e:
            logger.error(f"💥 综合内容提取失败: {str(e)}")
            return ""

        # 合并所有内容
        if all_content:
            combined_content = self._combine_extracted_content(all_content)
            logger.info(f"✅ 综合提取完成，总内容长度: {len(combined_content)} 字符")
            logger.debug(f"📊 提取到的内容类型: {[item[0] for item in all_content]}")
            return combined_content

        return ""

    def _extract_altchunk_content_internal(self, docx_zip: zipfile.ZipFile) -> str:
        """内部方法：提取altChunk内容，优先使用MHT方式"""
        try:
            # 检查document.xml中的altChunk引用
            if "word/document.xml" in docx_zip.namelist():
                doc_xml = docx_zip.read("word/document.xml").decode(
                    "utf-8", errors="replace"
                )
                if "altChunk" in doc_xml:
                    logger.info("🔍 检测到altChunk格式")

                    # 优先查找MHT文件（更简洁的处理方式）
                    mht_files = [
                        f
                        for f in docx_zip.namelist()
                        if f.endswith(".mht") and "word/" in f
                    ]
                    html_files = [
                        f
                        for f in docx_zip.namelist()
                        if f.endswith(".html") and "word/" in f
                    ]

                    # 优先处理MHT文件
                    for filename in mht_files:
                        logger.info(f"📄 优先处理MHT文件: {filename}")
                        content = docx_zip.read(filename).decode(
                            "utf-8", errors="replace"
                        )
                        return self._extract_html_from_mht(content)

                    # 如果没有MHT文件，再处理HTML文件
                    for filename in html_files:
                        logger.info(f"📄 处理HTML文件: {filename}")
                        content = docx_zip.read(filename).decode(
                            "utf-8", errors="replace"
                        )
                        return self._html_to_clean_text(content)

            return ""
        except Exception as e:
            logger.error(f"💥 提取altChunk内容失败: {str(e)}")
            return ""

    def _extract_standard_document_content(self, docx_zip: zipfile.ZipFile) -> str:
        """提取标准document.xml内容 - 只提取纯文本"""
        try:
            if "word/document.xml" in docx_zip.namelist():
                doc_xml = docx_zip.read("word/document.xml").decode(
                    "utf-8", errors="replace"
                )

                # 解码XML实体
                doc_xml = html.unescape(doc_xml)

                # 提取所有<w:t>标签中的文本（包括各种命名空间前缀）
                # 使用更宽松的正则表达式来匹配任何命名空间前缀
                text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                text_matches = re.findall(text_pattern, doc_xml)

                # 额外提取可能存在的无命名空间的<t>标签
                text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", doc_xml))

                if text_matches:
                    # 清理和组合文本
                    cleaned_texts = []
                    for text in text_matches:
                        # 解码XML实体
                        text = html.unescape(text)
                        # 移除多余的空白字符，但保留单个空格
                        text = re.sub(r"\s+", " ", text.strip())
                        if text:
                            cleaned_texts.append(text)

                    # 智能连接文本片段
                    content = ""
                    for i, text in enumerate(cleaned_texts):
                        if i == 0:
                            content = text
                        else:
                            # 如果前一个文本片段不是以标点结束，且当前文本不是以大写开头，则不加空格
                            prev_char = content[-1] if content else ""
                            curr_char = text[0] if text else ""

                            if (
                                prev_char in ".!?。！？\n"
                                or curr_char.isupper()
                                or curr_char in "，。！？；："
                            ):
                                content += " " + text
                            else:
                                content += text

                    # 最终清理
                    content = re.sub(r"\s+", " ", content)
                    content = content.strip()

                    logger.info(f"📝 从document.xml提取纯文本: {len(content)} 字符")
                    return content
            return ""
        except Exception as e:
            logger.error(f"💥 提取标准文档内容失败: {str(e)}")
            return ""

    def _extract_embedded_objects(self, docx_zip: zipfile.ZipFile) -> str:
        """提取嵌入对象内容"""
        try:
            embedded_content = []

            # 查找嵌入的文档对象
            for filename in docx_zip.namelist():
                if "word/embeddings/" in filename:
                    logger.info(f"📎 找到嵌入对象: {filename}")
                    # 这里可以根据文件类型进一步处理
                    # 例如：.docx, .xlsx, .txt等

            return " ".join(embedded_content) if embedded_content else ""
        except Exception as e:
            logger.error(f"💥 提取嵌入对象失败: {str(e)}")
            return ""

    def _extract_headers_footers(self, docx_zip: zipfile.ZipFile) -> str:
        """提取页眉页脚内容 - 只提取纯文本"""
        try:
            header_footer_content = []

            for filename in docx_zip.namelist():
                if (
                    "word/header" in filename or "word/footer" in filename
                ) and filename.endswith(".xml"):
                    logger.debug(f"📄 处理页眉页脚: {filename}")
                    content = docx_zip.read(filename).decode("utf-8", errors="replace")

                    # 解码XML实体
                    content = html.unescape(content)

                    # 提取文本内容 - 使用更宽松的模式
                    text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                    text_matches = re.findall(text_pattern, content)
                    text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", content))

                    if text_matches:
                        # 清理和组合文本
                        cleaned_texts = []
                        for text in text_matches:
                            text = html.unescape(text)
                            text = re.sub(r"\s+", " ", text.strip())
                            if text:
                                cleaned_texts.append(text)

                        if cleaned_texts:
                            # 合并文本片段
                            header_footer_text = " ".join(cleaned_texts)
                            header_footer_text = re.sub(
                                r"\s+", " ", header_footer_text.strip()
                            )
                            if header_footer_text:
                                header_footer_content.append(header_footer_text)

            if header_footer_content:
                logger.info(f"📑 提取页眉页脚纯文本: {len(header_footer_content)} 个")

            return "\n".join(header_footer_content) if header_footer_content else ""
        except Exception as e:
            logger.error(f"💥 提取页眉页脚失败: {str(e)}")
            return ""

    def _extract_comments(self, docx_zip: zipfile.ZipFile) -> str:
        """提取注释和批注内容 - 只提取纯文本"""
        try:
            if "word/comments.xml" in docx_zip.namelist():
                comments_xml = docx_zip.read("word/comments.xml").decode(
                    "utf-8", errors="replace"
                )

                # 解码XML实体
                comments_xml = html.unescape(comments_xml)

                # 提取注释文本 - 使用更宽松的模式
                text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                text_matches = re.findall(text_pattern, comments_xml)
                text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", comments_xml))

                if text_matches:
                    # 清理和组合文本
                    cleaned_texts = []
                    for text in text_matches:
                        text = html.unescape(text)
                        text = re.sub(r"\s+", " ", text.strip())
                        if text:
                            cleaned_texts.append(text)

                    if cleaned_texts:
                        comments_text = " ".join(cleaned_texts)
                        comments_text = re.sub(r"\s+", " ", comments_text.strip())
                        logger.info(f"💬 提取注释纯文本: {len(comments_text)} 字符")
                        return comments_text

            return ""
        except Exception as e:
            logger.error(f"💥 提取注释失败: {str(e)}")
            return ""

    def _extract_textbox_content(self, docx_zip: zipfile.ZipFile) -> str:
        """提取文本框和图形对象中的文本 - 只提取纯文本"""
        try:
            textbox_content = []

            # 查找可能包含文本框的文件
            for filename in docx_zip.namelist():
                if "word/" in filename and filename.endswith(".xml"):
                    content = docx_zip.read(filename).decode("utf-8", errors="replace")

                    # 解码XML实体
                    content = html.unescape(content)

                    # 查找文本框内容 (w:txbxContent)
                    textbox_matches = re.findall(
                        r"<[^:>]*:txbxContent[^>]*>(.*?)</[^:>]*:txbxContent>",
                        content,
                        re.DOTALL,
                    )

                    for match in textbox_matches:
                        # 从文本框内容中提取文本
                        text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                        text_matches = re.findall(text_pattern, match)
                        text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", match))

                        if text_matches:
                            # 清理和组合文本
                            cleaned_texts = []
                            for text in text_matches:
                                text = html.unescape(text)
                                text = re.sub(r"\s+", " ", text.strip())
                                if text:
                                    cleaned_texts.append(text)

                            if cleaned_texts:
                                textbox_text = " ".join(cleaned_texts)
                                textbox_text = re.sub(r"\s+", " ", textbox_text.strip())
                                if textbox_text:
                                    textbox_content.append(textbox_text)

            if textbox_content:
                logger.info(f"📦 提取文本框纯文本: {len(textbox_content)} 个")

            return "\n".join(textbox_content) if textbox_content else ""
        except Exception as e:
            logger.error(f"💥 提取文本框内容失败: {str(e)}")
            return ""

    def _combine_extracted_content(self, content_list: list) -> str:
        """合并提取到的各种内容 - 输出清晰的纯文本"""
        combined = []

        # 按重要性排序内容
        priority_order = [
            "altChunk",
            "standard",
            "header_footer",
            "textboxes",
            "comments",
            "embedded",
        ]

        for content_type in priority_order:
            for item_type, content in content_list:
                if item_type == content_type and content.strip():
                    # 清理内容中的多余空白
                    cleaned_content = re.sub(r"\s+", " ", content.strip())
                    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                    if cleaned_content:
                        # 根据内容类型添加简单的标记（仅在有多种内容类型时）
                        if len([1 for t, c in content_list if c.strip()]) > 1:
                            if item_type == "header_footer":
                                combined.append(f"[页眉页脚]\n{cleaned_content}")
                            elif item_type == "comments":
                                combined.append(f"[批注]\n{cleaned_content}")
                            elif item_type == "textboxes":
                                combined.append(f"[文本框]\n{cleaned_content}")
                            else:
                                combined.append(cleaned_content)
                        else:
                            combined.append(cleaned_content)

        # 添加其他未分类的内容
        for item_type, content in content_list:
            if item_type not in priority_order and content.strip():
                cleaned_content = re.sub(r"\s+", " ", content.strip())
                cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)
                if cleaned_content:
                    combined.append(cleaned_content)

        # 合并所有内容，使用双换行分隔不同部分
        final_content = "\n\n".join(combined) if combined else ""

        # 最终清理：确保没有过多的空行
        final_content = re.sub(r"\n{3,}", "\n\n", final_content)
        final_content = final_content.strip()

        return final_content

    def _extract_html_from_mht(self, mht_content: str) -> str:
        """从MHT内容中提取HTML部分并转换为简洁文本"""
        try:
            # MHT文件使用MIME格式，寻找HTML部分
            lines = mht_content.split("\n")
            in_html_section = False
            html_lines = []
            skip_headers = True

            for line in lines:
                # 检测HTML部分开始
                if "Content-Type: text/html" in line:
                    in_html_section = True
                    skip_headers = True
                    continue

                # 在HTML部分中
                if in_html_section:
                    # 跳过Content-*头部
                    if (
                        skip_headers
                        and line.strip()
                        and not line.startswith("Content-")
                    ):
                        skip_headers = False

                    # 空行表示头部结束，内容开始
                    if skip_headers and not line.strip():
                        skip_headers = False
                        continue

                    # 检查是否到达下一个MIME部分
                    if line.startswith("------=") and len(html_lines) > 0:
                        # HTML部分结束
                        break

                    # 收集HTML内容
                    if not skip_headers:
                        html_lines.append(line)

            # 合并所有HTML行
            html_content = "\n".join(html_lines)

            # 解码quoted-printable编码
            if "=3D" in html_content or "=\n" in html_content:
                try:
                    import quopri

                    html_content = quopri.decodestring(html_content.encode()).decode(
                        "utf-8", errors="replace"
                    )
                    logger.info("📧 解码quoted-printable编码")
                except Exception as e:
                    logger.warning(f"⚠️ quoted-printable解码失败: {str(e)}")

            logger.debug(f"📄 提取的HTML内容长度: {len(html_content)} 字符")

            # 转换为简洁文本
            return self._html_to_clean_text(html_content)

        except Exception as e:
            logger.error(f"💥 从MHT提取HTML失败: {str(e)}")
            return ""

    def _html_to_clean_text(self, html_content: str) -> str:
        """将HTML内容转换为简洁的纯文本，专门优化MHT内容"""
        try:
            # 首先解码HTML实体
            text = html.unescape(html_content)

            # 先尝试提取<body>标签内的所有内容
            body_match = re.search(
                r"<body[^>]*>(.*?)</body>", text, re.DOTALL | re.IGNORECASE
            )
            if body_match:
                main_content = body_match.group(1)
                logger.info("📄 提取<body>标签内容")
            else:
                main_content = text
                logger.info("📄 使用全部内容（未找到body标签）")

            # 特殊处理<pre><code>标签，保持其内部的格式
            pre_code_blocks = []

            def preserve_pre_code(match):
                idx = len(pre_code_blocks)
                pre_code_blocks.append(match.group(1))
                return f"__PRE_CODE_{idx}__"

            main_content = re.sub(
                r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>",
                preserve_pre_code,
                main_content,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # 处理其他HTML结构
            # 1. 先转换需要保留换行的标签
            main_content = re.sub(r"<br\s*/?>", "\n", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"</p>", "\n", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"<p[^>]*>", "", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"</div>", "\n", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"<div[^>]*>", "", main_content, flags=re.IGNORECASE)
            main_content = re.sub(
                r"</h[1-6]>", "\n\n", main_content, flags=re.IGNORECASE
            )
            main_content = re.sub(
                r"<h[1-6][^>]*>", "", main_content, flags=re.IGNORECASE
            )
            main_content = re.sub(r"</li>", "\n", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"<li[^>]*>", "• ", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"</tr>", "\n", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"</td>", " | ", main_content, flags=re.IGNORECASE)
            main_content = re.sub(r"</th>", " | ", main_content, flags=re.IGNORECASE)

            # 2. 移除style和script标签及其内容
            main_content = re.sub(
                r"<style[^>]*>.*?</style>",
                "",
                main_content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            main_content = re.sub(
                r"<script[^>]*>.*?</script>",
                "",
                main_content,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # 3. 移除所有剩余的HTML标签
            main_content = re.sub(r"<[^>]+>", "", main_content)

            # 4. 解码HTML实体（第二次，确保完全解码）
            main_content = html.unescape(main_content)

            # 5. 恢复<pre><code>块的内容
            for idx, pre_code_content in enumerate(pre_code_blocks):
                # 清理pre_code内容
                cleaned_pre_code = html.unescape(pre_code_content)
                main_content = main_content.replace(
                    f"__PRE_CODE_{idx}__", cleaned_pre_code
                )

            # 6. 清理多余的空白字符，但保持段落结构
            lines = main_content.split("\n")
            cleaned_lines = []

            for line in lines:
                # 清理每行的首尾空格
                line = line.strip()
                # 保留非空行
                if line:
                    # 清理行内多余空格
                    line = re.sub(r"[ \t]+", " ", line)
                    # 清理表格分隔符多余的空格
                    line = re.sub(r"\s*\|\s*", " | ", line)
                    cleaned_lines.append(line)
                else:
                    # 保留空行作为段落分隔
                    if cleaned_lines and cleaned_lines[-1] != "":
                        cleaned_lines.append("")

            # 7. 合并清理后的行
            main_content = "\n".join(cleaned_lines)

            # 8. 最终清理：移除多余的空行
            main_content = re.sub(r"\n{3,}", "\n\n", main_content)
            main_content = main_content.strip()

            logger.info(f"📝 HTML内容转换为简洁文本: {len(main_content)} 字符")

            return main_content

        except Exception as e:
            logger.error(f"💥 HTML转简洁文本失败: {str(e)}")
            # 如果转换失败，返回原始文本的基础清理版本
            return re.sub(r"<[^>]+>", "", html_content)

    def _html_to_text(self, html_content: str) -> str:
        """将HTML内容转换为纯文本（保留此方法用于其他HTML内容）"""
        # 对于非MHT的HTML内容，使用这个更通用的方法
        return self._html_to_clean_text(html_content)

    def extract_altchunk_content(self, docx_path: str) -> Optional[str]:
        """
        提取包含altChunk的DOCX文件内容 (保持向后兼容)
        """
        try:
            with zipfile.ZipFile(docx_path, "r") as docx:
                return self._extract_altchunk_content_internal(docx)
        except Exception as e:
            logger.error(f"💥 提取altChunk内容失败: {str(e)}")
            return None

    def read_docx_file(self, docx_path: str) -> str:
        """读取docx文件并转换为文本"""
        logger.info(f"📖 开始读取DOCX文件 - 文件: {docx_path}")

        try:
            # 首先尝试综合提取所有内容
            comprehensive_content = self.extract_all_content(docx_path)
            if comprehensive_content and comprehensive_content.strip():
                logger.info(
                    f"✨ 使用综合提取方式成功，内容长度: {len(comprehensive_content)} 字符"
                )
                return comprehensive_content

            # 如果综合提取失败，使用传统转换方式
            logger.info("🔄 综合提取失败或内容为空，使用传统转换方式")

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
            if not file_path.lower().endswith(".docx"):
                logger.warning(f"⚠️ 文件扩展名不是.docx: {file_path}")

            # 验证文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 文件大小: {file_size} 字节")

            if file_size == 0:
                logger.warning(f"⚠️ 文件大小为0字节: {file_path}")

            # 🏷️ 提取文件扩展名
            extension = self.get_file_extension(file_path)
            logger.debug(f"🏷️ 提取文件扩展名: {extension}")
            # 1) 处理开始：生成 DATA_PROCESSING 事件
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )
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

            # 2) 处理结束：根据内容是否非空生成 DATA_PROCESSED 或 DATA_PROCESS_FAILED 事件
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=(
                    LifeType.DATA_PROCESSED
                    if mk_content.strip()
                    else LifeType.DATA_PROCESS_FAILED
                ),
                usage_purpose="Parsing",
            )
            logger.debug("⚙️ 生成生命周期事件完成")

            # 3) 封装输出并添加生命周期
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)

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
            logger.error(
                f"💀 解析DOCX文件失败: {file_path}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}"
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
