import html
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import chardet
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType

# Try to import UNO processor
try:
    from datamax.utils.uno_handler import HAS_UNO, convert_with_uno
except ImportError:
    HAS_UNO = False
    logger.error(
        "❌ UNO processor import failed!\n"
        "🔧 Solutions:\n"
        "1. Install LibreOffice and python-uno:\n"
        "   - Ubuntu/Debian: sudo apt-get install libreoffice python3-uno\n"
        "   - CentOS/RHEL: sudo yum install libreoffice python3-uno\n"
        "   - macOS: brew install libreoffice\n"
        "   - Windows: Download and install LibreOffice\n"
        "2. Ensure Python can access uno module:\n"
        "   - Linux: export PYTHONPATH=/usr/lib/libreoffice/program:$PYTHONPATH\n"
        "   - Windows: Add LibreOffice\\program to system PATH\n"
        "3. Verify installation: python -c 'import uno'\n"
        "4. If issues persist, check complete documentation:\n"
        "   https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
    )


class DocxParser(BaseLife):
    def __init__(
        self,
        file_path: str | list,
        to_markdown: bool = False,
        use_uno: bool = True,
        domain: str = "Technology",
    ):
        super().__init__(domain=domain)
        self.file_path = file_path
        self.to_markdown = to_markdown

        # Prioritize UNO usage (unless explicitly disabled)
        if use_uno and HAS_UNO:
            self.use_uno = True
            logger.info(f"🚀 DocxParser initialized - Using UNO API for single-threaded efficient processing")
        else:
            self.use_uno = False
            if use_uno and not HAS_UNO:
                logger.warning(
                    f"⚠️ UNO unavailable, falling back to traditional command line method\n"
                    f"💡 Tip: UNO conversion is faster and more stable, strongly recommend installing and configuring UNO\n"
                    f"📖 Please refer to the installation guide in the error message above"
                )
            else:
                logger.info(f"🚀 DocxParser initialized - Using traditional command line method")

        logger.info(f"📄 File path: {file_path}, Convert to markdown: {to_markdown}")

    def docx_to_txt(self, docx_path: str, dir_path: str) -> str:
        """Convert .docx file to .txt file"""
        logger.info(
            f"🔄 Starting DOCX to TXT conversion - Source file: {docx_path}, Output directory: {dir_path}"
        )

        if self.use_uno:
            # Use UNO API for conversion
            try:
                logger.info("🎯 Using UNO API for document conversion...")
                txt_path = convert_with_uno(docx_path, "txt", dir_path)

                if not os.path.exists(txt_path):
                    logger.error(f"❌ Converted TXT file does not exist: {txt_path}")
                    raise Exception(f"File conversion failed {docx_path} ==> {txt_path}")
                else:
                    logger.info(f"🎉 TXT file conversion successful, file path: {txt_path}")
                    return txt_path

            except Exception as e:
                logger.error(
                    f"💥 UNO conversion failed: {str(e)}\n"
                    f"🔍 Diagnostic information:\n"
                    f"   - Error type: {type(e).__name__}\n"
                    f"   - Is LibreOffice installed? Try running: soffice --version\n"
                    f"   - Is Python UNO module available? Try: python -c 'import uno'\n"
                    f"   - Are there other LibreOffice instances running?\n"
                    f"   - Are file permissions correct?\n"
                    f"🔧 Possible solutions:\n"
                    f"   1. Ensure LibreOffice is correctly installed\n"
                    f"   2. Close all LibreOffice processes\n"
                    f"   3. Check file permissions and paths\n"
                    f'   4. Try manual execution: soffice --headless --convert-to txt "{docx_path}"'
                )
                logger.warning("⚠️ Automatically falling back to traditional command line method...")
                return self._docx_to_txt_subprocess(docx_path, dir_path)
        else:
            # Use traditional subprocess method
            return self._docx_to_txt_subprocess(docx_path, dir_path)

    def _docx_to_txt_subprocess(self, docx_path: str, dir_path: str) -> str:
        """Convert .docx file to .txt file using subprocess (traditional method)"""
        try:
            cmd = f'soffice --headless --convert-to txt "{docx_path}" --outdir "{dir_path}"'
            logger.debug(f"⚡ Executing conversion command: {cmd}")

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code == 0:
                logger.info(f"✅ DOCX to TXT conversion successful - Exit code: {exit_code}")
                if stdout:
                    logger.debug(
                        f"📄 Conversion output: {stdout.decode('utf-8', errors='replace')}"
                    )
            else:
                encoding = chardet.detect(stderr)["encoding"]
                if encoding is None:
                    encoding = "utf-8"
                error_msg = stderr.decode(encoding, errors="replace")
                logger.error(
                    f"❌ DOCX to TXT conversion failed - Exit code: {exit_code}, Error message: {error_msg}"
                )
                raise Exception(
                    f"Error Output (detected encoding: {encoding}): {error_msg}"
                )

            fname = str(Path(docx_path).stem)
            txt_path = os.path.join(dir_path, f"{fname}.txt")

            if not os.path.exists(txt_path):
                logger.error(f"❌ Converted TXT file does not exist: {txt_path}")
                raise Exception(f"File conversion failed {docx_path} ==> {txt_path}")
            else:
                logger.info(f"🎉 TXT file conversion successful, file path: {txt_path}")
                return txt_path

        except subprocess.SubprocessError as e:
            logger.error(f"💥 Subprocess execution failed: {str(e)}")
            raise Exception(f"Error occurred while executing conversion command: {str(e)}")
        except Exception as e:
            logger.error(f"💥 Unknown error occurred during DOCX to TXT conversion: {str(e)}")
            raise

    def read_txt_file(self, txt_path: str) -> str:
        """Read txt file content"""
        logger.info(f"📖 Starting to read TXT file: {txt_path}")

        try:
            # Detect file encoding
            with open(txt_path, "rb") as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"]
                if encoding is None:
                    encoding = "utf-8"
                logger.debug(f"🔍 Detected file encoding: {encoding}")

            # Read file content
            with open(txt_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()

            logger.info(f"📄 TXT file reading completed - Content length: {len(content)} characters")
            logger.debug(f"👀 First 100 characters preview: {content[:100]}...")

            return content

        except FileNotFoundError as e:
            logger.error(f"🚫 TXT file not found: {str(e)}")
            raise Exception(f"File not found: {txt_path}")
        except Exception as e:
            logger.error(f"💥 Error occurred while reading TXT file: {str(e)}")
            raise

    def extract_all_content(self, docx_path: str) -> str:
        """
        Comprehensively extract all content from DOCX file
        Supports multiple DOCX internal formats and storage methods
        """
        logger.info(f"🔍 Starting comprehensive content extraction: {docx_path}")

        all_content = []

        try:
            with zipfile.ZipFile(docx_path, "r") as docx:
                # 1. Check and extract altChunk content (HTML/MHT embedded)
                altchunk_content = self._extract_altchunk_content_internal(docx)
                if altchunk_content:
                    all_content.append(("altChunk", altchunk_content))

                # 2. Extract standard document.xml content
                standard_content = self._extract_standard_document_content(docx)
                if standard_content:
                    all_content.append(("standard", standard_content))

                # 3. Extract embedded objects content (embeddings)
                embedded_content = self._extract_embedded_objects(docx)
                if embedded_content:
                    all_content.append(("embedded", embedded_content))

                # 4. Extract header and footer content
                header_footer_content = self._extract_headers_footers(docx)
                if header_footer_content:
                    all_content.append(("header_footer", header_footer_content))

                # 5. Extract comments and annotations
                comments_content = self._extract_comments(docx)
                if comments_content:
                    all_content.append(("comments", comments_content))

                # 6. Extract text from text boxes and graphic objects
                textbox_content = self._extract_textbox_content(docx)
                if textbox_content:
                    all_content.append(("textboxes", textbox_content))

        except Exception as e:
            logger.error(f"💥 Comprehensive content extraction failed: {str(e)}")
            return ""

        # Combine all content
        if all_content:
            combined_content = self._combine_extracted_content(all_content)
            logger.info(f"✅ Comprehensive extraction completed, total content length: {len(combined_content)} characters")
            logger.debug(f"📊 Extracted content types: {[item[0] for item in all_content]}")
            return combined_content

        return ""

    def _extract_altchunk_content_internal(self, docx_zip: zipfile.ZipFile) -> str:
        """Internal method: Extract altChunk content, prioritizing MHT method"""
        try:
            # Check altChunk references in document.xml
            if "word/document.xml" in docx_zip.namelist():
                doc_xml = docx_zip.read("word/document.xml").decode(
                    "utf-8", errors="replace"
                )
                if "altChunk" in doc_xml:
                    logger.info("🔍 Detected altChunk format")

                    # Prioritize MHT files (simpler processing method)
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

                    # Process MHT files first
                    for filename in mht_files:
                        logger.info(f"📄 Processing MHT file with priority: {filename}")
                        content = docx_zip.read(filename).decode(
                            "utf-8", errors="replace"
                        )
                        return self._extract_html_from_mht(content)

                    # If no MHT files, then process HTML files
                    for filename in html_files:
                        logger.info(f"📄 Processing HTML file: {filename}")
                        content = docx_zip.read(filename).decode(
                            "utf-8", errors="replace"
                        )
                        return self._html_to_clean_text(content)

            return ""
        except Exception as e:
            logger.error(f"💥 Failed to extract altChunk content: {str(e)}")
            return ""

    def _extract_standard_document_content(self, docx_zip: zipfile.ZipFile) -> str:
        """Extract standard document.xml content - only extract plain text"""
        try:
            if "word/document.xml" in docx_zip.namelist():
                doc_xml = docx_zip.read("word/document.xml").decode(
                    "utf-8", errors="replace"
                )

                # Decode XML entities
                doc_xml = html.unescape(doc_xml)

                # Extract all text in <w:t> tags (including various namespace prefixes)
                # Use more lenient regex to match any namespace prefix
                text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                text_matches = re.findall(text_pattern, doc_xml)

                # Additionally extract possible namespace-less <t> tags
                text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", doc_xml))

                if text_matches:
                    # Clean and combine text
                    cleaned_texts = []
                    for text in text_matches:
                        # Decode XML entities
                        text = html.unescape(text)
                        # Remove excess whitespace but preserve single spaces
                        text = re.sub(r"\s+", " ", text.strip())
                        if text:
                            cleaned_texts.append(text)

                    # Intelligently connect text fragments
                    content = ""
                    for i, text in enumerate(cleaned_texts):
                        if i == 0:
                            content = text
                        else:
                            # If previous text fragment doesn't end with punctuation and current text doesn't start with uppercase, don't add space
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

                    # Final cleanup
                    content = re.sub(r"\s+", " ", content)
                    content = content.strip()

                    logger.info(f"📝 Extracted plain text from document.xml: {len(content)} characters")
                    return content
            return ""
        except Exception as e:
            logger.error(f"💥 Failed to extract standard document content: {str(e)}")
            return ""

    def _extract_embedded_objects(self, docx_zip: zipfile.ZipFile) -> str:
        """Extract embedded objects content"""
        try:
            embedded_content = []

            # Look for embedded document objects
            for filename in docx_zip.namelist():
                if "word/embeddings/" in filename:
                    logger.info(f"📎 Found embedded object: {filename}")
                    # Further processing can be done here based on file type
                    # For example: .docx, .xlsx, .txt etc.

            return " ".join(embedded_content) if embedded_content else ""
        except Exception as e:
            logger.error(f"💥 Failed to extract embedded objects: {str(e)}")
            return ""

    def _extract_headers_footers(self, docx_zip: zipfile.ZipFile) -> str:
        """Extract header and footer content - only extract plain text"""
        try:
            header_footer_content = []

            for filename in docx_zip.namelist():
                if (
                    "word/header" in filename or "word/footer" in filename
                ) and filename.endswith(".xml"):
                    logger.debug(f"📄 Processing header/footer: {filename}")
                    content = docx_zip.read(filename).decode("utf-8", errors="replace")

                    # Decode XML entities
                    content = html.unescape(content)

                    # Extract text content - using more lenient pattern
                    text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                    text_matches = re.findall(text_pattern, content)
                    text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", content))

                    if text_matches:
                        # Clean and combine text
                        cleaned_texts = []
                        for text in text_matches:
                            text = html.unescape(text)
                            text = re.sub(r"\s+", " ", text.strip())
                            if text:
                                cleaned_texts.append(text)

                        if cleaned_texts:
                            # Merge text fragments
                            header_footer_text = " ".join(cleaned_texts)
                            header_footer_text = re.sub(
                                r"\s+", " ", header_footer_text.strip()
                            )
                            if header_footer_text:
                                header_footer_content.append(header_footer_text)

            if header_footer_content:
                logger.info(f"📑 Extracted header/footer plain text: {len(header_footer_content)} items")

            return "\n".join(header_footer_content) if header_footer_content else ""
        except Exception as e:
            logger.error(f"💥 Failed to extract headers/footers: {str(e)}")
            return ""

    def _extract_comments(self, docx_zip: zipfile.ZipFile) -> str:
        """Extract comments and annotations content - only extract plain text"""
        try:
            if "word/comments.xml" in docx_zip.namelist():
                comments_xml = docx_zip.read("word/comments.xml").decode(
                    "utf-8", errors="replace"
                )

                # Decode XML entities
                comments_xml = html.unescape(comments_xml)

                # Extract comment text - using more lenient pattern
                text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                text_matches = re.findall(text_pattern, comments_xml)
                text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", comments_xml))

                if text_matches:
                    # Clean and combine text
                    cleaned_texts = []
                    for text in text_matches:
                        text = html.unescape(text)
                        text = re.sub(r"\s+", " ", text.strip())
                        if text:
                            cleaned_texts.append(text)

                    if cleaned_texts:
                        comments_text = " ".join(cleaned_texts)
                        comments_text = re.sub(r"\s+", " ", comments_text.strip())
                        logger.info(f"💬 Extracted comments plain text: {len(comments_text)} characters")
                        return comments_text

            return ""
        except Exception as e:
            logger.error(f"💥 Failed to extract comments: {str(e)}")
            return ""

    def _extract_textbox_content(self, docx_zip: zipfile.ZipFile) -> str:
        """Extract text from text boxes and graphic objects - only extract plain text"""
        try:
            textbox_content = []

            # Look for files that might contain text boxes
            for filename in docx_zip.namelist():
                if "word/" in filename and filename.endswith(".xml"):
                    content = docx_zip.read(filename).decode("utf-8", errors="replace")

                    # Decode XML entities
                    content = html.unescape(content)

                    # Look for text box content (w:txbxContent)
                    textbox_matches = re.findall(
                        r"<[^:>]*:txbxContent[^>]*>(.*?)</[^:>]*:txbxContent>",
                        content,
                        re.DOTALL,
                    )

                    for match in textbox_matches:
                        # Extract text from text box content
                        text_pattern = r"<[^:>]*:t[^>]*>([^<]*)</[^:>]*:t>"
                        text_matches = re.findall(text_pattern, match)
                        text_matches.extend(re.findall(r"<t[^>]*>([^<]*)</t>", match))

                        if text_matches:
                            # Clean and combine text
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
                logger.info(f"📦 Extracted text box plain text: {len(textbox_content)} items")

            return "\n".join(textbox_content) if textbox_content else ""
        except Exception as e:
            logger.error(f"💥 Failed to extract text box content: {str(e)}")
            return ""

    def _combine_extracted_content(self, content_list: list) -> str:
        """Combine extracted various content - output clear plain text"""
        combined = []

        # Sort content by importance
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
                    # Clean excess whitespace in content
                    cleaned_content = re.sub(r"\s+", " ", content.strip())
                    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                    if cleaned_content:
                        # Add simple markers based on content type (only when there are multiple content types)
                        if len([1 for t, c in content_list if c.strip()]) > 1:
                            if item_type == "header_footer":
                                combined.append(f"[Header/Footer]\n{cleaned_content}")
                            elif item_type == "comments":
                                combined.append(f"[Comments]\n{cleaned_content}")
                            elif item_type == "textboxes":
                                combined.append(f"[Text Boxes]\n{cleaned_content}")
                            else:
                                combined.append(cleaned_content)
                        else:
                            combined.append(cleaned_content)

        # Add other uncategorized content
        for item_type, content in content_list:
            if item_type not in priority_order and content.strip():
                cleaned_content = re.sub(r"\s+", " ", content.strip())
                cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)
                if cleaned_content:
                    combined.append(cleaned_content)

        # Combine all content, use double line breaks to separate different sections
        final_content = "\n\n".join(combined) if combined else ""

        # Final cleanup: ensure no excessive empty lines
        final_content = re.sub(r"\n{3,}", "\n\n", final_content)
        final_content = final_content.strip()

        return final_content

    def _extract_html_from_mht(self, mht_content: str) -> str:
        """Extract HTML part from MHT content and convert to clean text"""
        try:
            # MHT files use MIME format, look for HTML part
            lines = mht_content.split("\n")
            in_html_section = False
            html_lines = []
            skip_headers = True

            for line in lines:
                # Detect HTML section start
                if "Content-Type: text/html" in line:
                    in_html_section = True
                    skip_headers = True
                    continue

                # In HTML section
                if in_html_section:
                    # Skip Content-* headers
                    if (
                        skip_headers
                        and line.strip()
                        and not line.startswith("Content-")
                    ):
                        skip_headers = False

                    # Empty line indicates end of headers, content starts
                    if skip_headers and not line.strip():
                        skip_headers = False
                        continue

                    # Check if reached next MIME part
                    if line.startswith("------=") and len(html_lines) > 0:
                        # HTML section ends
                        break

                    # Collect HTML content
                    if not skip_headers:
                        html_lines.append(line)

            # Combine all HTML lines
            html_content = "\n".join(html_lines)

            # Decode quoted-printable encoding
            if "=3D" in html_content or "=\n" in html_content:
                try:
                    import quopri

                    html_content = quopri.decodestring(html_content.encode()).decode(
                        "utf-8", errors="replace"
                    )
                    logger.info("📧 Decoded quoted-printable encoding")
                except Exception as e:
                    logger.warning(f"⚠️ Quoted-printable decoding failed: {str(e)}")

            logger.debug(f"📄 Extracted HTML content length: {len(html_content)} characters")

            # Convert to clean text
            return self._html_to_clean_text(html_content)

        except Exception as e:
            logger.error(f"💥 Failed to extract HTML from MHT: {str(e)}")
            return ""

    def _html_to_clean_text(self, html_content: str) -> str:
        """Convert HTML content to clean plain text, specifically optimized for MHT content"""
        try:
            # First decode HTML entities
            text = html.unescape(html_content)

            # First try to extract all content within <body> tags
            body_match = re.search(
                r"<body[^>]*>(.*?)</body>", text, re.DOTALL | re.IGNORECASE
            )
            if body_match:
                main_content = body_match.group(1)
                logger.info("📄 Extracted <body> tag content")
            else:
                main_content = text
                logger.info("📄 Using all content (no body tag found)")

            # Special handling for <pre><code> tags, preserve their internal formatting
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

            # Handle other HTML structures
            # 1. First convert tags that need to preserve line breaks
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

            # 2. Remove style and script tags and their content
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

            # 3. Remove all remaining HTML tags
            main_content = re.sub(r"<[^>]+>", "", main_content)

            # 4. Decode HTML entities (second time, ensure complete decoding)
            main_content = html.unescape(main_content)

            # 5. Restore <pre><code> block content
            for idx, pre_code_content in enumerate(pre_code_blocks):
                # Clean pre_code content
                cleaned_pre_code = html.unescape(pre_code_content)
                main_content = main_content.replace(
                    f"__PRE_CODE_{idx}__", cleaned_pre_code
                )

            # 6. Clean excess whitespace while maintaining paragraph structure
            lines = main_content.split("\n")
            cleaned_lines = []

            for line in lines:
                # Clean leading and trailing spaces of each line
                line = line.strip()
                # Keep non-empty lines
                if line:
                    # Clean excess spaces within lines
                    line = re.sub(r"[ \t]+", " ", line)
                    # Clean excess spaces around table separators
                    line = re.sub(r"\s*\|\s*", " | ", line)
                    cleaned_lines.append(line)
                else:
                    # Keep empty lines as paragraph separators
                    if cleaned_lines and cleaned_lines[-1] != "":
                        cleaned_lines.append("")

            # 7. Merge cleaned lines
            main_content = "\n".join(cleaned_lines)

            # 8. Final cleanup: remove excess empty lines
            main_content = re.sub(r"\n{3,}", "\n\n", main_content)
            main_content = main_content.strip()

            logger.info(f"📝 HTML content converted to clean text: {len(main_content)} characters")

            return main_content

        except Exception as e:
            logger.error(f"💥 HTML to clean text conversion failed: {str(e)}")
            # If conversion fails, return basic cleaned version of original text
            return re.sub(r"<[^>]+>", "", html_content)

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text (keep this method for other HTML content)"""
        # For non-MHT HTML content, use this more general method
        return self._html_to_clean_text(html_content)

    def extract_altchunk_content(self, docx_path: str) -> str | None:
        """
        Extract DOCX file content containing altChunk (maintain backward compatibility)
        """
        try:
            with zipfile.ZipFile(docx_path, "r") as docx:
                return self._extract_altchunk_content_internal(docx)
        except Exception as e:
            logger.error(f"💥 Failed to extract altChunk content: {str(e)}")
            return None

    def read_docx_file(self, docx_path: str) -> str:
        """Read docx file and convert to text"""
        logger.info(f"📖 Starting to read DOCX file - File: {docx_path}")

        try:
            # First try comprehensive extraction of all content
            comprehensive_content = self.extract_all_content(docx_path)
            if comprehensive_content and comprehensive_content.strip():
                logger.info(
                    f"✨ Comprehensive extraction successful, content length: {len(comprehensive_content)} characters"
                )
                return comprehensive_content

            # If comprehensive extraction fails, use traditional conversion method
            logger.info("🔄 Comprehensive extraction failed or content empty, using traditional conversion method")

            with tempfile.TemporaryDirectory() as temp_path:
                logger.debug(f"📁 Created temporary directory: {temp_path}")

                temp_dir = Path(temp_path)

                file_path = temp_dir / "tmp.docx"
                shutil.copy(docx_path, file_path)
                logger.debug(f"📋 Copied file to temporary directory: {docx_path} -> {file_path}")

                # Convert DOCX to TXT
                txt_file_path = self.docx_to_txt(str(file_path), str(temp_path))
                logger.info(f"🎯 DOCX to TXT conversion completed: {txt_file_path}")

                # Read TXT file content
                content = self.read_txt_file(txt_file_path)
                logger.info(f"✨ TXT file content reading completed, content length: {len(content)} characters")

                return content

        except FileNotFoundError as e:
            logger.error(f"🚫 File not found: {str(e)}")
            raise Exception(f"File not found: {docx_path}")
        except PermissionError as e:
            logger.error(f"🔒 File permission error: {str(e)}")
            raise Exception(f"No permission to access file: {docx_path}")
        except Exception as e:
            logger.error(f"💥 Error occurred while reading DOCX file: {str(e)}")
            raise

    def parse(self, file_path: str):
        """Parse DOCX file"""
        logger.info(f"🎬 Starting to parse DOCX file: {file_path}")

        try:
            # Validate file exists
            if not os.path.exists(file_path):
                logger.error(f"🚫 File does not exist: {file_path}")
                raise FileNotFoundError(f"File does not exist: {file_path}")

            # Validate file extension
            if not file_path.lower().endswith(".docx"):
                logger.warning(f"⚠️ File extension is not .docx: {file_path}")

            # Validate file size
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 File size: {file_size} bytes")

            if file_size == 0:
                logger.warning(f"⚠️ File size is 0 bytes: {file_path}")

            # 🏷️ Extract file extension
            extension = self.get_file_extension(file_path)
            logger.debug(f"🏷️ Extracted file extension: {extension}")
            # 1) Processing start: Generate DATA_PROCESSING event
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Parsing",
            )
            # Use soffice to convert to txt and read content
            logger.info("📝 Using soffice to convert DOCX to TXT and read content")
            content = self.read_docx_file(docx_path=file_path)

            # Decide whether to keep original format or process as markdown format based on to_markdown parameter
            if self.to_markdown:
                # Simple text to markdown conversion (maintain paragraph structure)
                mk_content = self.format_as_markdown(content)
                logger.info("🎨 Content formatted as markdown")
            else:
                mk_content = content
                logger.info("📝 Maintaining original text format")

            logger.info(f"🎊 File content parsing completed, final content length: {len(mk_content)} characters")

            # Check if content is empty
            if not mk_content.strip():
                logger.warning(f"⚠️ Parsed content is empty: {file_path}")

            # 2) Processing end: Generate DATA_PROCESSED or DATA_PROCESS_FAILED event based on whether content is non-empty
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
            logger.debug("⚙️ Lifecycle event generation completed")

            # 3) Encapsulate output and add lifecycle
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)

            result = output_vo.to_dict()
            logger.info(f"🏆 DOCX file parsing completed: {file_path}")
            logger.debug(f"🔑 Return result keys: {list(result.keys())}")

            return result

        except FileNotFoundError as e:
            logger.error(f"🚫 File not found error: {str(e)}")
            raise
        except PermissionError as e:
            logger.error(f"🔒 File permission error: {str(e)}")
            raise Exception(f"No permission to access file: {file_path}")
        except Exception as e:
            logger.error(
                f"💀 DOCX file parsing failed: {file_path}, Error type: {type(e).__name__}, Error message: {str(e)}"
            )
            raise

    def format_as_markdown(self, content: str) -> str:
        """Format plain text as simple markdown format"""
        if not content.strip():
            return content

        lines = content.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue

            # Simple markdown formatting rules
            # Can be extended with more rules as needed
            formatted_lines.append(line)

        return "\n".join(formatted_lines)
