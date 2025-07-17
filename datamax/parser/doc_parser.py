import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import chardet
from loguru import logger

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType

# Try to import OLE-related libraries (for reading DOC internal structure)
try:
    import olefile

    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False
    logger.warning("⚠️ olefile library not installed, advanced DOC parsing unavailable")

# Try to import UNO processor
try:
    from datamax.utils.uno_handler import HAS_UNO, convert_with_uno
except ImportError:
    HAS_UNO = False
    logger.error(
        "❌ UNO processor import failed!\n"
        "🔧 Solution:\n"
        "1. Install LibreOffice and python-uno:\n"
        "   - Ubuntu/Debian: sudo apt-get install libreoffice python3-uno\n"
        "   - CentOS/RHEL: sudo yum install libreoffice python3-uno\n"
        "   - macOS: brew install libreoffice\n"
        "   - Windows: Download and install LibreOffice\n"
        "2. Ensure Python can access uno module:\n"
        "   - Linux: export PYTHONPATH=/usr/lib/libreoffice/program:$PYTHONPATH\n"
        "   - Windows: Add LibreOffice\\program to system PATH\n"
        "3. Verify installation: python -c 'import uno'\n"
        "4. If issues persist, see full documentation:\n"
        "   https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
    )


class DocParser(BaseLife):
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

        # Prioritize UNO (unless explicitly disabled)
        if use_uno and HAS_UNO:
            self.use_uno = True
            logger.info(f"🚀 DocParser initialized - using UNO API for single-threaded efficient processing")
        else:
            self.use_uno = False
            if use_uno and not HAS_UNO:
                logger.warning(
                    f"⚠️ UNO unavailable, falling back to traditional command line approach\n"
                    f"💡 Tip: UNO conversion is faster and more stable, strongly recommend installing and configuring UNO\n"
                    f"📖 Please refer to installation guide in error message above"
                )
            else:
                logger.info(f"🚀 DocParser initialized - using traditional command line approach")

        logger.info(f"📄 File path: {file_path}, convert to markdown: {to_markdown}")

    def extract_all_content(self, doc_path: str) -> str:
        """
        Comprehensively extract all content from DOC files
        Supports multiple DOC internal formats and storage methods
        """
        logger.info(f"🔍 Starting comprehensive content extraction: {doc_path}")

        all_content = []

        try:
            # 1. Try to extract content using OLE parsing (if available)
            if HAS_OLEFILE:
                ole_content = self._extract_ole_content(doc_path)
                if ole_content:
                    all_content.append(("ole", ole_content))

            # 2. Try to extract embedded objects
            embedded_content = self._extract_embedded_objects(doc_path)
            if embedded_content:
                all_content.append(("embedded", embedded_content))

            # 3. If none of the above methods extracted content, use traditional conversion
            if not all_content:
                logger.info("🔄 Using traditional conversion method to extract content")
                return ""  # Return empty, let caller use traditional method

            # Check content quality, especially for WPS files
            for content_type, content in all_content:
                if content and self._check_content_quality(content):
                    logger.info(f"✅ Successfully extracted content using {content_type}")
                    return content

            # If all content quality is poor, return empty
            logger.warning("⚠️ All extraction methods produced poor quality content")
            return ""

        except Exception as e:
            logger.error(f"💥 Comprehensive content extraction failed: {str(e)}")
            return ""

    def _extract_ole_content(self, doc_path: str) -> str:
        """Extract DOC content using OLE parsing"""
        try:
            ole = olefile.OleFileIO(doc_path)
            logger.info(f"📂 Successfully opened OLE file: {doc_path}")

            # List all streams
            streams = ole.listdir()
            logger.debug(f"📋 Available OLE streams: {streams}")

            # Check if it's a WPS-generated file
            is_wps = any("WpsCustomData" in str(stream) for stream in streams)
            if is_wps:
                logger.info("📝 Detected WPS DOC file, traditional conversion method recommended")
                # For WPS files, OLE parsing may be unreliable, return empty to use traditional method
                ole.close()
                return ""

            all_texts = []

            # Try to extract WordDocument stream
            if ole.exists("WordDocument"):
                try:
                    word_stream = ole.openstream("WordDocument").read()
                    logger.info(f"📄 WordDocument stream size: {len(word_stream)} bytes")
                    text = self._parse_word_stream(word_stream)
                    if text:
                        all_texts.append(text)
                except Exception as e:
                    logger.error(f"💥 Failed to parse WordDocument stream: {str(e)}")

            # Try to read other streams that might contain text
            text_content = []
            for entry in ole.listdir():
                if any(name in str(entry) for name in ["Text", "Content", "Body"]):
                    try:
                        stream = ole.openstream(entry)
                        data = stream.read()
                        # Try to decode
                        decoded = self._try_decode_bytes(data)
                        if decoded and len(decoded.strip()) > 10:
                            text_content.append(decoded)
                    except:
                        continue

            if text_content:
                combined = "\n".join(text_content)
                logger.info(f"📄 Extracted text from OLE streams: {len(combined)} characters")
                return self._clean_extracted_text(combined)

            ole.close()

            return ""

        except Exception as e:
            logger.warning(f"⚠️ OLE parsing failed: {str(e)}")

        return ""

    def _parse_word_stream(self, data: bytes) -> str:
        """Parse text from WordDocument stream"""
        try:
            # DOC file format is complex, this provides basic text extraction
            # Look for text fragments
            text_parts = []

            # Try multiple encodings, especially for Chinese text
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
                    # Check if contains reasonable Chinese characters
                    chinese_chars = len(
                        [c for c in decoded if "\u4e00" <= c <= "\u9fff"]
                    )
                    if chinese_chars > 10 or (decoded and len(decoded.strip()) > 50):
                        # Filter printable characters, but keep Chinese
                        cleaned = self._filter_printable_text(decoded)
                        if cleaned and len(cleaned.strip()) > 20:
                            text_parts.append(cleaned)
                            logger.debug(
                                f"📝 Successfully decoded using {encoding}, contains {chinese_chars} Chinese characters"
                            )
                            break
                except:
                    continue

            return "\n".join(text_parts) if text_parts else ""

        except Exception as e:
            logger.error(f"💥 Failed to parse Word stream: {str(e)}")
            return ""

    def _filter_printable_text(self, text: str) -> str:
        """Filter text, keeping printable characters and Chinese"""
        result = []
        for char in text:
            # Keep Chinese characters
            if "\u4e00" <= char <= "\u9fff":
                result.append(char)
            # Keep Japanese characters
            elif "\u3040" <= char <= "\u30ff":
                result.append(char)
            # Keep Korean characters
            elif "\uac00" <= char <= "\ud7af":
                result.append(char)
            # Keep ASCII printable characters and whitespace
            elif char.isprintable() or char.isspace():
                result.append(char)
            # Keep common punctuation marks
            elif char in '，。！？；：""' "（）【】《》、·…—":
                result.append(char)

        return "".join(result)

    def _try_decode_bytes(self, data: bytes) -> str:
        """Try to decode byte data using multiple encodings"""
        # Prioritize Chinese encodings
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

        # First try to detect encoding using chardet
        try:
            import chardet

            detected = chardet.detect(data)
            if detected["encoding"] and detected["confidence"] > 0.7:
                encodings.insert(0, detected["encoding"])
                logger.debug(
                    f"🔍 Detected encoding: {detected['encoding']} (confidence: {detected['confidence']})"
                )
        except:
            pass

        for encoding in encodings:
            try:
                decoded = data.decode(encoding, errors="ignore")
                # Check if contains meaningful text (including Chinese)
                if decoded and (
                    any(c.isalnum() for c in decoded)
                    or any("\u4e00" <= c <= "\u9fff" for c in decoded)
                ):
                    # Further clean the text
                    cleaned = self._filter_printable_text(decoded)
                    if cleaned and len(cleaned.strip()) > 10:
                        return cleaned
            except:
                continue

        return ""

    def _extract_embedded_objects(self, doc_path: str) -> str:
        """Extract embedded objects from DOC file"""
        try:
            if not HAS_OLEFILE:
                return ""

            embedded_content = []

            with olefile.OleFileIO(doc_path) as ole:
                # Look for embedded objects
                for entry in ole.listdir():
                    entry_name = "/".join(entry)

                    # Check if it's an embedded object
                    if any(
                        pattern in entry_name.lower()
                        for pattern in ["object", "embed", "package"]
                    ):
                        logger.info(f"📎 Found embedded object: {entry_name}")
                        try:
                            stream = ole.openstream(entry)
                            data = stream.read()

                            # Try to extract text content
                            text = self._try_decode_bytes(data)
                            if text and len(text.strip()) > 20:
                                embedded_content.append(text.strip())
                        except:
                            continue

            return "\n\n".join(embedded_content) if embedded_content else ""

        except Exception as e:
            logger.warning(f"⚠️ Failed to extract embedded objects: {str(e)}")
            return ""

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text, thoroughly remove all XML tags and control characters, keep only plain text"""
        try:
            # 1. Decode HTML/XML entities
            text = html.unescape(text)

            # 2. Remove all XML/HTML tags
            text = re.sub(r"<[^>]+>", "", text)

            # 3. Remove XML namespace prefixes
            text = re.sub(r"\b\w+:", "", text)

            # 4. Remove NULL characters and other control characters
            text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

            # 5. Remove special XML character sequences
            text = re.sub(r"&[a-zA-Z]+;", "", text)
            text = re.sub(r"&#\d+;", "", text)
            text = re.sub(r"&#x[0-9a-fA-F]+;", "", text)

            # 6. Keep meaningful characters, remove other special characters
            # Keep: Chinese, Japanese, Korean, English, numbers, common punctuation and whitespace
            allowed_chars = (
                r"\w\s"  # Letters, numbers and whitespace
                r"\u4e00-\u9fff"  # Chinese
                r"\u3040-\u30ff"  # Japanese
                r"\uac00-\ud7af"  # Korean
                r'，。！？；：""'
                "（）【】《》、·…—"  # Chinese punctuation
                r'.,!?;:()[\]{}"\'`~@#$%^&*+=\-_/\\'  # English punctuation and common symbols
            )

            # Use stricter filtering, but keep all meaningful characters
            cleaned_text = "".join(
                char for char in text if re.match(f"[{allowed_chars}]", char)
            )

            # 7. Remove excessively long meaningless character sequences (usually binary garbage)
            cleaned_text = re.sub(r"([^\s\u4e00-\u9fff])\1{5,}", r"\1", cleaned_text)

            # 8. Clean excessive whitespace, but preserve paragraph structure
            cleaned_text = re.sub(
                r"[ \t]+", " ", cleaned_text
            )  # Multiple spaces/tabs become single space
            cleaned_text = re.sub(
                r"\n\s*\n\s*\n+", "\n\n", cleaned_text
            )  # Multiple empty lines become double empty lines
            cleaned_text = re.sub(
                r"^\s+|\s+$", "", cleaned_text, flags=re.MULTILINE
            )  # Remove leading/trailing whitespace

            # 9. Further cleaning: remove standalone punctuation lines
            lines = cleaned_text.split("\n")
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line:
                    # Check if line is mainly meaningful content
                    # Calculate ratio of Chinese, English letters and numbers
                    meaningful_chars = sum(
                        1 for c in line if (c.isalnum() or "\u4e00" <= c <= "\u9fff")
                    )

                    # Keep if meaningful chars ratio > 30%, or line length < 5 (might be title)
                    if len(line) < 5 or (
                        meaningful_chars > 0 and meaningful_chars / len(line) > 0.3
                    ):
                        cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1]:  # Keep paragraph separators
                    cleaned_lines.append("")

            result = "\n".join(cleaned_lines).strip()

            # 10. Final check
            if len(result) < 10:
                logger.warning("⚠️ Cleaned text too short, may have issues")
                return ""

            # Check if still contains XML tags
            if re.search(r"<[^>]+>", result):
                logger.warning("⚠️ Cleaned text still contains XML tags, performing second cleanup")
                result = re.sub(r"<[^>]+>", "", result)

            return result

        except Exception as e:
            logger.error(f"💥 Failed to clean text: {str(e)}")
            return text

    def _combine_extracted_content(self, content_list: list) -> str:
        """Combine various extracted content"""
        combined = []

        # Sort content by priority
        priority_order = ["ole", "embedded", "converted", "fallback"]

        for content_type in priority_order:
            for item_type, content in content_list:
                if item_type == content_type and content.strip():
                    combined.append(content.strip())

        # Add other uncategorized content
        for item_type, content in content_list:
            if item_type not in priority_order and content.strip():
                combined.append(content.strip())

        return "\n\n".join(combined) if combined else ""

    def doc_to_txt(self, doc_path: str, dir_path: str) -> str:
        """Convert .doc file to .txt file"""
        logger.info(
            f"🔄 Starting DOC to TXT conversion - source file: {doc_path}, output directory: {dir_path}"
        )

        if self.use_uno:
            # Use UNO API for conversion
            try:
                logger.info("🎯 Using UNO API for document conversion...")
                txt_path = convert_with_uno(doc_path, "txt", dir_path)

                if not os.path.exists(txt_path):
                    logger.error(f"❌ Converted TXT file does not exist: {txt_path}")
                    raise Exception(f"File conversion failed {doc_path} ==> {txt_path}")
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
                    f"   1. Ensure LibreOffice is properly installed\n"
                    f"   2. Close all LibreOffice processes\n"
                    f"   3. Check file permissions and paths\n"
                    f'   4. Try manual run: soffice --headless --convert-to txt "{doc_path}"'
                )
                logger.warning("⚠️ Automatically falling back to traditional command line method...")
                return self._doc_to_txt_subprocess(doc_path, dir_path)
        else:
            # Use traditional subprocess method
            return self._doc_to_txt_subprocess(doc_path, dir_path)

    def _doc_to_txt_subprocess(self, doc_path: str, dir_path: str) -> str:
        """Convert .doc file to .txt file using subprocess (traditional method)"""
        try:
            cmd = f'soffice --headless --convert-to txt "{doc_path}" --outdir "{dir_path}"'
            logger.debug(f"⚡ Executing conversion command: {cmd}")

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code == 0:
                logger.info(f"✅ DOC to TXT conversion successful - exit code: {exit_code}")
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
                    f"❌ DOC to TXT conversion failed - exit code: {exit_code}, error message: {error_msg}"
                )
                raise Exception(
                    f"Error Output (detected encoding: {encoding}): {error_msg}"
                )

            fname = str(Path(doc_path).stem)
            txt_path = os.path.join(dir_path, f"{fname}.txt")

            if not os.path.exists(txt_path):
                logger.error(f"❌ Converted TXT file does not exist: {txt_path}")
                raise Exception(f"File conversion failed {doc_path} ==> {txt_path}")
            else:
                logger.info(f"🎉 TXT file conversion successful, file path: {txt_path}")
                return txt_path

        except subprocess.SubprocessError as e:
            logger.error(f"💥 subprocess execution failed: {str(e)}")
            raise Exception(f"Error occurred while executing conversion command: {str(e)}")
        except Exception as e:
            logger.error(f"💥 Unknown error occurred during DOC to TXT conversion: {str(e)}")
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

            logger.info(f"📄 TXT file reading complete - content length: {len(content)} characters")
            logger.debug(f"👀 First 100 characters preview: {content[:100]}...")

            return content

        except FileNotFoundError as e:
            logger.error(f"🚫 TXT file not found: {str(e)}")
            raise Exception(f"File not found: {txt_path}")
        except Exception as e:
            logger.error(f"💥 Error occurred while reading TXT file: {str(e)}")
            raise

    def read_doc_file(self, doc_path: str) -> str:
        """Read doc file and convert to text"""
        logger.info(f"📖 Starting to read DOC file - file: {doc_path}")

        try:
            # First try comprehensive extraction (if advanced parsing features available)
            if HAS_OLEFILE:
                comprehensive_content = self.extract_all_content(doc_path)
                if comprehensive_content and comprehensive_content.strip():
                    # Check content quality
                    if self._check_content_quality(comprehensive_content):
                        logger.info(
                            f"✨ Comprehensive extraction successful, content length: {len(comprehensive_content)} characters"
                        )
                        return comprehensive_content
                    else:
                        logger.warning("⚠️ Comprehensive extraction content quality poor, trying other methods")

            # Fallback to traditional conversion method
            logger.info("🔄 Using traditional conversion method")

            with tempfile.TemporaryDirectory() as temp_path:
                logger.debug(f"📁 Created temporary directory: {temp_path}")

                temp_dir = Path(temp_path)

                file_path = temp_dir / "tmp.doc"
                shutil.copy(doc_path, file_path)
                logger.debug(f"📋 Copied file to temporary directory: {doc_path} -> {file_path}")

                # Convert DOC to TXT
                txt_file_path = self.doc_to_txt(str(file_path), str(temp_path))
                logger.info(f"🎯 DOC to TXT conversion complete: {txt_file_path}")

                # Read TXT file content
                content = self.read_txt_file(txt_file_path)
                logger.info(f"✨ TXT file content reading complete, content length: {len(content)} characters")

                return content

        except FileNotFoundError as e:
            logger.error(f"🚫 File not found: {str(e)}")
            raise Exception(f"File not found: {doc_path}")
        except PermissionError as e:
            logger.error(f"🔒 File permission error: {str(e)}")
            raise Exception(f"No permission to access file: {doc_path}")
        except Exception as e:
            logger.error(f"💥 Error occurred while reading DOC file: {str(e)}")
            raise

    def _check_content_quality(self, content: str) -> bool:
        """Check quality of extracted content"""
        if not content or len(content) < 50:
            return False

        # Calculate ratio of garbled characters
        total_chars = len(content)
        # Recognizable characters: ASCII, Chinese, Japanese, Korean, common punctuation
        recognizable = sum(
            1
            for c in content
            if (
                c.isascii()
                or "\u4e00" <= c <= "\u9fff"  # Chinese
                or "\u3040" <= c <= "\u30ff"  # Japanese
                or "\uac00" <= c <= "\ud7af"  # Korean
                or c in '，。！？；：""' "（）【】《》、·…—\n\r\t "
            )
        )

        # If recognizable character ratio < 70%, consider quality poor
        if recognizable / total_chars < 0.7:
            logger.warning(
                f"⚠️ Content quality check failed: recognizable character ratio {recognizable}/{total_chars} = {recognizable/total_chars:.2%}"
            )
            return False

        return True

    def parse(self, file_path: str):
        """Parse DOC file"""
        logger.info(f"🎬 Starting to parse DOC file: {file_path}")

        try:
            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"🚫 File does not exist: {file_path}")
                raise FileNotFoundError(f"File does not exist: {file_path}")

            # Verify file extension
            if not file_path.lower().endswith(".doc"):
                logger.warning(f"⚠️ File extension is not .doc: {file_path}")

            # Verify file size
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 File size: {file_size} bytes")

            if file_size == 0:
                logger.warning(f"⚠️ File size is 0 bytes: {file_path}")
            # Lifecycle: Data Processing start
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                life_type=LifeType.DATA_PROCESSING,
                usage_purpose="Documentation",
            )

            # 🏷️ Extract file extension
            extension = self.get_file_extension(file_path)
            logger.debug(f"🏷️ Extracted file extension: {extension}")

            # Read file content
            logger.info("📝 Reading DOC file content")
            content = self.read_doc_file(doc_path=file_path)

            # Decide whether to keep original format or process as markdown based on to_markdown parameter
            if self.to_markdown:
                # Simple text to markdown conversion (preserve paragraph structure)
                mk_content = self.format_as_markdown(content)
                logger.info("🎨 Content formatted as markdown")
            else:
                mk_content = content
                logger.info("📝 Keeping original text format")
            # 3) Lifecycle: Data Processed or Failed
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

            logger.info(f"🎊 File content parsing complete, final content length: {len(mk_content)} characters")

            # Check if content is empty
            if not mk_content.strip():
                logger.warning(f"⚠️ Parsed content is empty: {file_path}")

            lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type="LLM_ORIGIN",
            )
            logger.debug("⚙️ Lifecycle information generation complete")

            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)
            # output_vo.add_lifecycle(lc_origin)

            result = output_vo.to_dict()
            logger.info(f"🏆 DOC file parsing complete: {file_path}")
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
                f"💀 Failed to parse DOC file: {file_path}, error type: {type(e).__name__}, error message: {str(e)}"
            )
            raise

    def format_as_markdown(self, content: str) -> str:
        """Format plain text as simple markdown"""
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
            # Can extend with more rules as needed
            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _extract_text_from_wps_stream(self, data: bytes) -> str:
        """Extract text from WPS WordDocument stream (using more lenient strategy)"""
        try:
            text_parts = []

            # WPS files may use different encodings and structures
            # Try multiple strategies to extract text

            # Strategy 1: Try to find continuous text blocks
            # Look for byte sequences that look like text
            i = 0
            while i < len(data):
                # Look for possible text start positions
                if i + 2 < len(data):
                    # Check if it's Unicode text (little endian)
                    if data[i + 1] == 0 and 32 <= data[i] <= 126:
                        # Might be ASCII character Unicode encoding
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
                    # Check if it's UTF-8 or GBK Chinese
                    elif 0xE0 <= data[i] <= 0xEF or 0x81 <= data[i] <= 0xFE:
                        # Might be multi-byte character
                        text_block = bytearray()
                        j = i
                        while j < len(data):
                            if data[j] < 32 and data[j] not in [9, 10, 13]:
                                break
                            text_block.append(data[j])
                            j += 1
                        if len(text_block) > 20:
                            # Try to decode
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

            # Combine text parts
            if text_parts:
                combined = "\n".join(text_parts)
                return self._clean_extracted_text(combined)

            # If above method fails, fallback to original method
            return self._parse_word_stream(data)

        except Exception as e:
            logger.error(f"💥 Failed to parse WPS stream: {str(e)}")
            return ""
