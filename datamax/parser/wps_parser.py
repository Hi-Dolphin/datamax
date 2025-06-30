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
        "4. If you still have issues, please check the full documentation:\n"
        "   https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
    )


class WpsParser(BaseLife):
    """
    WPS Document Parser
    
    Supports parsing WPS format files, mainly including:
    1. Microsoft Works .wps files
    2. WPS Office proprietary format .wps files
    
    Uses LibreOffice as the underlying conversion engine, supporting both UNO API and command line methods
    """

    def __init__(
        self,
        file_path: Union[str, list],
        to_markdown: bool = False,
        use_uno: bool = True,
    ):
        super().__init__()
        self.file_path = file_path
        self.to_markdown = to_markdown

        # Prefer UNO (unless explicitly disabled)
        if use_uno and HAS_UNO:
            self.use_uno = True
            logger.info(f"🚀 WpsParser initialized - Using UNO API for single-threaded efficient processing")
        else:
            self.use_uno = False
            if use_uno and not HAS_UNO:
                logger.warning(
                    f"⚠️ UNO unavailable, falling back to traditional command line method\n"
                    f"💡 Tip: UNO conversion is faster and more stable, strongly recommend installing and configuring UNO\n"
                    f"📖 Please refer to the installation guide in the error message above"
                )
            else:
                logger.info(f"🚀 WpsParser initialized - Using traditional command line method")

        logger.info(f"📄 File path: {file_path}, Convert to markdown: {to_markdown}")

    def wps_to_txt(self, wps_path: str, dir_path: str) -> str:
        """Convert .wps file to .txt file"""
        logger.info(
            f"🔄 Starting WPS to TXT conversion - Source file: {wps_path}, Output directory: {dir_path}"
        )

        if self.use_uno:
            # Use UNO API for conversion
            try:
                logger.info("🎯 Using UNO API for WPS document conversion...")
                txt_path = convert_with_uno(wps_path, "txt", dir_path)

                if not os.path.exists(txt_path):
                    logger.error(f"❌ Converted TXT file does not exist: {txt_path}")
                    raise Exception(f"File conversion failed {wps_path} ==> {txt_path}")
                else:
                    logger.info(f"🎉 TXT file conversion successful, file path: {txt_path}")
                    return txt_path

            except Exception as e:
                logger.error(
                    f"💥 UNO conversion of WPS file failed: {str(e)}\n"
                    f"🔍 Diagnostic information:\n"
                    f"   - Error type: {type(e).__name__}\n"
                    f"   - Is LibreOffice installed? Try running: soffice --version\n"
                    f"   - Is Python UNO module available? Try: python -c 'import uno'\n"
                    f"   - Are there other LibreOffice instances running?\n"
                    f"   - Are file permissions correct?\n"
                    f"   - Is the WPS file corrupted or using an unsupported version?\n"
                    f"🔧 Possible solutions:\n"
                    f"   1. Ensure LibreOffice is properly installed and supports WPS format\n"
                    f"   2. Close all LibreOffice processes\n"
                    f"   3. Check file permissions and paths\n"
                    f'   4. Try manual execution: soffice --headless --convert-to txt "{wps_path}"'
                )
                logger.warning("⚠️ Automatically falling back to traditional command line method...")
                return self._wps_to_txt_subprocess(wps_path, dir_path)
        else:
            # Use traditional subprocess method
            return self._wps_to_txt_subprocess(wps_path, dir_path)

    def _wps_to_txt_subprocess(self, wps_path: str, dir_path: str) -> str:
        """Convert .wps file to .txt file using subprocess (traditional method)"""
        try:
            cmd = f'soffice --headless --convert-to txt "{wps_path}" --outdir "{dir_path}"'
            logger.debug(f"⚡ Executing WPS conversion command: {cmd}")

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code == 0:
                logger.info(f"✅ WPS to TXT conversion successful - Exit code: {exit_code}")
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
                    f"❌ WPS to TXT conversion failed - Exit code: {exit_code}, Error message: {error_msg}"
                )
                
                # Check if it's a format not supported issue
                if "not supported" in error_msg.lower() or "filter" in error_msg.lower():
                    logger.warning(
                        f"⚠️ LibreOffice may not support this WPS file format\n"
                        f"💡 Suggestions:\n"
                        f"   1. Check if the WPS file is a valid format\n"
                        f"   2. Try saving as .doc or .docx format using WPS Office software\n"
                        f"   3. Confirm if LibreOffice version supports this WPS format"
                    )
                
                raise Exception(
                    f"WPS file conversion failed (detected encoding: {encoding}): {error_msg}"
                )

            fname = str(Path(wps_path).stem)
            txt_path = os.path.join(dir_path, f"{fname}.txt")

            if not os.path.exists(txt_path):
                logger.error(f"❌ Converted TXT file does not exist: {txt_path}")
                raise Exception(f"WPS file conversion failed {wps_path} ==> {txt_path}")
            else:
                logger.info(f"🎉 TXT file conversion successful, file path: {txt_path}")
                return txt_path

        except subprocess.SubprocessError as e:
            logger.error(f"💥 Subprocess execution failed: {str(e)}")
            raise Exception(f"Error occurred while executing WPS conversion command: {str(e)}")
        except Exception as e:
            logger.error(f"💥 Unknown error occurred during WPS to TXT conversion: {str(e)}")
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

    def detect_wps_format(self, wps_path: str) -> str:
        """Detect the specific format type of WPS file"""
        logger.info(f"🔍 Detecting WPS file format: {wps_path}")
        
        try:
            with open(wps_path, "rb") as f:
                # Read file header
                header = f.read(512)
                
                # Check if it's Microsoft Works format
                if b"Microsoft Works" in header:
                    logger.info("📋 Detected Microsoft Works WPS format")
                    return "ms_works"
                
                # Check if it's OLE format (possibly WPS Office format)
                if header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                    logger.info("📋 Detected OLE format WPS file")
                    return "ole_based"
                
                # Check if it contains WPS Office characteristics
                if b"WPS" in header or b"Kingsoft" in header:
                    logger.info("📋 Detected WPS Office format")
                    return "wps_office"
                
                # Default processing as generic WPS format
                logger.info("📋 Detected generic WPS format")
                return "generic"
                
        except Exception as e:
            logger.warning(f"⚠️ Error occurred while detecting WPS format: {str(e)}, using default processing method")
            return "generic"

    def read_wps_file(self, wps_path: str) -> str:
        """Directly read WPS file content (as backup solution)"""
        logger.info(f"📖 Attempting to directly read WPS file content: {wps_path}")
        
        format_type = self.detect_wps_format(wps_path)
        
        try:
            # Use different reading strategies for different formats
            content = ""
            
            if format_type == "ms_works":
                content = self._read_ms_works_wps(wps_path)
            elif format_type == "ole_based":
                content = self._read_ole_based_wps(wps_path)
            else:
                content = self._read_generic_wps(wps_path)
            
            if content and len(content.strip()) > 10:
                logger.info(f"✅ Direct reading of WPS file successful - Content length: {len(content)} characters")
                return content
            else:
                logger.warning("⚠️ Direct reading of WPS file content is empty or too short")
                return ""
                
        except Exception as e:
            logger.error(f"💥 Direct reading of WPS file failed: {str(e)}")
            return ""

    def _read_ms_works_wps(self, wps_path: str) -> str:
        """Read Microsoft Works WPS format"""
        try:
            with open(wps_path, "rb") as f:
                data = f.read()
                
                # Try multiple encodings
                for encoding in ["utf-8", "gbk", "gb18030", "cp1252", "latin1"]:
                    try:
                        text = data.decode(encoding, errors="ignore")
                        # Filter out readable text
                        cleaned_text = self._extract_readable_text(text)
                        if cleaned_text and len(cleaned_text.strip()) > 20:
                            return cleaned_text
                    except:
                        continue
                        
            return ""
        except Exception as e:
            logger.error(f"💥 Failed to read MS Works WPS file: {str(e)}")
            return ""

    def _read_ole_based_wps(self, wps_path: str) -> str:
        """Read OLE-based WPS format"""
        try:
            # Try to import olefile library to handle OLE format
            try:
                import olefile
                HAS_OLEFILE = True
            except ImportError:
                HAS_OLEFILE = False
                logger.warning("⚠️ olefile library not installed, cannot parse OLE format WPS files")
                return ""
            
            if HAS_OLEFILE:
                with olefile.OleFileIO(wps_path) as ole:
                    # Try to extract text content
                    streams = ole.listdir()
                    logger.debug(f"📋 Found OLE streams: {streams}")
                    
                    for stream in streams:
                        try:
                            if any(name in str(stream).lower() for name in ["content", "text", "body", "document"]):
                                data = ole.openstream(stream).read()
                                text = self._try_decode_bytes(data)
                                if text and len(text.strip()) > 20:
                                    return text
                        except:
                            continue
            
            return ""
        except Exception as e:
            logger.error(f"💥 Failed to read OLE format WPS file: {str(e)}")
            return ""

    def _read_generic_wps(self, wps_path: str) -> str:
        """Read generic WPS format"""
        try:
            with open(wps_path, "rb") as f:
                data = f.read()
                
                # Try to decode
                decoded_text = self._try_decode_bytes(data)
                if decoded_text:
                    return self._extract_readable_text(decoded_text)
                    
            return ""
        except Exception as e:
            logger.error(f"💥 Failed to read generic WPS file: {str(e)}")
            return ""

    def _try_decode_bytes(self, data: bytes) -> str:
        """Try to decode byte data using multiple encodings"""
        # Prefer Chinese encodings and common encodings
        encodings = [
            "utf-8", "gbk", "gb18030", "gb2312", "big5",
            "utf-16-le", "utf-16-be", "cp936", "cp1252", "latin1"
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
                # Check if it contains meaningful text
                if decoded and (
                    any(c.isalnum() for c in decoded)
                    or any("\u4e00" <= c <= "\u9fff" for c in decoded)
                ):
                    return decoded
            except:
                continue

        return ""

    def _extract_readable_text(self, text: str) -> str:
        """Extract readable content from raw text"""
        try:
            # Remove control characters, but keep Chinese, English and common punctuation
            lines = []
            for line in text.split('\n'):
                # Extract readable characters from each line
                readable_chars = []
                for char in line:
                    # Keep Chinese characters
                    if "\u4e00" <= char <= "\u9fff":
                        readable_chars.append(char)
                    # Keep ASCII alphanumeric
                    elif char.isalnum():
                        readable_chars.append(char)
                    # Keep common punctuation and spaces
                    elif char in " .,!?;:()[]{}\"'-_/\\":
                        readable_chars.append(char)
                    # Keep Chinese punctuation
                    elif char in "，。！？；：""''（）【】《》、":
                        readable_chars.append(char)
                
                line_text = "".join(readable_chars).strip()
                if line_text and len(line_text) > 2:
                    lines.append(line_text)
            
            # Merge all valid lines
            result = "\n".join(lines)
            
            # Remove too short meaningless fragments
            if len(result.strip()) < 10:
                return ""
            
            return result
            
        except Exception as e:
            logger.error(f"💥 Failed to extract readable text: {str(e)}")
            return text

    def parse(self, file_path: str):
        """
        Main method for parsing WPS files
        
        Args:
            file_path: WPS file path
            
        Returns:
            MarkdownOutputVo: Object containing parsing results
        """
        logger.info(f"🚀 Starting WPS file parsing: {file_path}")

        # Validate file existence
        if not os.path.exists(file_path):
            logger.error(f"❌ WPS file does not exist: {file_path}")
            raise FileNotFoundError(f"WPS file does not exist: {file_path}")

        # Validate file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"❌ WPS file is empty: {file_path}")
            raise ValueError(f"WPS file is empty: {file_path}")

        logger.info(f"📊 WPS file information - Size: {file_size} bytes")

        # Add start processing lifecycle record
        processing_lifecycle = self.generate_lifecycle(
            source_file=file_path,
            domain="office",
            life_type=LifeType.DATA_PROCESSING,
            usage_purpose="Starting WPS document parsing"
        )

        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"📁 Created temporary directory: {temp_dir}")

                content = ""
                extraction_method = ""

                try:
                    # Method 1: Use LibreOffice conversion
                    logger.info("🔄 Attempting LibreOffice format conversion...")
                    txt_path = self.wps_to_txt(file_path, temp_dir)
                    content = self.read_txt_file(txt_path)
                    extraction_method = "LibreOffice conversion"
                    
                    # Validate conversion result quality
                    if len(content.strip()) < 10:
                        logger.warning("⚠️ LibreOffice conversion result has too little content, trying other methods")
                        content = ""
                        
                except Exception as e:
                    logger.warning(f"⚠️ LibreOffice conversion failed: {str(e)}")
                    content = ""

                # Method 2: Direct reading (as backup solution)
                if not content:
                    logger.info("🔄 Attempting to directly read WPS file content...")
                    content = self.read_wps_file(file_path)
                    if content:
                        extraction_method = "Direct reading"

                # If still no content, return error information
                if not content:
                    error_msg = (
                        "Unable to extract WPS file content. Possible reasons:\n"
                        "1. WPS file format is not supported\n"
                        "2. File is corrupted\n"
                        "3. LibreOffice version does not support this WPS format\n"
                        "Suggestion: Use WPS Office to save the file as .doc or .docx format"
                    )
                    logger.error(f"❌ {error_msg}")
                    content = error_msg
                    extraction_method = "Failed"

                logger.info(f"✅ WPS file parsing completed - Extraction method: {extraction_method}, Content length: {len(content)} characters")

                # Create return object
                file_extension = self.get_file_extension(file_path)
                result = MarkdownOutputVo(file_extension, content)
                
                # Add lifecycle information
                result.add_lifecycle(processing_lifecycle)  # Add start processing lifecycle
                
                lifecycle = self.generate_lifecycle(
                    source_file=file_path,
                    domain="office",
                    life_type=LifeType.DATA_PROCESSED,
                    usage_purpose=f"WPS document parsing - {extraction_method}"
                )
                result.add_lifecycle(lifecycle)

                # If need to convert to Markdown format
                if self.to_markdown and extraction_method != "Failed":
                    logger.info("🔄 Converting content to Markdown format...")
                    markdown_content = self.format_as_markdown(content)
                    result.content = markdown_content
                    
                    # Add Markdown conversion lifecycle
                    markdown_lifecycle = self.generate_lifecycle(
                        source_file=file_path,
                        domain="markdown",
                        life_type=LifeType.DATA_PROCESSED,
                        usage_purpose="WPS content Markdown formatting"
                    )
                    result.add_lifecycle(markdown_lifecycle)

                return result

        except Exception as e:
            logger.error(f"💥 Serious error occurred during WPS file parsing: {str(e)}")
            
            # Create error return object
            file_extension = self.get_file_extension(file_path)
            error_content = f"WPS file parsing failed: {str(e)}"
            result = MarkdownOutputVo(file_extension, error_content)
            
            # Add lifecycle information
            result.add_lifecycle(processing_lifecycle)  # Add start processing lifecycle
            
            # Add error lifecycle
            error_lifecycle = self.generate_lifecycle(
                source_file=file_path,
                domain="error",
                life_type=LifeType.DATA_PROCESS_FAILED,
                usage_purpose=f"WPS parsing error: {str(e)}"
            )
            result.add_lifecycle(error_lifecycle)
            
            return result

    def format_as_markdown(self, content: str) -> str:
        """Format WPS content as Markdown"""
        if not content or content.startswith("Unable to extract"):
            return content

        try:
            logger.info("📝 Starting to format WPS content as Markdown...")

            # Basic Markdown formatting
            lines = content.split('\n')
            markdown_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    markdown_lines.append("")
                    continue
                
                # Simple title detection (shorter lines that don't end with punctuation)
                if len(line) < 50 and not line.endswith(('。', '.', '！', '!', '？', '?')):
                    # Possibly a title
                    if len(line) < 20:
                        markdown_lines.append(f"## {line}")
                    else:
                        markdown_lines.append(f"### {line}")
                else:
                    # Regular paragraph
                    markdown_lines.append(line)
                    
            markdown_content = '\n\n'.join(markdown_lines)
            
            # Clean up excessive blank lines
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
            
            logger.info("✅ Markdown formatting completed")
            return markdown_content.strip()
            
        except Exception as e:
            logger.error(f"💥 Markdown formatting failed: {str(e)}")
            return content
