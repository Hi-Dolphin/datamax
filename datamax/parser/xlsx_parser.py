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
    """XLSX Parser - Uses pandas to read and convert to markdown, supports multi-process handling"""

    def __init__(self, file_path, domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path
        logger.info(f"🚀 XlsxParser initialization complete - File path: {file_path}")

    def _parse_with_pandas(self, file_path: str) -> str:
        """Use pandas to read Excel and convert to markdown"""
        logger.info(f"🐼 Start reading Excel file with pandas: {file_path}")

        try:
            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"🚫 Excel file does not exist: {file_path}")
                raise FileNotFoundError(f"File does not exist: {file_path}")

            # Verify file size
            file_size = os.path.getsize(file_path)
            logger.info(f"📏 File size: {file_size} bytes")

            if file_size == 0:
                logger.warning(f"⚠️ File size is 0 bytes: {file_path}")
                return "*File is empty*"

            # Use pandas to read Excel file
            logger.debug("📊 Reading Excel data...")
            df = pd.read_excel(file_path, sheet_name=None)  # Read all worksheets

            markdown_content = ""

            if isinstance(df, dict):
                # Multiple worksheets
                logger.info(f"📑 Detected multiple worksheets, total: {len(df)}")
                for sheet_name, sheet_df in df.items():
                    logger.debug(f"📋 Processing worksheet: {sheet_name}, shape: {sheet_df.shape}")
                    markdown_content += f"## Worksheet: {sheet_name}\n\n"

                    if not sheet_df.empty:
                        # Clean data: remove completely empty rows and columns
                        sheet_df = sheet_df.dropna(how="all").dropna(axis=1, how="all")

                        if not sheet_df.empty:
                            sheet_markdown = sheet_df.to_markdown(index=False)
                            markdown_content += sheet_markdown + "\n\n"
                            logger.debug(
                                f"✅ Worksheet {sheet_name} conversion complete, valid data shape: {sheet_df.shape}"
                            )
                        else:
                            markdown_content += "*This worksheet has no valid data*\n\n"
                            logger.warning(f"⚠️ Worksheet {sheet_name} has no valid data after cleaning")
                    else:
                        markdown_content += "*This worksheet is empty*\n\n"
                        logger.warning(f"⚠️ Worksheet {sheet_name} is empty")
            else:
                # Single worksheet
                logger.info(f"📄 Single worksheet, shape: {df.shape}")
                if not df.empty:
                    # Clean data: remove completely empty rows and columns
                    df = df.dropna(how="all").dropna(axis=1, how="all")

                    if not df.empty:
                        markdown_content = df.to_markdown(index=False)
                        logger.info(f"✅ Worksheet conversion complete, valid data shape: {df.shape}")
                    else:
                        markdown_content = "*Worksheet has no valid data*"
                        logger.warning("⚠️ Worksheet has no valid data after cleaning")
                else:
                    markdown_content = "*Worksheet is empty*"
                    logger.warning("⚠️ Worksheet is empty")

            logger.info(
                f"🎊 Pandas conversion complete, markdown content length: {len(markdown_content)} characters"
            )
            logger.debug(f"👀 First 200 characters preview: {markdown_content[:200]}...")

            return markdown_content

        except FileNotFoundError as e:
            logger.error(f"🚫 File not found: {str(e)}")
            raise
        except PermissionError as e:
            logger.error(f"🔒 File permission error: {str(e)}")
            raise Exception(f"No permission to access file: {file_path}")
        except pd.errors.EmptyDataError as e:
            logger.error(f"📭 Excel file is empty: {str(e)}")
            raise Exception(f"Excel file is empty or cannot be read: {file_path}")
        except Exception as e:
            logger.error(f"💥 Pandas Excel reading failed: {str(e)}")
            raise

    def _parse(self, file_path: str, result_queue: Queue) -> dict:
        """Core method for parsing Excel files"""
        logger.info(f"🎬 Start parsing Excel file: {file_path}")

        # —— Lifecycle: Start processing —— #
        lc_start = self.generate_lifecycle(
            source_file=file_path,
            domain=self.domain,
            usage_purpose="Documentation",
            life_type=LifeType.DATA_PROCESSING,
        )
        logger.debug("⚙️ DATA_PROCESSING lifecycle generated")

        try:
            # Parse Excel using pandas
            logger.info("🐼 Parsing Excel using pandas mode")
            mk_content = self._parse_with_pandas(file_path)

            # Check if content is empty
            if not mk_content.strip():
                logger.warning(f"⚠️ Parsed content is empty: {file_path}")
                mk_content = "*Unable to parse file content*"

            logger.info(f"🎊 File content parsing complete, final content length: {len(mk_content)} characters")

            # —— Lifecycle: Processing complete —— #
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESSED,
            )
            logger.debug("⚙️ DATA_PROCESSED lifecycle generated")

            # Create output object and add both lifecycles
            extension = self.get_file_extension(file_path)
            output_vo = MarkdownOutputVo(extension, mk_content)
            output_vo.add_lifecycle(lc_start)
            output_vo.add_lifecycle(lc_end)

            result = output_vo.to_dict()
            result_queue.put(result)
            logger.info(f"🏆 Excel file parsing complete: {file_path}")
            logger.debug(f"🔑 Return result keys: {list(result.keys())}")

            time.sleep(0.5)  # Give queue some time
            return result

        except Exception as e:
            # —— Lifecycle: Processing failed —— #
            try:
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    usage_purpose="Documentation",
                    life_type=LifeType.DATA_PROCESS_FAILED,
                )
                logger.debug("⚙️ DATA_PROCESS_FAILED lifecycle generated")
                # If needed, this can also be added to error_result:
                # error_result = {"error": str(e), "file_path": file_path, "lifecycle":[lc_fail.to_dict()]}
            except Exception:
                pass

            # —— Lifecycle: Processing failed —— #
            try:
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    usage_purpose="Documentation",
                    life_type=LifeType.DATA_PROCESS_FAILED,
                )
                logger.debug("⚙️ DATA_PROCESS_FAILED lifecycle generated")
            except Exception:
                pass

            logger.error(f"💀 Excel file parsing failed: {file_path}, error: {str(e)}")
            # Put error into queue as well
            error_result = {
                "error": str(e),
                "file_path": file_path,
                # Also return the failed lifecycle for optional verification in tests
                "lifecycle": [lc_fail.to_dict()] if "lc_fail" in locals() else [],
            }
            result_queue.put(error_result)
            raise

    def parse(self, file_path: str) -> dict:
        """Parse Excel file - supports multi-process and timeout control"""
        logger.info(f"🚀 Starting Excel parsing process - File: {file_path}")

        try:
            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"🚫 File does not exist: {file_path}")
                raise FileNotFoundError(f"File does not exist: {file_path}")

            # Verify file extension
            if not file_path.lower().endswith((".xlsx", ".xls")):
                logger.warning(f"⚠️ File extension is not Excel format: {file_path}")

            result_queue = Queue()
            process = multiprocessing.Process(
                target=self._parse, args=(file_path, result_queue)
            )
            process.start()
            logger.debug(f"⚡ Started subprocess, PID: {process.pid}")

        except Exception as e:
            logger.error(
                f"💀 Excel parsing failed: {file_path}, error type: {type(e).__name__}, error message: {str(e)}"
            )
            raise
