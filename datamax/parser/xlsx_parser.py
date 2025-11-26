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
            self._validate_file(file_path)
            df = self._read_excel(file_path)
            markdown = self._convert_excel_to_markdown(df)

            logger.info(
                f"🎊 Pandas conversion complete, markdown content length: {len(markdown)} characters"
            )
            logger.debug(f"👀 First 200 characters preview: {markdown[:200]}...")

            return markdown

        except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError):
            raise
        except Exception as e:
            self._handle_pandas_parse_error(e, file_path)
            raise

    def _validate_file(self, file_path: str):
        """Check file existence and size."""
        if not os.path.exists(file_path):
            logger.error(f"🚫 Excel file does not exist: {file_path}")
            raise FileNotFoundError(f"File does not exist: {file_path}")

        file_size = os.path.getsize(file_path)
        logger.info(f"📏 File size: {file_size} bytes")

        if file_size == 0:
            logger.warning(f"⚠️ File size is 0 bytes: {file_path}")
            raise pd.errors.EmptyDataError("Empty file")

    def _read_excel(self, file_path: str):
        """Read Excel file using pandas."""
        try:
            logger.debug("📊 Reading Excel data...")
            return pd.read_excel(file_path, sheet_name=None)
        except PermissionError as e:
            logger.error(f"🔒 File permission error: {str(e)}")
            raise PermissionError(f"No permission to access file: {file_path}")
        except pd.errors.EmptyDataError as e:
            logger.error(f"📭 Excel file is empty: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error reading Excel: {e}")
            raise

    def _convert_excel_to_markdown(self, df) -> str:
        """Convert entire Excel file (single or multiple sheets) to markdown."""
        if isinstance(df, dict):
            return self._convert_multi_sheet(df)
        return self._convert_single_sheet(df)

    def _convert_multi_sheet(self, df_dict: dict) -> str:
        """Convert multiple Excel sheets to markdown."""
        logger.info(f"📑 Detected multiple worksheets, total: {len(df_dict)}")

        markdown = ""
        for sheet_name, sheet_df in df_dict.items():
            markdown += f"## Worksheet: {sheet_name}\n\n"
            markdown += self._convert_sheet(sheet_df)

        return markdown

    def _convert_single_sheet(self, df) -> str:
        """Convert a single Excel sheet to markdown."""
        logger.info(f"📄 Single worksheet, shape: {df.shape}")
        return self._convert_sheet(df)

    def _convert_sheet(self, sheet_df: pd.DataFrame) -> str:
        """Clean and convert a single sheet to markdown."""
        if sheet_df.empty:
            logger.warning("⚠️ Worksheet is empty")
            return "*This worksheet is empty*\n\n"

        # Clean data: remove empty rows/columns
        cleaned = sheet_df.dropna(how="all").dropna(axis=1, how="all")

        if cleaned.empty:
            logger.warning("⚠️ Worksheet has no valid data after cleaning")
            return "*This worksheet has no valid data*\n\n"

        logger.debug(f"📋 Converting cleaned sheet, shape: {cleaned.shape}")
        return cleaned.to_markdown(index=False) + "\n\n"

    def _handle_pandas_parse_error(self, e: Exception, file_path: str):
        """Generate lifecycle and log error."""
        lc_fail = None

        try:
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Documentation",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            logger.debug("⚙️ DATA_PROCESS_FAILED lifecycle generated")
        except Exception as lifecycle_error:
            logger.debug(
                f"Failed to generate DATA_PROCESS_FAILED lifecycle: {lifecycle_error}"
            )

        logger.error(
            f"💀 Excel file parsing failed: {file_path}, error: {str(e)}, lifecycle: {lc_fail}"
        )

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

            logger.info(
                f"🎊 File content parsing complete, final content length: {len(mk_content)} characters"
            )

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
            lc_fail = None
            try:
                lc_fail = self.generate_lifecycle(
                    source_file=file_path,
                    domain=self.domain,
                    usage_purpose="Documentation",
                    life_type=LifeType.DATA_PROCESS_FAILED,
                )
                logger.debug("⚙️ DATA_PROCESS_FAILED lifecycle generated")
            except Exception as lifecycle_error:
                logger.debug(
                    f"Failed to generate DATA_PROCESS_FAILED lifecycle: {lifecycle_error}"
                )

            logger.error(f"💀 Excel file parsing failed: {file_path}, error: {str(e)}")
            error_result = {
                "error": str(e),
                "file_path": file_path,
                "lifecycle": [lc_fail.to_dict()] if lc_fail else [],
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
