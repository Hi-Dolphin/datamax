"""Main client interface for DataMax SDK."""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from datamax.config import DataMaxSettings, get_settings, configure
from datamax.exceptions import (
    ConfigurationError,
    UnsupportedFormatError,
    ParseError,
)
from datamax.parser.core import DataMax as _DataMaxCore
from datamax.loader.core import DataLoader


class DataMaxClient:
    """Main client for DataMax SDK operations.
    
    This is the recommended entry point for all DataMax operations.
    
    Example:
        # Basic usage
        client = DataMaxClient()
        result = client.parse_file("document.pdf")
        
        # With configuration
        client = DataMaxClient(
            ai_config={
                "api_key": "your-key",
                "base_url": "https://api.openai.com/v1"
            }
        )
        
        # Parse and annotate
        result = client.parse_file("document.pdf")
        qa_data = client.annotate(result["content"])
    """

    def __init__(
        self,
        config: DataMaxSettings | None = None,
        config_path: str | Path | None = None,
        domain: str = "Technology",
        **kwargs: Any,
    ):
        """Initialize DataMax client.
        
        Args:
            config: Pre-configured settings object
            config_path: Path to configuration file  
            domain: Default domain for processing
            **kwargs: Additional configuration options
        """
        if config:
            self.settings = config
        elif config_path:
            self.settings = configure(config_path=config_path)
        else:
            # Use kwargs to override defaults
            config_dict = {"domain": domain}
            config_dict.update(kwargs)
            self.settings = configure(**config_dict)
        
        self._parser_cache = {}
        self._data_loader = None

    @property
    def data_loader(self) -> DataLoader:
        """Get or create data loader instance."""
        if self._data_loader is None:
            if self.settings.storage is None:
                raise ConfigurationError("Storage configuration required for data loading")
            
            storage = self.settings.storage
            self._data_loader = DataLoader(
                endpoint=storage.endpoint,
                secret_key=storage.secret_key,
                access_key=storage.access_key,
                bucket_name=storage.bucket_name,
                source=storage.provider,
            )
        return self._data_loader

    def parse_file(
        self,
        file_path: str | Path,
        use_mineru: bool | None = None,
        to_markdown: bool | None = None,
        ttl_cache: int | None = None,
    ) -> dict[str, Any]:
        """Parse a single file.
        
        Args:
            file_path: Path to file to parse
            use_mineru: Override default MinerU setting
            to_markdown: Override default markdown setting  
            ttl_cache: Override default cache TTL
            
        Returns:
            Parsed file data with content and metadata
            
        Raises:
            ParseError: If parsing fails
            UnsupportedFormatError: If file format not supported
        """
        file_path = str(file_path)
        
        try:
            parser = _DataMaxCore(
                file_path=file_path,
                use_mineru=use_mineru or self.settings.parse.use_mineru,
                to_markdown=to_markdown or self.settings.parse.to_markdown,
                ttl=ttl_cache or self.settings.parse.ttl_cache,
                domain=self.settings.domain,
            )
            
            result = parser.get_data()
            if result is None:
                raise UnsupportedFormatError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Successfully parsed file: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            raise ParseError(f"Failed to parse {file_path}: {e}") from e

    def parse_files(
        self,
        file_paths: list[str | Path],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Parse multiple files.
        
        Args:
            file_paths: List of file paths to parse
            **kwargs: Arguments passed to parse_file
            
        Returns:
            List of parsed file data
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.parse_file(file_path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {e}")
                continue
        
        return results

    def parse_directory(
        self,
        directory_path: str | Path,
        pattern: str = "*",
        recursive: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Parse all files in a directory.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern to match (e.g., "*.pdf")
            recursive: Whether to search recursively
            **kwargs: Arguments passed to parse_file
            
        Returns:
            List of parsed file data
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))
        
        file_paths = [f for f in files if f.is_file()]
        logger.info(f"Found {len(file_paths)} files to parse in {directory_path}")
        
        return self.parse_files(file_paths, **kwargs)

    def clean_data(
        self,
        content: str | dict[str, Any],
        methods: list[str] | None = None,
    ) -> str | dict[str, Any]:
        """Clean parsed data.
        
        Args:
            content: Content to clean (string or parsed data dict)
            methods: Cleaning methods ["abnormal", "filter", "private"]
            
        Returns:
            Cleaned content
        """
        if methods is None:
            methods = ["abnormal", "filter"]
        
        if isinstance(content, dict):
            text = content.get("content", "")
            file_path = content.get("file_path", "unknown")
        else:
            text = content
            file_path = "string_input"
        
        parser = _DataMaxCore(
            file_path=file_path,
            domain=self.settings.domain,
        )
        parser.parsed_data = content if isinstance(content, dict) else {"content": content}
        
        result = parser.clean_data(method_list=methods, text=text)
        logger.info(f"Successfully cleaned data using methods: {methods}")
        
        return result

    def annotate(
        self,
        content: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate AI annotations for content.
        
        Args:
            content: Text content to annotate
            api_key: Override API key
            base_url: Override base URL
            model_name: Override model name
            **kwargs: Additional parameters for annotation
            
        Returns:
            List of QA pairs with annotations
        """
        # Validate AI config
        if api_key or base_url or model_name:
            # Use provided parameters
            ai_config = {
                "api_key": api_key or (self.settings.ai.api_key if self.settings.ai else ""),
                "base_url": base_url or (self.settings.ai.base_url if self.settings.ai else ""),
                "model_name": model_name or (self.settings.ai.model_name if self.settings.ai else "gpt-3.5-turbo"),
            }
        else:
            # Use settings
            self.settings.validate_ai_config()
            ai_config = {
                "api_key": self.settings.ai.api_key,
                "base_url": self.settings.ai.base_url,
                "model_name": self.settings.ai.model_name,
            }
        
        # Set defaults from settings for other parameters
        annotation_params = {
            "chunk_size": self.settings.parse.chunk_size,
            "chunk_overlap": self.settings.parse.chunk_overlap,
            "question_number": (self.settings.ai.question_number if self.settings.ai else 5),
            "max_workers": (self.settings.ai.max_workers if self.settings.ai else 5),
            "language": (self.settings.ai.language if self.settings.ai else "zh"),
        }
        annotation_params.update(kwargs)
        
        parser = _DataMaxCore(
            file_path="string_input",
            domain=self.settings.domain,
        )
        
        result = parser.get_pre_label(
            content=content,
            **ai_config,
            **annotation_params,
        )
        
        logger.info(f"Successfully generated {len(result)} QA pairs")
        return result

    def load_from_cloud(
        self,
        remote_path: str,
        local_path: str | None = None,
    ) -> str:
        """Load file from cloud storage.
        
        Args:
            remote_path: Remote file path
            local_path: Local download path (optional)
            
        Returns:
            Local file path
        """
        if local_path is None:
            local_path = f"./downloads/{os.path.basename(remote_path)}"
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        self.data_loader.download_file(remote_path, local_path)
        logger.info(f"Downloaded {remote_path} to {local_path}")
        
        return local_path

    def process_pipeline(
        self,
        file_path: str | Path,
        clean_methods: list[str] | None = None,
        annotate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run complete processing pipeline.
        
        Args:
            file_path: File to process
            clean_methods: Data cleaning methods to apply
            annotate: Whether to generate annotations
            **kwargs: Additional parameters
            
        Returns:
            Complete processing results
        """
        # Parse file
        result = self.parse_file(file_path, **kwargs)
        
        # Clean data if requested
        if clean_methods:
            result = self.clean_data(result, methods=clean_methods)
        
        # Generate annotations if requested
        if annotate:
            content = result.get("content", "") if isinstance(result, dict) else result
            annotations = self.annotate(content, **kwargs)
            if isinstance(result, dict):
                result["annotations"] = annotations
            else:
                result = {"content": result, "annotations": annotations}
        
        return result