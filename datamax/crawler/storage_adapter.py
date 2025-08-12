"""Storage Adapter for Crawler Data

Provides unified storage interface for crawler data with support for
local file system and cloud storage backends.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from .exceptions import CrawlerException


class StorageAdapter(ABC):
    """Abstract base class for storage adapters.
    
    Provides a unified interface for storing crawler data
    across different storage backends.
    """
    
    @abstractmethod
    async def save(self, data: Dict[str, Any], identifier: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save data to storage.
        
        Args:
            data: Data to save
            identifier: Unique identifier for the data
            metadata: Optional metadata to save with the data
            
        Returns:
            Storage path or identifier where data was saved
        """
        pass
    
    @abstractmethod
    async def load(self, identifier: str) -> Dict[str, Any]:
        """Load data from storage.
        
        Args:
            identifier: Identifier of the data to load
            
        Returns:
            Loaded data dictionary
        """
        pass
    
    @abstractmethod
    async def exists(self, identifier: str) -> bool:
        """Check if data exists in storage.
        
        Args:
            identifier: Identifier to check
            
        Returns:
            True if data exists
        """
        pass
    
    @abstractmethod
    async def delete(self, identifier: str) -> bool:
        """Delete data from storage.
        
        Args:
            identifier: Identifier of data to delete
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def list_items(self, prefix: Optional[str] = None) -> List[str]:
        """List all items in storage.
        
        Args:
            prefix: Optional prefix to filter items
            
        Returns:
            List of item identifiers
        """
        pass


class LocalStorageAdapter(StorageAdapter):
    """Local file system storage adapter.
    
    Stores crawler data as JSON files in the local file system.
    """
    
    def __init__(self, base_path: str = "./output", format: str = "json"):
        """Initialize local storage adapter.
        
        Args:
            base_path: Base directory for storing files
            format: Storage format ('json' or 'yaml')
        """
        self.base_path = Path(base_path)
        self.format = format.lower()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        if self.format not in ['json', 'yaml']:
            raise CrawlerException(f"Unsupported storage format: {format}")
    
    def _get_file_path(self, identifier: str) -> Path:
        """Get file path for an identifier.
        
        Args:
            identifier: Data identifier
            
        Returns:
            Path object for the file
        """
        # Sanitize identifier for file system
        safe_identifier = "".join(c for c in identifier if c.isalnum() or c in ('-', '_', '.'))
        extension = f".{self.format}"
        return self.base_path / f"{safe_identifier}{extension}"
    
    def _get_metadata_path(self, identifier: str) -> Path:
        """Get metadata file path for an identifier.
        
        Args:
            identifier: Data identifier
            
        Returns:
            Path object for the metadata file
        """
        file_path = self._get_file_path(identifier)
        return file_path.with_suffix(f".meta{file_path.suffix}")
    
    async def save(self, data: Dict[str, Any], identifier: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save data to local file system.
        
        Args:
            data: Data to save
            identifier: Unique identifier for the data
            metadata: Optional metadata to save with the data
            
        Returns:
            File path where data was saved
        """
        file_path = self._get_file_path(identifier)
        
        try:
            # Save main data
            if self.format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif self.format == 'yaml':
                import yaml
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            
            # Save metadata if provided
            if metadata:
                metadata_with_timestamp = {
                    **metadata,
                    'saved_at': datetime.now().isoformat(),
                    'identifier': identifier
                }
                
                metadata_path = self._get_metadata_path(identifier)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_with_timestamp, f, indent=2, ensure_ascii=False)
            
            return str(file_path)
            
        except Exception as e:
            raise CrawlerException(f"Failed to save data to {file_path}: {str(e)}") from e
    
    async def load(self, identifier: str) -> Dict[str, Any]:
        """Load data from local file system.
        
        Args:
            identifier: Identifier of the data to load
            
        Returns:
            Loaded data dictionary
        """
        file_path = self._get_file_path(identifier)
        
        if not file_path.exists():
            raise CrawlerException(f"Data not found: {identifier}")
        
        try:
            if self.format == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif self.format == 'yaml':
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
                    
        except Exception as e:
            raise CrawlerException(f"Failed to load data from {file_path}: {str(e)}") from e
    
    async def exists(self, identifier: str) -> bool:
        """Check if data exists in local file system.
        
        Args:
            identifier: Identifier to check
            
        Returns:
            True if data exists
        """
        file_path = self._get_file_path(identifier)
        return file_path.exists()
    
    async def delete(self, identifier: str) -> bool:
        """Delete data from local file system.
        
        Args:
            identifier: Identifier of data to delete
            
        Returns:
            True if deletion was successful
        """
        file_path = self._get_file_path(identifier)
        metadata_path = self._get_metadata_path(identifier)
        
        success = True
        
        try:
            if file_path.exists():
                file_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
                
        except Exception as e:
            success = False
            raise CrawlerException(f"Failed to delete data {identifier}: {str(e)}") from e
        
        return success
    
    async def list_items(self, prefix: Optional[str] = None) -> List[str]:
        """List all items in local storage.
        
        Args:
            prefix: Optional prefix to filter items
            
        Returns:
            List of item identifiers
        """
        items = []
        pattern = f"*.{self.format}"
        
        for file_path in self.base_path.glob(pattern):
            # Skip metadata files
            if file_path.stem.endswith('.meta'):
                continue
                
            identifier = file_path.stem
            
            if prefix is None or identifier.startswith(prefix):
                items.append(identifier)
        
        return sorted(items)


class CloudStorageAdapter(StorageAdapter):
    """Cloud storage adapter (placeholder implementation).
    
    This is a placeholder for future cloud storage implementations
    such as AWS S3, Google Cloud Storage, or Azure Blob Storage.
    """
    
    def __init__(self, provider: str, **kwargs):
        """Initialize cloud storage adapter.
        
        Args:
            provider: Cloud storage provider ('s3', 'gcs', 'azure')
            **kwargs: Provider-specific configuration
        """
        self.provider = provider
        self.config = kwargs
        
        # This is a placeholder - actual implementation would
        # initialize the specific cloud storage client
        raise NotImplementedError(
            f"Cloud storage provider '{provider}' is not yet implemented. "
            "Please use LocalStorageAdapter for now."
        )
    
    async def save(self, data: Dict[str, Any], identifier: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def load(self, identifier: str) -> Dict[str, Any]:
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def exists(self, identifier: str) -> bool:
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def delete(self, identifier: str) -> bool:
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def list_items(self, prefix: Optional[str] = None) -> List[str]:
        raise NotImplementedError("Cloud storage not yet implemented")


def create_storage_adapter(config: Dict[str, Any]) -> StorageAdapter:
    """Factory function to create storage adapter based on configuration.
    
    Args:
        config: Storage configuration dictionary
        
    Returns:
        Configured storage adapter instance
    """
    cloud_config = config.get('cloud_storage', {})
    
    if cloud_config.get('enabled', False):
        provider = cloud_config.get('provider', 's3')
        return CloudStorageAdapter(provider, **cloud_config)
    else:
        output_dir = config.get('output_dir', './output')
        format_type = config.get('default_format', 'json')
        return LocalStorageAdapter(output_dir, format_type)