import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from datamax.utils.tokenizer import DashScopeClient


class LifeCycle:
    """
    Life cycle class
    """

    def __init__(self, update_time: str, life_type: list, life_metadata: Dict[str, str]):
        self.update_time = update_time  # Update time
        self.life_type = life_type  # Life cycle type
        self.life_metadata = life_metadata  # Life cycle metadata

    def update(self, update_time: str, life_type: list, life_metadata: Dict[str, str]):
        self.update_time = update_time
        self.life_type = life_type
        self.life_metadata.update(life_metadata)

    def __str__(self):
        metadata_str = ', '.join(f'{k}: {v}' for k, v in self.life_metadata.items())
        return f'update_time: {self.update_time}, life_type: {self.life_type}, life_metadata: {{{metadata_str}}}'

    def to_dict(self):
        return {
            'update_time': self.update_time,
            'life_type': self.life_type,
            'life_metadata': self.life_metadata
        }


class MarkdownOutputVo:
    """
    Markdown output conversion
    """

    def __init__(self, extension: str, content: str):
        self.extension: str = extension  # File type
        self.content: str = content  # Markdown content
        self.lifecycle: List[LifeCycle] = []  # Life cycle data

    def add_lifecycle(self, lifecycle: LifeCycle):
        self.lifecycle.append(lifecycle)

    def to_dict(self):
        data_dict = {
            'extension': self.extension,
            'content': self.content,
            'lifecycle': [lc.to_dict() for lc in self.lifecycle]
        }
        return data_dict


class BaseLife:
    tk_client = DashScopeClient()

    @staticmethod
    def generate_lifecycle(source_file, domain, life_type, usage_purpose) -> LifeCycle:
        update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        life_type = [life_type]
        storage = os.stat(source_file)
        life_metadata = {
            # "token_count": token_count,  # Token count of the text
            "storage_size": storage.st_size,  # Storage size in bytes
            "source_file": source_file,  # Source file
            "domain": domain,  # Domain
            "usage_purpose": usage_purpose  # Usage purpose
        }
        return LifeCycle(update_time, life_type, life_metadata)

    @staticmethod
    def get_file_extension(file_path):
        file_path = Path(file_path)
        return file_path.suffix[1:].lower()