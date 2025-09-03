import os
from datetime import datetime
from pathlib import Path
from typing import List
from datamax.utils.lifecycle_types import LifeType


class LifeCycle:
    """
    Life cycle class
    """

    def __init__(
        self, update_time: str, life_type: list, life_metadata: dict[str, str]
    ):
        self.update_time = update_time  # Update time
        self.life_type = life_type  # Life cycle type
        self.life_metadata = life_metadata  # Life cycle metadata

    def update(self, update_time: str, life_type: list, life_metadata: dict[str, str]):
        self.update_time = update_time
        self.life_type = life_type
        self.life_metadata.update(life_metadata)

    def __str__(self):
        metadata_str = ", ".join(f"{k}: {v}" for k, v in self.life_metadata.items())
        return f"update_time: {self.update_time}, life_type: {self.life_type}, life_metadata: {{{metadata_str}}}"

    def to_dict(self):
        return {
            "update_time": self.update_time,
            "life_type": self.life_type,
            "life_metadata": self.life_metadata,
        }


class MarkdownOutputVo:
    """
    Markdown output conversion
    """

    def __init__(self, extension: str, content: str):
        self.extension: str = extension  # File type
        self.content: str = content  # Markdown content
        self.lifecycle: list[LifeCycle] = []  # Life cycle data

    def add_lifecycle(self, lifecycle: LifeCycle):
        self.lifecycle.append(lifecycle)

    def to_dict(self):
        data_dict = {
            "extension": self.extension,
            "content": self.content,
            "lifecycle": [lc.to_dict() for lc in self.lifecycle],
        }
        return data_dict

# ========== New: Predefined domain list ==========
PREDEFINED_DOMAINS = [
    "Technology",
    "Finance",
    "Health",
    "Education",
    "Legal",
    "Marketing",
    "Sales",
    "Entertainment",
    "Science",
    # … Can be extended as needed
]


class BaseLife:
    def __init__(self, *, domain: str = "Technology", **kwargs):
        """
        BaseLife initialization: receives domain and performs validation/warning,
        other parameters are passed to the parent class (if any).
        """
        # 1) Predefined list validation
        if domain not in PREDEFINED_DOMAINS:
            # You can also change to logger.warning
            print(f"⚠️ Domain '{domain}' is not in the predefined list, will be handled as custom.")
        # 2) Save domain
        self.domain = domain

        # 3) If there's a parent class __init__, pass through other parameters
        super_init = getattr(super(), "__init__", None)
        if callable(super_init):
            super_init(**kwargs)

    @staticmethod
    def generate_lifecycle(
        source_file: str,
        domain: str,
        life_type: LifeType | str | list[LifeType | str],
        usage_purpose: str,
    ) -> LifeCycle:
        """
        Construct a LifeCycle record, can pass in a single enum/string or a mixed list
        """
        # 1) First unify to list
        if isinstance(life_type, (list, tuple)):
            raw = list(life_type)
        else:
            raw = [life_type]

        # 2) If it's an enum, take its value
        life_list: List[str] = [
            lt.value if isinstance(lt, LifeType) else lt for lt in raw
        ]

        update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            storage = os.path.getsize(source_file)
        except Exception:
            storage = 0
        life_metadata = {
            "storage_size": storage,
            "source_file": source_file,
            "domain": domain,
            "usage_purpose": usage_purpose,
        }
        return LifeCycle(update_time, life_list, life_metadata)

    @staticmethod
    def get_file_extension(file_path):
        file_path = Path(file_path)
        return file_path.suffix[1:].lower()
