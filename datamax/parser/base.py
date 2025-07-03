import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

from datamax.utils.lifecycle_types import LifeType
from datamax.utils.tokenizer import DashScopeClient


class LifeCycle:
    """
    Life cycle class
    """

    def __init__(
        self, update_time: str, life_type: list, life_metadata: Dict[str, str]
    ):
        self.update_time = update_time  # Update time
        self.life_type = life_type  # Life cycle type
        self.life_metadata = life_metadata  # Life cycle metadata

    def update(self, update_time: str, life_type: list, life_metadata: Dict[str, str]):
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
        self.lifecycle: List[LifeCycle] = []  # Life cycle data

    def add_lifecycle(self, lifecycle: LifeCycle):
        self.lifecycle.append(lifecycle)

    def to_dict(self):
        data_dict = {
            "extension": self.extension,
            "content": self.content,
            "lifecycle": [lc.to_dict() for lc in self.lifecycle],
        }
        return data_dict

# ========== 新增：预置领域列表 ==========
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
    # … 如有需要可以继续扩展
]

class BaseLife:
    tk_client = DashScopeClient()
    def __init__(self, *, domain: str = "Technology", **kwargs):
        """
        BaseLife 初始化：接收 domain 并做校验／警告，
        其余参数向上层传递（如果有父类的话）。
        """
        # 1) 预置列表校验
        if domain not in PREDEFINED_DOMAINS:
            # 你也可以换成 logger.warning
            print(f"⚠️ 域 “{domain}” 不在预置列表，将按自定义处理。")
        # 2) 保存域
        self.domain = domain

        # 3) 如果有父类 __init__，就把其余参数透传
        super_init = getattr(super(), "__init__", None)
        if callable(super_init):
            super_init(**kwargs)

    @staticmethod
    def generate_lifecycle(
        source_file: str,
        domain: str,
        life_type: Union[LifeType, str, List[Union[LifeType, str]]],
        usage_purpose: str,
    ) -> LifeCycle:
        """
        构造一个 LifeCycle 记录，可以传入单个枚举/字符串或列表混合
        """
        # 1) 先统一成 list
        if isinstance(life_type, (list, tuple)):
            raw = list(life_type)
        else:
            raw = [life_type]

        # 2) 如果是枚举，就取它的 value
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
