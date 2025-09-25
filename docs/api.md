# API 参考（精选）

## 解析与清洗
```python
from datamax import DataMax

dm = DataMax(
  file_path: str | list,
  domain: str = "Technology",
  to_markdown: bool = False,
  use_mineru: bool = False,
  use_qwen_vl_ocr: bool = False,
  use_mllm: bool = False,
  mllm_system_prompt: str | None = None,
  api_key: str | None = None,
  base_url: str | None = None,
  model_name: str | None = None,
)

dm.get_data() -> dict | list[dict]
dm.clean_data(methods: list[str], text: str | None = None) -> dict | str
```

## 标注与生成
```python
dm.get_pre_label(
  api_key: str,
  base_url: str,
  model_name: str,
  question_number: int = 5,
  max_workers: int = 5,
  use_tree_label: bool = False,
  interactive_tree: bool = False,
  messages: list | None = None,
) -> list | dict

dm.save_label_data(label_data: list | dict, save_file_name: str = "qa_pairs") -> None

from datamax.generator import full_qa_labeling_process, generate_multimodal_qa_pairs
```

## 爬虫
```python
from datamax.crawler import crawl

crawl(keyword_or_url: str, engine: str = "auto" | "web" | "arxiv") -> dict

from datamax.parser import CrawlerParser
CrawlerParser(file_path: str).parse() -> MarkdownOutputVo
```

## 清洗（独立类）
```python
from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization
```

## 评估
```python
from datamax.evaluator import TextQualityEvaluator, MultimodalConsistencyEvaluator
```

## 输出结构
```json
{
  "extension": "md",
  "content": "...",
  "lifecycle": [
    {
      "update_time": "2025-01-01 12:00:00",
      "life_type": ["DATA_PROCESSING"],
      "life_metadata": {"source_file": "...", "domain": "...", "usage_purpose": "..."}
    }
  ]
}
```

