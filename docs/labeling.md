# 标注（文本与多模态）

## 文本 QA 全流程
流程：分块 → 生成领域树（可交互/自定义）→ 生成问题 → 标签匹配 → 生成答案 → JSONL 保存。

### 快速示例
```python
from datamax import DataMax

dm = DataMax(file_path="a.pdf", to_markdown=True, use_mineru=True)
qa = dm.get_pre_label(
    api_key="${DASHSCOPE_API_KEY}",
    base_url="${DASHSCOPE_BASE_URL}",   # 未带 /chat/completions 会自动补全
    model_name="qwen-max",
    question_number=8,
    max_workers=5,
    use_tree_label=True,       # 使用领域树
    interactive_tree=False     # 交互式修订（可选）
)
dm.save_label_data(qa, "train")  # 生成 train.jsonl
```

### 自定义领域树
```python
custom_tree = [
  {"label": "1 概述", "child": [{"label": "1.1 背景"}, {"label": "1.2 术语"}]},
  {"label": "2 方法"}
]

qa = dm.get_pre_label(
  api_key="...", base_url="...", model_name="...",
  use_tree_label=True, interactive_tree=False,
  custom_domain_tree=custom_tree
)
```

### 以纯文本为输入（无需文件）
```python
from datamax.generator import full_qa_labeling_process

result = full_qa_labeling_process(
  content="你的长文本...",
  api_key="...", base_url="...", model_name="qwen-max",
  chunk_size=500, chunk_overlap=100,
  question_number=6, max_workers=5,
  use_tree_label=True, debug=False
)
qa_pairs = result.get("qa_pairs", [])
```

## 多模态 QA（Markdown + 图片）
当你的 Markdown 中包含图片（`![]()`），可生成图文对话式 QA。

```python
from datamax.generator import generate_multimodal_qa_pairs

qa = generate_multimodal_qa_pairs(
  file_path="with_images.md",
  api_key="${OPENAI_API_KEY}",
  model_name="gpt-4o",
  question_number=2,
  max_workers=5
)
```

## 接口说明（要点）
- 兼容 OpenAI `/chat/completions` 接口；`base_url` 未含该路径时会自动补全
- `question_number` 控制每段生成问题数量；并发通过 `max_workers` 控制
- `save_label_data` 输出 `*.jsonl`，可直接用于训练

## 示例脚本

### 文本 QA
--8<-- "examples/scripts/generate_qa.py"

### 多模态 QA
--8<-- "examples/scripts/generate_multimodal_qa.py"
