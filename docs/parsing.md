# 解析

## 支持格式
- 文档：PDF, DOC/DOCX/WPS, PPT/PPTX, EPUB, HTML, TXT, MD, JSON
- 图片：JPG/PNG/WEBP（可选视觉模型 MLLM）
- 表格：XLS/XLSX/CSV
- 代码：常见主流语言（语义友好分块）

## 统一入口与输出
- 入口：`from datamax import DataMax`
- 输出统一结构：
  ```json
  {
    "extension": "md",
    "content": "...",
    "lifecycle": [ {"update_time": "...", "life_type": ["DATA_PROCESSING"], ...}, ... ]
  }
  ```

## 高级选项
- PDF：
  - `use_mineru=True` → Magic‑PDF 高保真解析（需下载模型）
  - `use_qwen_vl_ocr=True` → Qwen‑VL OCR（DASHSCOPE_* 环境变量）
- 图片：
  - `use_mllm=True, model_name="gpt-4o"` → 视觉模型解析为 Markdown 描述
- 文档转 Markdown：
  - `to_markdown=True`（适用于 doc/docx 等）

## 示例模板

### 单文件解析
```python
from datamax import DataMax

res = DataMax(file_path="a.docx", to_markdown=True).get_data()
print(res["extension"], len(res["content"]))
```

### PDF 高保真/二维码/表格友好（MinerU）
```python
from datamax import DataMax

res = DataMax(file_path="a.pdf", to_markdown=True, use_mineru=True).get_data()
```

### PDF OCR（Qwen‑VL OCR）
```python
from datamax import DataMax

res = DataMax(
    file_path="scan.pdf",
    to_markdown=True,
    use_qwen_vl_ocr=True,
    api_key="${DASHSCOPE_API_KEY}",
    base_url="${DASHSCOPE_BASE_URL}",
    model_name="qwen-vl-max-latest",
).get_data()
```

### 图片 → 视觉模型（MLLM）
```python
from datamax import DataMax

res = DataMax(
    file_path="img.png",
    use_mllm=True,
    mllm_system_prompt="描述图片内容，输出为 Markdown",
    api_key="${OPENAI_API_KEY}",
    base_url="${OPENAI_BASE_URL}",  # 兼容 /chat/completions
    model_name="gpt-4o",
).get_data()
```

## 注意事项
- MLLM 当前仅用于图片解析；`use_mineru` 与 `use_qwen_vl_ocr` 互斥
- UNO 未安装时自动回退到 `soffice` 子进程（建议安装提升稳定性）

## 示例脚本

--8<-- "examples/scripts/parse_file.py"
