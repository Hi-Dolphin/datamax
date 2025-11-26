# DataMax

<div align="center">

<img src="logo.png" alt="DataMax logo" width="140" />

[中文](README_zh.md) | **English** | [中文文档](https://hi-dolphin.github.io/datamax) | [最佳实践](https://github.com/Hi-Dolphin/datamax/blob/main/examples/scripts/generate_qa.py)

[![PyPI version](https://badge.fury.io/py/pydatamax.svg)](https://badge.fury.io/py/pydatamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

A powerful multi-format file parsing, data cleaning, and AI annotation toolkit built for modern Python applications.

## ✨ Key Features

- 🔁 **Full QA Pipeline**: Single-script automation chains parsing, QA generation, and quality evaluation, so datasets are curated end-to-end without manual orchestration.
- 🔄 **Multi-format Support**: Unified loaders handle PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, HTML, EPUB, TXT, and mainstream image formats without extra plugins.
- 🧹 **Intelligent Cleaning**: Built-in anomaly detection, privacy-aware redaction, and customizable filters normalize noisy enterprise documents.
- 🤖 **AI Annotation**: LLM-powered workflows auto-generate Q&A pairs, summaries, and structured labels for downstream model training.
- ⚡ **High Performance**: Streaming chunkers, caching, and parallel execution keep large batch jobs fast and resource-efficient.
- 🎯 **Developer Friendly**: Type-hinted SDK with declarative configs, pluggable pipelines, and rich error handling simplifies integration.
- ☁️ **Cloud Ready**: Native connectors for OSS, MinIO, and S3-compatible storage make hybrid or fully managed deployments straightforward.

## 🚀 Quick Start

### Install

```bash
pip install pydatamax
```

### Examples

```python
from datamax import DataMax

# prepare info
FILE_PATHS = ["/your/file/path/1.md", "/your/file/path/2.doc", "/your/file/path/3.xlsx"]
LABEL_LLM_API_KEY = "YOUR_API_KEY"
LABEL_LLM_BASE_URL = "YOUR_BASE_URL"
LABEL_LLM_MODEL_NAME = "YOUR_MODEL_NAME"
LLM_TRAIN_OUTPUT_FILE_NAME = "train"

# init client
client = DataMax(file_path=FILE_PATHS)

# get data
data = dm.get_data()

# get content
content = data.get("content")

# get pre label. return trainable qa list
qa = dm.get_pre_label(
    content=content,
    api_key=api_key,
    base_url=base_url,
    model_name=model,
    question_number=50,  # question_number_per_chunk
    max_qps=100.0,
    debug=False,
    structured_data=True,  # enable structured output
    auto_self_review_mode=True,  # auto review qa, pass with 4 and 5 score, drop with 1, 2 and 3 score.
    review_max_qps=100.0,
)


# save label data
client.save_label_data(qa_list, LLM_TRAIN_OUTPUT_FILE_NAME)
```

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact Us

- 📧 Email: cy.kron@foxmail.com, wang.xiangyuxy@outlook.com
- 🐛 Issues: [GitHub Issues](https://github.com/Hi-Dolphin/datamax/issues)
- 📚 Best Practice: [How to generate qa](https://github.com/Hi-Dolphin/datamax/blob/main/examples/scripts/generate_qa.py)
- 💬 Wechat Group: <br><img src='wechat.jpg' width=300>
---

⭐ If this project helps you, please give us a star!

