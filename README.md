# DataMax

<div align="center">

[中文](README_zh.md) | **English**

[![PyPI version](https://badge.fury.io/py/pydatamax.svg)](https://badge.fury.io/py/pydatamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

A powerful multi-format file parsing, data cleaning, and AI annotation toolkit built for modern Python applications.

## ✨ Key Features

- 🔄 **Multi-format Support**: PDF, DOCX/DOC, PPT/PPTX, XLS/XLSX, HTML, EPUB, TXT, images, and more
- 🧹 **Intelligent Cleaning**: Advanced data cleaning with anomaly detection, privacy protection, and text filtering
- 🤖 **AI Annotation**: LLM-powered automatic annotation and QA generation
- ⚡ **High Performance**: Efficient batch processing with caching and parallel execution
- 🎯 **Developer Friendly**: Modern SDK design with type hints, configuration management, and comprehensive error handling
- ☁️ **Cloud Ready**: Built-in support for OSS, MinIO, and other cloud storage providers

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

# get pre label. return trainable qa list
qa_list = client.get_pre_label(
    api_key=LABEL_LLM_API_KEY,
    base_url=LABEL_LLM_BASE_URL,
    model_name=LABEL_LLM_MODEL_NAME,
    question_number=10,
    max_workers=5)

# save label data
client.save_label_data(qa_list, LLM_TRAIN_OUTPUT_FILE_NAME)
```


## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact Us

- 📧 Email: cy.kron@foxmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Hi-Dolphin/datamax/issues)
- 📚 Documentation: [Project Homepage](https://github.com/Hi-Dolphin/datamax)
- 💬 Wechat Group: <br><img src='wechat.jpg' width=300>
---

⭐ If this project helps you, please give us a star!