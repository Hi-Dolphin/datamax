# DataMax

<div align="center">

**中文** | [English](README.md)

[![PyPI version](https://badge.fury.io/py/datamax.svg)](https://badge.fury.io/py/datamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**项目文档站点：** https://hi-dolphin.github.io/datamax

一个强大的多格式文件解析、数据清洗和AI标注工具包，为现代Python应用而建。

## ✨ 主要功能

- 🔄 **多格式支持**：PDF, DOCX/DOC, PPT/PPTX, XLS/XLSX, HTML, EPUB, TXT, 图像 等

- 🧹 **智能清洗**：高级数据清洗，包括异常检测、隐私保护和文本过滤

- 🤖 **AI标注**：基于LLM的自动标注和QA生成

- ⚡ **高性能**：高效的批处理，带有缓存和并行执行

- 🎯 **开发者友好**：现代SDK设计，带有类型提示、配置管理和全面错误处理

- ☁️ **云就绪**：内置支持OSS、MinIO和其他云存储提供商

## 🚀 快速开始

### 安装

```bash
pip install pydatamax
```

### 示例

```python
from datamax import DataMax

# prepare info
FILE_PATHS = ["/your/file/path/1.pdf", "/your/file/path/2.doc", "/your/file/path/3.xlsx"]
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
    max_qps=5.0)

# save label data
client.save_label_data(qa_list, LLM_TRAIN_OUTPUT_FILE_NAME)
```

## 📚 文档

- 文档入口：`docs/index.md`
- 文档文本模态 QA 生成完整 Pipeline 使用示例请参考 [examples/scripts/generate_qa.py](examples/scripts/generate_qa.py)

## 🤝 贡献

欢迎提出 Issues 和 Pull Requests！

## 📄 许可

本项目基于 [MIT License](LICENSE) 许可。

## 📞 联系我们

- 📧 邮箱: cy.kron@foxmail.com
- 🐛 问题: [GitHub Issues](https://github.com/Hi-Dolphin/datamax/issues)
- 📚 文档: [项目主页](https://github.com/Hi-Dolphin/datamax)
- 💬 微信群: <br><img src='wechat.jpg' width=300>

---

⭐ 如果这个项目对你有帮助，请给我们一个星！
