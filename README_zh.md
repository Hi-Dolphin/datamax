# 数据工厂

<div align="center">

<img src="logo_zh.png" alt="DataMax logo" width="130" />

**中文** | [English](README.md) | [中文文档](https://hi-dolphin.github.io/datamax) | [最佳实践](https://github.com/Hi-Dolphin/datamax/blob/main/examples/scripts/generate_qa.py)

[![PyPI version](https://badge.fury.io/py/pydatamax.svg)](https://badge.fury.io/py/pydatamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

一款面向现代 Python 应用的多格式文件解析、数据清洗与 AI 标注工具包。

## ✨ 关键特性

- 🔁 **全流程 QA 管线**：单脚本自动化串联解析、QA 生成与质量评估，端到端完成数据集构建无需手动编排。
- 🔄 **多格式支持**：统一加载器可直接处理 PDF、DOC/DOCX、PPT/PPTX、XLS/XLSX、HTML、EPUB、TXT 及主流图片等文件类型，无需额外插件。
- 🧹 **智能清洗**：内置异常检测、隐私脱敏与自定义过滤规则，帮助规范化企业级的噪声文档。
- 🤖 **AI 标注**：借助 LLM 自动生成问答对、摘要与结构化标签，用于下游模型训练。
- ⚡ **高性能**：流式分块、缓存及并行执行让大批量作业保持高效且资源友好。
- 🎯 **开发者友好**：提供类型提示的 SDK、声明式配置、可插拔流水线以及完善的错误处理，集成更省心。
- ☁️ **云就绪**：原生支持 OSS、MinIO 以及兼容 S3 的存储，便于部署到混合或全托管环境。

## 🚀 快速开始

### 安装

```bash
pip install pydatamax
```

### 示例

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

## 🤝 贡献

欢迎通过 Issues 与 Pull Requests 提交改进！

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 📞 联系我们

- 📧 邮箱：cy.kron@foxmail.com, wang.xiangyuxy@outlook.com
- 🐛 Issues: [GitHub Issues](https://github.com/Hi-Dolphin/datamax/issues)
- 📚 最佳实践: [How to generate qa](https://github.com/Hi-Dolphin/datamax/blob/main/examples/scripts/generate_qa.py)
- 💬 微信群：<br><img src='wechat.jpg' width=300>
---

⭐ 如果这个项目对你有帮助，欢迎给我们一个 Star！
