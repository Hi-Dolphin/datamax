# DataMax

<div align="center">

**中文** | [English](README.md)

[![PyPI version](https://badge.fury.io/py/pydatamax.svg)](https://badge.fury.io/py/pydatamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

一个强大的多格式文件解析、数据清洗和AI标注工具库。

## ✨ 核心特性

- 🔄 **多格式支持**: PDF、DOCX/DOC、PPT/PPTX、XLS/XLSX、HTML、EPUB、TXT、图片等
- 🧹 **智能清洗**: 异常检测、隐私保护、文本过滤三层清洗流程
- 🤖 **AI标注**: 基于LLM的自动数据标注和预标记
- ⚡ **批量处理**: 高效的多文件并行处理
- 🎯 **易于集成**: 简洁的API设计，开箱即用

## 🚀 快速开始

### 安装

```bash
pip install pydatamax
```

### 基础用法

```python
from datamax import DataMax

# 解析单个文件
dm = DataMax(file_path="document.pdf")
data = dm.get_data()

# 批量处理
dm = DataMax(file_path=["file1.docx", "file2.pdf"])
data = dm.get_data()

# 数据清洗
cleaned_data = dm.clean_data(method_list=["abnormal", "private", "filter"])

# AI标注
qa_data = dm.get_pre_label(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model_name="gpt-3.5-turbo"
)
```

## 📖 详细文档

### 文件解析

#### 支持的格式

| 格式 | 扩展名 | 特殊功能 |
|------|--------|----------|
| 文档 | `.pdf`, `.docx`, `.doc` | OCR支持、Markdown转换 |
| 表格 | `.xlsx`, `.xls` | 结构化数据提取 |
| 演示 | `.pptx`, `.ppt` | 幻灯片内容提取 |
| 网页 | `.html`, `.epub` | 标签解析 |
| 图片 | `.jpg`, `.png`, `.jpeg` | OCR文字识别 |
| 文本 | `.txt` | 编码自动检测 |

#### 高级功能

```python
# PDF高级解析（需要MinerU）
dm = DataMax(file_path="complex.pdf", use_mineru=True)

# Word转Markdown
dm = DataMax(file_path="document.docx", to_markdown=True)

# 图片OCR
dm = DataMax(file_path="image.jpg", use_ocr=True)
```

### 数据清洗

```python
# 三种清洗模式
dm.clean_data(method_list=[
    "abnormal",  # 异常数据处理
    "private",   # 隐私信息脱敏
    "filter"     # 文本过滤规范化
])
```

### AI标注

```python
# 自定义标注任务
qa_data = dm.get_pre_label(
    api_key="sk-xxx",
    base_url="https://api.provider.com/v1",
    model_name="model-name",
    chunk_size=500,        # 文本块大小
    chunk_overlap=100,     # 重叠长度
    question_number=5,     # 每块生成问题数
    max_workers=5          # 并发数
)
```

## ⚙️ 环境配置

### 可选依赖

#### LibreOffice（DOC文件支持）

**Ubuntu/Debian:**
```bash
sudo apt-get install libreoffice
```

**Windows:**
1. 下载安装 [LibreOffice](https://www.libreoffice.org/download/)
2. 添加到环境变量: `C:\Program Files\LibreOffice\program`

#### MinerU（高级PDF解析）

```bash
# 创建虚拟环境
conda create -n mineru python=3.10
conda activate mineru

# 安装MinerU
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
```

详细配置请参考 [MinerU文档](https://github.com/opendatalab/MinerU)

## 🛠️ 开发

### 本地安装

```bash
git clone https://github.com/Hi-Dolphin/datamax.git
cd datamax
pip install -r requirements.txt
python setup.py install
```


## 📋 系统要求

- Python >= 3.10
- 支持 Windows、macOS、Linux

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 📞 联系我们

- 📧 Email: cy.kron@foxmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Hi-Dolphin/datamax/issues)
- 📚 文档: [项目主页](https://github.com/Hi-Dolphin/datamax)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

