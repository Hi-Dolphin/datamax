# DataMax

<div align="center">

[中文](README_zh.md) | **English**

[![PyPI version](https://badge.fury.io/py/pydatamax.svg)](https://badge.fury.io/py/pydatamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

A powerful multi-format file parsing, data cleaning, and AI annotation toolkit.

## ✨ Core Features

- 🔄 **Multi-format Support**: PDF, DOCX/DOC, PPT/PPTX, XLS/XLSX, HTML, EPUB, TXT, images, and more
- 🧹 **Intelligent Cleaning**: Three-layer cleaning process with anomaly detection, privacy protection, and text filtering
- 🤖 **AI Annotation**: LLM-based automatic data annotation and pre-labeling
- ⚡ **Batch Processing**: Efficient multi-file parallel processing
- 🎯 **Easy Integration**: Clean API design, ready to use out of the box

## 🚀 Quick Start

### Installation

```bash
pip install pydatamax
```

### Basic Usage

```python
from datamax import DataMax

# Parse a single file
dm = DataMax(file_path="document.pdf")
data = dm.get_data()

# Batch processing
dm = DataMax(file_path=["file1.docx", "file2.pdf"])
data = dm.get_data()

# Data cleaning
cleaned_data = dm.clean_data(method_list=["abnormal", "private", "filter"])

# AI annotation
qa_data = dm.get_pre_label(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model_name="gpt-3.5-turbo"
)
```

## 📖 Detailed Documentation

### File Parsing

#### Supported Formats

| Format | Extensions | Special Features |
|--------|------------|------------------|
| Documents | `.pdf`, `.docx`, `.doc` | OCR support, Markdown conversion |
| Spreadsheets | `.xlsx`, `.xls` | Structured data extraction |
| Presentations | `.pptx`, `.ppt` | Slide content extraction |
| Web | `.html`, `.epub` | Tag parsing |
| Images | `.jpg`, `.png`, `.jpeg` | OCR text recognition |
| Text | `.txt` | Automatic encoding detection |

#### Advanced Features

```python
# Advanced PDF parsing (requires MinerU)
dm = DataMax(file_path="complex.pdf", use_mineru=True)

# Word to Markdown conversion
dm = DataMax(file_path="document.docx", to_markdown=True)

# Image OCR
dm = DataMax(file_path="image.jpg", use_ocr=True)
```

### Data Cleaning

```python
# Three cleaning modes
dm.clean_data(method_list=[
    "abnormal",  # Anomaly data processing
    "private",   # Privacy information masking
    "filter"     # Text filtering and normalization
])
```

### AI Annotation

```python
# Custom annotation tasks
qa_data = dm.get_pre_label(
    api_key="sk-xxx",
    base_url="https://api.provider.com/v1",
    model_name="model-name",
    chunk_size=500,        # Text chunk size
    chunk_overlap=100,     # Overlap length
    question_number=5,     # Questions per chunk
    max_workers=5          # Concurrency
)
```

## ⚙️ Environment Setup

### Optional Dependencies

#### LibreOffice (DOC file support)

**Ubuntu/Debian:**
```bash
sudo apt-get install libreoffice
```

**Windows:**
1. Download and install [LibreOffice](https://www.libreoffice.org/download/)
2. Add to environment variables: `C:\Program Files\LibreOffice\program`

#### MinerU (Advanced PDF parsing)

```bash
# Create virtual environment
conda create -n mineru python=3.10
conda activate mineru

# Install MinerU
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
```

For detailed configuration, please refer to [MinerU Documentation](https://github.com/opendatalab/MinerU)

## 🛠️ Development

### Local Installation

```bash
git clone https://github.com/Hi-Dolphin/datamax.git
cd datamax
pip install -r requirements.txt
python setup.py install
```

## 📋 System Requirements

- Python >= 3.10
- Supports Windows, macOS, Linux

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact Us

- 📧 Email: cy.kron@foxmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Hi-Dolphin/datamax/issues)
- 📚 Documentation: [Project Homepage](https://github.com/Hi-Dolphin/datamax)

---

⭐ If this project helps you, please give us a star! 