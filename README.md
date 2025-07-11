# DataMax

<div align="center">

[中文](README_zh.md) | **English**

[![PyPI version](https://badge.fury.io/py/datamax.svg)](https://badge.fury.io/py/datamax) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

### Installation

```bash
# Install the latest version
pip install datamax

# With optional dependencies
pip install datamax[mineru,ocr,dev]
```

### Basic Usage

```python
import datamax

# Simple file parsing
client = datamax.DataMaxClient()
result = client.parse_file("document.pdf")
print(result["content"])

# Batch processing
results = client.parse_files(["doc1.pdf", "doc2.docx", "doc3.txt"])

# Parse entire directory
results = client.parse_directory("./documents", pattern="*.pdf")

# With configuration
client = datamax.DataMaxClient(
    domain="Legal",
    parse={"use_mineru": True, "chunk_size": 1000},
    ai={"api_key": "your-key", "model_name": "gpt-4"}
)
```

### Advanced Usage

```python
import datamax

# Configure globally
datamax.configure(
    domain="Finance",
    ai={
        "api_key": "your-openai-key",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4"
    }
)

# Complete pipeline: parse -> clean -> annotate
client = datamax.DataMaxClient()
result = client.process_pipeline(
    file_path="financial_report.pdf",
    clean_methods=["abnormal", "private", "filter"],
    annotate=True,
    question_number=5
)

# Data cleaning
cleaned = client.clean_data(
    content="Raw text with noise and PII data...",
    methods=["abnormal", "filter", "private"]
)

# AI annotation
annotations = client.annotate(
    content="Document content for annotation...",
    question_number=10,
    language="en"
)
    api_key="sk-xxx",
    base_url="https://api.provider.com/v1",
    model_name="model-name",
    chunk_size=500,        # 文本块大小
    chunk_overlap=100,     # 重叠长度
    question_number=5,     # 每块生成问题数
    max_workers=5          # 并发数
)
dm.save_label_data(qa_data)
```

## 📖 Detailed Documentation

### File Parsing

#### 可选参数：domain
All parsers support an optional domain: str parameter for specifying the business domain, with "Technology" set as the default value.
Predefined domain options include：["Technology","Finance","Health","Education","Legal","Marketing","Sales","Entertainment","Science"]，Custom strings can also be passed as needed.

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
### Batch Processing
```python
# Parse multiple files in batch
dm = DataMax(
    file_path=["file1.pdf", "file2.docx"],
    use_mineru=True
)
data = dm.get_data()
```

### Cache parsed results
```python
# Cache parsed results to avoid repeated parsing
dm = DataMax(
    file_path=["file1.pdf", "file2.docx"],
    ttl=3600  # Cache duration in seconds, default 3600s, 0 means no caching
)
data = dm.get_data()
```

### Data Cleaning
## Exception Handling

- remove_abnormal_chars Remove abnormal characters from text
- remove_html_tags Remove HTML tags
- convert_newlines Convert \r to \n and merge multiple \n into single \n
- single_space Convert multiple spaces (more than 2) to single space
- tabs_to_spaces Convert tabs to 4 spaces
- remove_invisible_chars Remove invisible ASCII characters
- simplify_chinese Convert traditional Chinese to simplified Chinese

## Text Filtering

- filter_by_word_repetition Filter by word repetition rate
- filter_by_char_count Filter by character count
- filter_by_numeric_content Filter by numeric content ratio

## Privacy Desensitization

- replace_ip
- replace_email
- replace_customer_number Clean hotline numbers like 4008-123-123
- replace_bank_id
- replace_phone_number
- replace_qq
- replace_id_card



```python
# Three cleaning modes
dm.clean_data(method_list=[
    "abnormal",  # Anomaly data processing
    "private",   # Privacy information masking
    "filter"     # Text filtering and normalization
])

# Custom cleaning mode
from datamax.utils.data_cleaner import TextFilter, PrivacyDesensitization, AbnormalCleaner
dm = DataMax(
    file_path=r"C:\Users\cykro\Desktop\HongKongDevMachine.txt"
)
parsed_data = dm.get_data().get('content')
# 1. Text filtering
tf = TextFilter(parsed_data=parsed_data)
    # Word repetition filtering - default threshold is 0.6 (max 60% of characters can be repeated)
tf_bool = tf.filter_by_word_repetition(threshold=0.6)
if tf_bool:
    print("Text passed word repetition filtering")
else:
    print("Text failed word repetition filtering")
    
# Character count filtering - default min_chars=30 (minimum 30 chars), max_chars=500000 (maximum 500000 chars)
tf_bool = tf.filter_by_char_count(min_chars=30, max_chars=500000)
if tf_bool:
    print("Text passed character count filtering")
else:
    print("Text failed character count filtering")

# Numeric content filtering - default threshold=0.6 (max 60% of characters can be digits)
tf_bool = tf.filter_by_numeric_content(threshold=0.6)
if tf_bool:
    print("Text passed numeric ratio filtering")
else:
    print("Text failed numeric ratio filtering")

# 2. Privacy desensitization
pd = PrivacyDesensitization(parsed_data=parsed_data)
res = pd.replace_ip(
    token="MyIP"
)
print(res)

# 3. Abnormal character cleaning
ac = AbnormalCleaner(parsed_data=parsed_data)
res = ac.remove_abnormal_chars()
res = ac.remove_html_tags()
res = ac.convert_newlines()
res = ac.single_space()
res = ac.tabs_to_spaces()
res = ac.remove_invisible_chars()
res = ac.simplify_chinese()
print(res)
```
# Text Segmentation
```python
dm.split_data(
    chunk_size=500,      # Chunk size
    chunk_overlap=100,    # Overlap length
    use_langchain=True   # Use LangChain for text segmentation
)

# When use_langchain is False, use custom segmentation method
# Using 。！？ as separators, consecutive separators will be merged
# chunk_size strictly limits the string length
for chunk in parser.split_data(chunk_size=500, chunk_overlap=100, use_langchain=False).get("content"):
    print(chunk)
```

### Enhanced QA Generation

The QA generator now supports:
- User-provided domain tree for custom initialization
- Retry mechanism for LLM calls
- Fallback to text-only mode if domain tree generation fails
- Use of domain tree labels for more accurate annotation
- Interactive domain tree editing for fine-tuning


```python
# Enhanced QA generation with domain tree integration and interactive editing
qa_data = dm.get_pre_label(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model_name="gpt-3.5-turbo",
    custom_domain_tree=your_domain_tree, #user's domain tree for custom initialization
    use_tree_label=True,  # new parameter for domain tree integration
    interactive_tree=True,  # enable interactive tree editing during QA generation
    chunk_size=500,
    chunk_overlap=100,
    question_number=5,
    max_workers=5
)
```

### AI Annotation for multimodal files

```python
import os
import json
from datamax.parser.core import DataMax
from datamax.parser.pdf_parser import PdfParser

def main():
    file_path = "your pdf file_path or directory here"
    parser = DataMax(file_path=file_path, use_mineru=True)
    # result = parser.get_data()
    # print(result)

    # --- 配置API信息 ---
    api_key = os.getenv("DATAMAX_API_KEY", "your_api_key_here")
    base_url = os.getenv("DATAMAX_BASE_URL", "your_base_url_here")
    model_name = "qwen-vl-max-latest"

    if api_key == "your_api_key_here" or base_url == "your_base_url_here":
        print("警告: 请在运行前设置您的API Key和Base URL。")
        return

    qa_list = parser.get_pre_label(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        question_number=5,
        max_workers=5,
        use_mllm=True
    )

    if qa_list:
        print("\n✅ 成功生成多模态问答对:")
        # 使用json.dumps美化输出
        pretty_json = json.dumps(qa_list, indent=2, ensure_ascii=False)
        print(pretty_json)

        # --- 保存结果 ---
        # save_file_name = os.path.join(file_path.replace('pdf',), "qa_pairs")
        parser.save_label_data(qa_list)
        print(f"\n✅ 已将问答对保存至jsonl")


if __name__ == "__main__":
    main()
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
# 1.Install MinerU in virtual environment
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com

# 2.Install the models
python datamax/scripts/download_models.py
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

### Developer Mode

For developers who want to contribute to the project or make modifications, we recommend using developer mode for a better development experience.

#### Setup Developer Mode

```bash
# Clone the repository
git clone https://github.com/Hi-Dolphin/datamax.git
cd datamax

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in developer mode
pip install -e .
```

#### Benefits of Developer Mode

- **Live Updates**: Changes to source code are immediately reflected without reinstallation
- **Easy Testing**: Test your modifications instantly
- **Debugging**: Better debugging experience with direct access to source code
- **Development Workflow**: Seamless integration with your development environment

#### Development Commands

```bash
# Run tests
pytest

# Install development dependencies
pip install -r requirements-dev.txt  # if available

# Check code style
flake8 datamax/
black datamax/

# Build package
python setup.py sdist bdist_wheel
```

#### Making Changes

After installing in developer mode, you can:

1. Edit source code in the `datamax/` directory
2. Changes are automatically available when you import the module
3. Test your changes immediately without reinstalling
4. Submit pull requests with your improvements

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
- 💬 Wechat Group: <br><img src='wechat.png' width=300>
---

⭐ If this project helps you, please give us a star!