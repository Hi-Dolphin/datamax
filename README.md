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

# AI annotation default with tree
qa_data = dm.generate_qa_with_tree(
    api_key="your-api-key"
    base_url="https://api.openai.com/v1",
    model_name="gpt-3.5-turbo",
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

### AI Annotation with optional Domain Tree

```python

#Custom annotation tasks with domain tree
dm = DataMax(file_path="your_file.md")
qa_data = dm.generate_qa_with_tree(
    api_key="sk-xxx",          
    base_url="https://api.provider.com/v1",         
    model_name="model-name",          
    chunk_size=500,                   # Text chunk size
    chunk_overlap=100,                # Overlap between chunks
    question_number=5,                # Number of questions per chunk
    max_workers=5                     # Number of threads for parallel processing
    use_tree_label=True               # Whether use tree label or not
)

#After generating the domain tree, user is allowed to customize their tree

"""
Supported Commands:
1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）
2. 增加节点：xxx；父节点：xxx；子节点：xxx
3. 删除节点：xxx
4. 更新节点：新名称；原先节点：旧名称
5. 结束树操作
注意，节点的格式通常为：x.xx xxxx,如：‘1.1 货物运输组织与路径规划’或‘1 运输系统组织’
"""

# Save the QA label data to file
dm.save_label_data(qa_data, save_file_name="qa_label_data")
```

## 🌳 Tree Editing Workflow (AI Annotation)

After using the `generate_qa_with_tree` method to generate a domain tree for AI annotation, the terminal will display the current tree structure and enter the tree editing mode by default. Users can follow the terminal prompts and refer to the command formats below to flexibly adjust the tree structure directly in the terminal.

### Workflow

1. **After the tree is generated, the terminal will display the current tree structure and enter the editing mode.**

2. **In editing mode, users can input the following commands as prompted to operate on the tree structure:**

   - Add node: `Add node: <Node Name>; Parent node: <Parent Name>` (Parent node can be left blank to add as a root node)
   - Add node: `Add node: <Node Name>; Parent node: <Parent Name>; Child node: <Child Name>`
   - Delete node: `Delete node: <Node Name>`
   - Update node: `Update node: <New Name>; Original node: <Old Name>`
   - Finish editing: `Finish editing`

   > Node names are usually in the format: `1.1 Organization and Route Planning of Cargo Transportation` or `1 Transportation System Organization`

3. **After each operation, the terminal will immediately display the updated tree structure and continue to wait for user input, until the user enters `Finish editing` to end the tree editing process.**

4. **The final tree structure will be used for subsequent AI annotation tasks.**


---

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
