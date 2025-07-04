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

# 带领域树的AI标注
qa_data = dm.generate_qa_with_tree(
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
dm = DataMax(file_path="image.jpg", use_mineru=True)
```

### 批处理解析
```python
# 批量解析多个文件
dm = DataMax(
    file_path=["file1.pdf", "file2.docx"],
    use_mineru=True
)
data = dm.get_data()
```

### 文件缓存
```python
# 缓存解析结果，避免重复解析
dm = DataMax(
    file_path=["file1.pdf", "file2.docx"],
    ttl=3600  # 缓存时间，单位秒, 默认3600秒, 如果为0则不缓存
)
data = dm.get_data()
```

### 数据清洗
## 异常处理

- remove_abnormal_chars 从文本中移除异常字符
- remove_html_tags 移除HTML标签
- convert_newlines 将\r转换为\n并将多个\n合并为单个\n
- single_space 将多个空格(2个以上)转换为单个空格
- tabs_to_spaces 将制表符转换为4个空格
- remove_invisible_chars 移除不可见ASCII字符
- simplify_chinese 将繁体中文转换为简体中文

## 文本过滤

- filter_by_word_repetition 词重复率过滤
- filter_by_char_count 按字符数量过滤
- filter_by_numeric_content 按数字占比过滤

## 隐私脱敏

- replace_ip
- replace_email
- replace_customer_number   4008-123-123 清洗热线电话
- replace_bank_id
- replace_phone_number
- replace_qq
- replace_id_card


```python
# 三种清洗模式(快速使用不支持自定义)
dm.clean_data(method_list=[
    "abnormal",  # 异常数据处理
    "private",   # 隐私信息脱敏
    "filter"     # 文本过滤规范化
])

# 自定义清洗流程(支持自定义)
from datamax.utils.data_cleaner import TextFilter, PrivacyDesensitization, AbnormalCleaner
dm = DataMax(
    file_path=r"C:\Users\cykro\Desktop\香港开发机.txt"
)
parsed_data = dm.get_data().get('content')
# 1. 文本过滤
tf = TextFilter(parsed_data=parsed_data)
    # 词重复率过滤 参数 threshold 默认为 0.6，即文本中最多允许 60% 的字符是重复的
tf_bool = tf.filter_by_word_repetition(threshold=0.6)
if tf_bool:
    print("文本通过词重复率过滤")
else:
    print("文本未通过词重复率过滤")
    
# 按字符数量过滤 参数 min_chars 默认为 30，即文本中最少允许 30 个字符, max_chars 默认为 500000，即文本中最多允许 500000 个字符
tf_bool = tf.filter_by_char_count(min_chars=30, max_chars=500000)
if tf_bool:
    print("文本通过字符数量过滤")
else:
    print("文本未通过字符数量过滤")

# 按数字占比过滤 参数 threshold 默认为 0.6，即文本中最多允许 60% 的字符是数字
tf_bool = tf.filter_by_numeric_content(threshold=0.6)
if tf_bool:
    print("文本通过数字比例过滤")
else:
print("文本未通过数字比例过滤")

# 2. 隐私脱敏
pd = PrivacyDesensitization(parsed_data=parsed_data)
res = pd.replace_ip(
    token="MyIP"
)
print(res)

# 3. 异常字符清洗
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

### 文本切分

```python
dm.split_data(
    chunk_size=500,      # 文本块大小
    chunk_overlap=100,    # 重叠长度
    use_langchain=True  # 使用LangChain进行文本切分
)

# 当use_langchain为False时，使用自定义切分方法
# 。！？作为分隔符，连续的分隔符会被合并 chunk_size是严格的字符串长度不会超过
for chunk in parser.split_data(chunk_size=500, chunk_overlap=100, use_langchain=False).get("content"):
    print(chunk)
```

### AI标注（带领域树标签）

```python
#带领域树的自定义生成QA
dm = DataMax(file_path="your_file.md")
qa_data = dm.generate_qa_with_tree(
    api_key="sk-xxx",          
    base_url="https://api.provider.com/v1",         
    model_name="model-name",          
    chunk_size=500,                   # Text chunk size
    chunk_overlap=100,                # Overlap between chunks
    question_number=5,                # Number of questions per chunk
    max_workers=5                     # Number of threads for parallel processing
    use_tree_label=True               # Whether use tree label or not(choose yes here)
)

#完成领域树生成后，用户能够在终端交互地修改领域树

# 保存QA对结果
dm.save_label_data(qa_data, save_file_name="qa_label_data")

#或者，你也可以不使用领域树直接快速生成QA对
dm = DataMax(file_path="your_file.md")
qa_data = dm.generate_qa_with_tree(
    api_key="sk-xxx",          
    base_url="https://api.provider.com/v1",         
    model_name="model-name",          
    chunk_size=500,                   # Text chunk size
    chunk_overlap=100,                # Overlap between chunks
    question_number=5,                # Number of questions per chunk
    max_workers=5                     # Number of threads for parallel processing
    use_tree_label=False               # Whether use tree label or not(choose not here)
)
```
```

#### 领域树交互操作说明

生成树后，终端会输出当前树结构，并自动进入编辑模式。用户可根据提示输入以下命令，对树结构进行灵活调整：

支持的命令：
 1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）
 2. 增加节点：xxx；父节点：xxx；子节点：xxx
 3. 删除节点：xxx
 4. 更新节点：新名称；原先节点：旧名称
 5. 结束树操作

> 节点名称格式通常为：x.xx xxxx，例如：'1.1 货物运输组织与路径规划' 或 '1 运输系统组织';而键入的指令必须严格参上（语言，格式等）

每次操作后，终端会实时反馈树结构的最新状态，直到用户输入“结束树操作”结束编辑，最终树结构将用于后续AI标注任务。

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

