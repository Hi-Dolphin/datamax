# DataMax

## Overview
DataMax is designed as a comprehensive solution for processing diverse file formats, performing data cleaning, and facilitating data annotation.

## Key Features

### File Processing Capabilities
Currently supports reading, conversion, and extraction from:
- PDF, HTML  
- DOCX/DOC, PPT/PPTX  
- EPUB  
- Images
- XLS/XLSX spreadsheets  
- Plain text (TXT)  

### Data Cleaning Pipeline
Three-tiered cleaning process:
1. Anomaly detection and handling  
2. Privacy protection processing  
3. Text filtering and normalization  

### AI-Powered Data Annotation
Implements an LLM+Prompt to:
- Continuously generate pre-labeled datasets  
- Provide optimized training data for model fine-tuning  


## Installation Guide (Key Dependencies)
Dependencies include libreoffice, datamax, and MinerU.

### 1. Installing libreoffice Dependency
**Note:** Without datamax, .doc files will not be supported.

#### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install libreoffice
```
### Windows
```text
Install LibreOffice from: [Download LibreOffice](https://www.libreoffice.org/download/download-libreoffice/?spm=5176.28103460.0.0.5b295d275bpHzh)  
Add to environment variable: `$env:PATH += ";C:\Program Files\LibreOffice\program"`
```
### Checking LibreOffice Installation
```bash
soffice --version
```

## 2. Installing MinerU Dependency
Note: Without MinerU, advanced OCR parsing for PDFs will not be supported.
### Create a Virtual Environment and Install Basic Dependencies
```bash
conda create -n mineru python=3.10
conda activate mineru
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple
```
### Installing Model Weight Files
https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_zh_cn.md
```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```

### Modify the Configuration File magic-pdf.json (Located in the User Directory, Template Preview Below)
```json
{
    "models-dir": "path\\to\\folder\\PDF-Extract-Kit-1___0\\models",
    "layoutreader-model-dir": "path\\to\\folder\\layoutreader",
    "device-mode": "cpu",
    ...
}
```

##  3. Installing Basic Dependencies for datamax
1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies into conda:
   ```bash
   cd datamax
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```


## Features
- **Multi-format Support**: Capable of handling various text file types such as PDF, HTML, DOCX, and TXT.
- **Content Extraction**: Provides powerful content extraction capabilities to accurately retrieve information from complex document structures.
- **Data Conversion**: Supports converting processed data into markdown format for further analysis.
- **Batch Processing**: Can handle multiple files at once, improving work efficiency.
- **Customizable Configuration**: Users can adjust processing parameters according to their needs to meet different business requirements.
- **Cross-platform Compatibility**: This SDK can run on multiple operating systems, including Windows, MacOS, and Linux.


## Technology Stack

- **Programming Language**: Python >= 3.10  
- **Dependency Libraries**:  
  - PyMuPDF: For PDF file parsing.  
  - BeautifulSoup: For HTML file parsing.  
  - python-docx: For DOCX file parsing.  
  - pandas: For data processing and conversion.  
  - paddleocr: For parsing scanned PDFs, tables, and images.  
- **Development Environment**: Visual Studio Code or PyCharm  
- **Version Control**: Git  

## Usage Instructions
### Installing the SDK
- **Installation Commands**:
  ```bash
  ## Local Installation
  python setup.py sdist bdist_wheel
  pip install dist/datamax-0.1.3-py3-none-any.whl
  
  ## Pip Installation
  pip install pydatamax
  ```
  

- **Importing the Code**:
    ```python
    # File Parsing
    from datamax import DataMax
    
    ## Handling a Single File in Two Ways
    # 1. Using a List of Length 1
    data = DataMax(file_path=[r"docx_files_example/船视宝概述.doc"])
    data = data.get_data()
    
    # 2. Using a String
    data = DataMax(file_path=r"docx_files_example/船视宝概述.doc")
    data = data.get_data()
    
    ## Handling Multiple Files
    # 1. Using a List of Length n
    data = DataMax(file_path=[r"docx_files_example/船视宝概述1.doc", r"docx_files_example/船视宝概述2.doc"])
    data = data.get_data()
    
    # 2. Passing a Folder Path as a String
    data = DataMax(file_path=r"docx_files_example/")
    data = data.get_data()
    
    # Data Cleaning
    """
    Cleaning rules can be found in datamax/utils/data_cleaner.py
    abnormal: Abnormal cleaning
    private: Privacy processing
    filter: Text filtering
    """
    # Direct Use: Clean the text parameter directly and return a string
    dm = DataMax()
    data = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
    
    # Process Use: Use after get_data() to return the complete data structure
    dm = DataMax(file_path=r"C:\Users\cykro\Desktop\数据库开发手册.pdf", use_ocr=True)
    data2 = dm.get_data()
    cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
    
    # Large Model Pre-annotation Supporting any model that can be called via OpenAI SDK
    data = DataMax(file_path=r"path\to\xxx.docx")
    parsed_data = data.get_data()
    # If no custom messages are passed, the default messages in the SDK will be used
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}
    ]
    qa_datas = data.get_pre_label(
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_name="qwen-max",
        chunk_size=500,
        chunk_overlap=100,
        question_number=5,
        max_workers=5,
        # message=[]
    )
    print(f'Annotated result:{qa_datas}')
    ```


## Examples
    ```python
    ## docx | doc | epub | html | txt | ppt | pptx | xls | xlsx
    from datamax import DataMax
    data = DataMax(file_path=r"docx_files_example/船视宝概述.doc", to_markdown=True)
    """
    Parameters: 
    file_path: Relative file path / Absolute file path
    to_markdown: Whether to convert to markdown (default value False, directly returns text) This parameter only supports word files (doc | docx)
    """
    
    ## jpg | jpeg | png | ...(image types)
    data = DataMax(file_path=r"image.jpg", use_mineru=True)
    """
    Parameters:
    file_path: Relative file path / Absolute file path
    use_mineru: Whether to use MinerU enhancement
    """
    
    ## pdf
    from datamax import DataMax
    data = DataMax(file_path=r"docx_files_example/船视宝概述.pdf", use_mineru=True)
    """
    Parameters: 
    file_path: Relative file path / Absolute file path
    use_mineru: Whether to use MinerU enhancement
    """
    ```

## Contribution Guide
We welcome any form of contribution, whether it is reporting bugs, suggesting new features, or submitting code improvements. Please read our Contributor's Guide to learn how to get started.
## License
This project is licensed under the MIT License. For more details, see the LICENSE file.

## Contact Information
If you encounter any issues during use, or have any suggestions or feedback, please contact us through the following means:
- Email: cy.kron@foxmail.com | zhibaohe@hotmail.com
- Project Homepage: GitHub Project Link

