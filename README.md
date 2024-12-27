# DataMax 0.1.6


## Project Introduction
The Text Processing Tool is a multifunctional data processing project designed to provide an efficient and easy-to-use way to handle various formats of text files. The project supports reading, converting, and extracting from common text formats such as PDF, HTML, DOCX/DOC, PPT/PPTX, EPUB, images (.jpg | .png), and TXT.


## Installation Guide

## Install

### Linux（Debian/Ubuntu）
```
sudo apt-get update
sudo apt-get install libreoffice
```
### Windows
```
Windows install LibreOffice：https://www.libreoffice.org/download/download-libreoffice/?spm=5176.28103460.0.0.5b295d275bpHzh
$env:PATH += ";C:\Program Files\LibreOffice\program"
```
### Check LibreOffice Installation
```
soffice --version
```
### Install Dependencies
```
pip install --no-cache-dir -r requirements.txt
```

1. Clone the repository to local:
   ```bash
   git clone 
   ```
2. Enter the project directory and create a virtual environment (e.g., uv):
   ```bash
   cd DataMax
   uv init
   uv add requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   uv sync
   . .venv/bin/activate
   ```


## Features
- **Multi-format Support**: Capable of handling PDF, HTML, DOCX, and TXT, among other text file types.
- **Content Extraction**: Provides powerful content extraction capabilities to accurately extract needed information from complex document structures.
- **Data Conversion**: Supports conversion of processed data to Markdown format for further data analysis.
- **Batch Processing**: Capable of processing a large number of files at once, improving work efficiency.
- **Custom Configuration**: Users can adjust processing parameters according to their needs to meet various business requirements.
- **Cross-platform Compatibility**: This SDK can run on multiple operating systems including Windows, MacOS, and Linux.


## Technology Stack
- **Programming Language**: Python >= 3.9
- **Dependency Libraries**:
  - PyMuPDF: For parsing PDF files.
  - BeautifulSoup: For parsing HTML files.
  - python-docx: For parsing DOCX files.
  - pandas: For data processing and conversion.
  - paddleocr: For parsing PDF scans, forms, and images.
- **Development Environment**: Visual Studio Code or PyCharm
- **Version Control**: Git


## Usage Instructions
### Install SDK
- Installation Commands:
  ```bash
  ## Local Installation
  python setup.py sdist bdist_wheel
  pip install dist/datamax-0.1.3-py3-none-any.whl
  
  ## pip Installation
  pip install pydatamax
  ```
- Import Code
    ```python
    from datamax import DataMaxParser
    ## Handling a single file in two ways
    # 1. List of length 1
    data = DataMaxParser(file_path=[r"docx_files_example/ShipViewOverview.doc"])
    data = data.get_data()
    # 2. String
    data = DataMaxParser(file_path=r"docx_files_example/ShipViewOverview.doc")
    data = data.get_data()
    
    ## Handling multiple files
    ## 1. List of length n
    data = DataMaxParser(file_path=[r"docx_files_example/ShipViewOverview1.doc", r"docx_files_example/ShipViewOverview2.doc"])
    data = data.get_data()
    
    ## 2. Passing a folder string
    data = DataMaxParser(file_path=r"docx_files_example/")
    data = data.get_data()
  
  # Data Cleaning
    """
    For specific cleaning rules, please refer to datamax/utils/data_cleaner.py
    abnormal: abnormal data cleaning
    private: privacy processing
    filter: text filtering
    """
    # Direct use: Supports direct cleaning of text content in the text parameter and returns a string
    dm = DataMaxParser()
    data = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
        
    # Process use: Supports using after get_data(), which returns the complete data structure
    dm = DataMaxParser(file_path=r"C:\Users\cykro\Desktop\Database Development Manual.pdf", use_ocr=True)
    data2 = dm.get_data()
    cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
    ```

## Examples
```python
  ## docx | doc | epub | html | txt | jpg | png | ppt | pptx
  from datamax import DataMaxParser
  data = DataMaxParser(file_path=r"docx_files_example/ShipViewOverview.doc", to_markdown=True)
  """
  Parameters: 
  file_path: File relative path / File absolute path
  to_markdown: Whether to convert to markdown (default value False, returns text directly) This parameter only supports Word files (doc | docx)
  """
  

  ## pdf
  from datamax import DataMaxParser
  data = DataMaxParser(file_path=r"docx_files_example/ShipViewOverview.pdf", use_ocr=True, use_gpu=True, gpu_id=0)
"""
  Parameters: 
  file_path: file relative path / file absolute path
  use_ocr: whether to use ocr (True returns markdown, False returns plain text)
  use_gpu: whether to use gpu 
    use_gpu=True: 
        if there is a cuda environment => executes nvcc -V. It will automatically install the paddlepaddle-gpu environment
        else: automatically installs the basic paddlepaddle environment
    use_gpu=False: 
        automatically installs the basic paddlepaddle environment
  gpu_id: gpu card (default is 6)
"""
```

## Contribution Guidelines
We welcome any form of contribution, whether it's reporting bugs, suggesting new features, or directly submitting code improvements. Please read our [Contributor Guidelines](CONTRIBUTING.md) to get started.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
If you encounter any issues during use, or have any suggestions and feedback, please contact us through the following methods:
- Email: cy.kron@foxmail.com | zhibaohe@hotmail.com
- Project Homepage: [GitHub Project Link](xxxx)

## RoadMap
- [x] Implement OSS Data Class
- [x] Implement OBS Data Class
- [x] Implement PGSQL Data Class
- [x] Implement Local Data Class
- [x] OSS Data Class should be able to obtain bucket information, metadata, and file lists from the OSS data source
- [x] OBS Data Class should be able to obtain bucket information, metadata, and file lists from the OBS data source
- [ ] PGSQL Data Class should be able to obtain schema information, metadata, and table structures from the PGSQL data source


- (Optional) Data Classes (Access Data Sources) / Direct Local Reading DataSourceClass
    - MinIO
    - OSS
    - OBS
    - PostgreSQL

DataLoader.load(fp: file_path, ) -> local path 

DataLoader.load(source: DataSourceClass, ) -> Access data source, return metadata of the data source: able to know file size & file path & file download address & storage space occupied by the data source ->

- Data Loading Class (Load and read PDF, Word, Excel. Able to load the following types of data from the data source on demand)
    - .pdf
    - .pdf（Image type / Scanned copy）
    - .docx
    - .html
    - .pptx
    - .epub
    - .txt
    - .md
    ---
    - .csv
    - .json
    - .xlsx
    ---
    - http / https
    --- 
    - Multimodal
        - .png
        - .jpg
        - .jpeg
        - .bmp
        - .gif



class DataLoaderClass:
    def read_docx(self, DocxInputVo):  # TODO: ccy
        ...
        
    def read_pdf():
        ...


- Data Parsing Class (Parsing algorithms, parsing logic & output string objects)
    - .pdf
    - .pdf (Image type / Scanned copy) PaddleOCR / AI LAB OCR /  xx OCR
    - .doc
    - .docx
    - .html
    - .ppt
    - .pptx
    - .epub
    - .txt
    - .md
    ---
    - .csv
    - .json
    - .xlsx
    - .xls
    ---
    - http / https
    --- 
    - Multimodal
        - .png
        - .jpg
        - .jpeg
        - .bmp
        - .gif


- Data Cleaning Class (Input: String object, according to cleaning rules, get cleaned markdown format text. Output: markdown)

- Large Model Class (Pre-tagging)

- Data Output Class (Unified format)


## Structure
```bash
├── api            # Various interfaces exposed by the project
├── datamax        # Core functions and classes of various SDKs
├── dockerfiles    # Docker configuration files
├── docs           # Project documentation
├── example        # Example code
├── README.md      # Project description file
├── scripts        # Various scripts
└── test           # Test code
```



## Standards

1. Class names are declared using PascalCase
2. Function names are declared using lowercase and underscores
3. Function input and output need to be declared with data types using colons
4. Remove print logger before pushing code
5. Rebase:
    1. git add .
    2. git commit -m ":boom: new feature"
    3. git pull --rebase
    4. git push