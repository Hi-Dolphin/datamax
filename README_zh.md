# 数据工厂 0.1.6


## 项目简介
文本处理工具是一个多功能的数据处理项目，旨在提供一种高效且易于使用的方式来处理各种格式的文本文件。
该项目支持PDF、HTML、DOCX\DOC、PPT\PPTX、EPUB、图片（.jpg | .png）以及TXT等常见文本格式的读取、转换与提取。


## 安装指南

## Install

### Linux（Debian/Ubuntu）
```
sudo apt-get update
sudo apt-get install libreoffice
```
### Windows
```
在Windows上安装LibreOffice：https://www.libreoffice.org/download/download-libreoffice/?spm=5176.28103460.0.0.5b295d275bpHzh
需加入到环境变量：$env:PATH += ";C:\Program Files\LibreOffice\program"
```
### 检查LibreOffice是否安装
```
soffice --version
```
### 安装依赖
```
pip install --no-cache-dir -r requirements.txt
```

1. 克隆仓库到本地：
   ```bash
   git clone 
   ```
2. 进入项目目录并创建虚拟环境（举例uv）：
   ```bash
   cd DataMax
   uv init
   uv add requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   uv sync
   . .venv/bin/activate
   ```


## 功能特性
- **多格式支持**: 能够处理PDF、HTML、DOCX和TXT等不同类型的文本文件。
- **内容提取**: 提供强大的内容提取功能，可以准确地从复杂的文档结构中提取所需信息。
- **数据转换**: 支持将处理后的数据转换为markdown格式，方便进一步的数据分析。
- **批量处理**: 可以一次性处理大量文件，提高工作效率。
- **自定义配置**: 用户可以根据需要调整处理参数，满足不同的业务需求。
- **跨平台兼容**: 本SDK可在Windows、MacOS和Linux等多种操作系统上运行。


## 技术栈
- **编程语言**: Python >= 3.9
- **依赖库**:
  - PyMuPDF: 用于PDF文件的解析。
  - BeautifulSoup: 用于HTML文件的解析。
  - python-docx: 用于DOCX文件的解析。
  - pandas: 用于数据处理和转换。
  - paddleocr: 用于PDF扫描件、表格、图片的解析。
- **开发环境**: Visual Studio Code 或 PyCharm
- **版本控制**: Git

## 使用说明
### 安装SDK
- 安装命令：
  ```bash
  ## 本地安装
  python setup.py sdist bdist_wheel
  pip install dist/datamax-0.1.3-py3-none-any.whl
  
  ## pip安装
  pip install pydatamax
  ```
  
- 引入代码
    ```python
  # 文件解析
  from datamax import DataMaxParser
  ##  处理单个文件的两种方式
  # 1.长度为1的列表 
  data = DataMaxParser(file_path=[r"docx_files_example/船视宝概述.doc"])
  data = data.get_data()
  # 2.字符串
  data = DataMaxParser(file_path=r"docx_files_example/船视宝概述.doc")
  data = data.get_data()
  
  ## 处理多个文件
  ## 1.长度为n的列表
  data = DataMaxParser(file_path=[r"docx_files_example/船视宝概述1.doc", r"docx_files_example/船视宝概述2.doc"])
  data = data.get_data()

  ## 2.传递文件夹字符串
  data = DataMaxParser(file_path=r"docx_files_example/")
  data = data.get_data()
  
  # 文件清洗
  """
  具体清洗规则可以从 datamax/utils/data_cleaner.py 查看 
  abnormal: 异常清洗
  private: 隐私处理
  filter： 文本过滤
  """
  # 直接使用：支持对text参数中的文本内容进行直接清洗,返回字符串
  dm = DataMaxParser()
  data = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
    
  # 过程使用：支持在get_data()后使用,即可返回完整的数据结构
  dm = DataMaxParser(file_path=r"C:\Users\cykro\Desktop\数据库开发手册.pdf", use_ocr=True)
  data2 = dm.get_data()
  cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
  ```


## 示例
```python
  ## docx | doc | epub | html | txt | jpg | png | ppt | pptx
  from datamax import DataMaxParser
  data = DataMaxParser(file_path=r"docx_files_example/船视宝概述.doc", to_markdown=True)
  """
  参数： 
  file_path: 文件相对路径 / 文件绝对路径
  to_markdown: 是否转换markdown(默认值False，直接返回文本) 该参数只支持word文件 （doc | docx）
  """
  
  
  ## pdf
  from datamax import DataMaxParser
  data = DataMaxParser(file_path=r"docx_files_example/船视宝概述.pdf", use_ocr=True, use_gpu=True, gpu_id=0)
  """
  参数： 
  file_path: 文件相对路径 / 文件绝对路径
  use_ocr: 是否使用ocr (True返回markdown, False返回纯文本)
  use_gpu: 是否使用gpu 
    use_gpu=True: 
        if 当有cuda环境 => 执行 nvcc -V。 会自动安装 paddlepaddle-gpu环境
        else： 自动安装基础的 paddlepaddle环境
    use_gpu=False：
        自动安装基础的 paddlepaddle环境
  gpu_id: gpu卡 （默认6卡）
  """
```

## 贡献指南
我们欢迎任何形式的贡献，无论是报告bug、提出新功能建议还是直接提交代码改进。请先阅读我们的[贡献者指南](CONTRIBUTING.md)了解如何开始。

## 许可证
本项目采用MIT许可证，详情参见[LICENSE](LICENSE)文件。

## 联系方式
如果在使用过程中遇到任何问题，或者有任何建议和反馈，请通过以下方式联系我们：
- 电子邮件: cy.kron@foxmail.com | zhibaohe@hotmail.com
- 项目主页: [GitHub项目链接](xxxx)


## RoadMap
- [x] 实现OSS数据类
- [x] 实现OBS数据类
- [x] 实现PGSQL数据类
- [x] 实现本地数据类
- [x] OSS数据类能获得OSS数据源内的桶信息、元数据、文件列表
- [x] OBS数据类能获得OBS数据源内的桶信息、元数据、文件列表
- [ ] PGSQL数据类能够获取PGSQL数据源内的Schema信息、元数据、表结构


- (Optional) 数据类（接入数据源）  /  本地直接读取 DataSourceClass
    - MinIO
    - OSS
    - OBS
    - PostgreSQL

DataLoader.load(fp: file_path, ) -> 本地直接读取

DataLoader.load(source: DataSourceClass, ) -> 接入数据源，返回数据源的元数据: 能知道文件大小 & 文件路径 & 文件下载地址 & 数据源的存储占用空间大小 -> 

- 数据加载类（加载读取PDF、Word、Excel。能够按需加载数据源中的以下类型的数据）
    - .pdf
    - .pdf（图片型 / 扫描件）
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
    - 多模态
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


- 数据解析类（解析算法、解析逻辑 & 输出字符串对象）
    - .pdf
    - .pdf（图片型 / 扫描件） PaddleOCR / AI LAB OCR /  xx OCR
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
    - 多模态
        - .png
        - .jpg
        - .jpeg
        - .bmp
        - .gif


- 数据清洗类（输入：字符串对象，根据清洗规则、得到清洗后的markdown格式文本。输出：markdown）
    - 


- 大模型类（预标）

- 数据输出类（格式统一）


## 结构

├── api            # 项目中开放出来的各种接口
├── datamax        # 各种 SDK 的核心函数与类
├── dockerfiles    # Docker 配置文件
├── docs           # 项目文档
├── example        # 示例代码
├── README.md      # 项目说明文件
├── scripts        # 各种脚本
└── test           # 测试代码


## 规范

1. 类名的声明用大驼峰
2. 函数名的声明用小写+下划线
3. 函数的输入输出需要用冒号声明数据类型
4. print logger 在推送代码前删除掉
5. 变基:
    1. git add .
    2. git commit -m ":boom: new feature"
    3. git pull --rebase
    4. git push


    

    