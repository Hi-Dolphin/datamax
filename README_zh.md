# Datamax

## 概述
DataMax 是一个面向多格式文件文本解析、数据清洗与数据标注的一体化解决方案。

## 核心功能

### 文件处理能力
当前支持以下格式的读取、转换与内容提取：
- PDF、HTML  
- DOCX/DOC、PPT/PPTX  
- EPUB电子书  
- 图片（.jpg|.png|.jpeg|.webp）  
- XLS/XLSX表格文件  
- 纯文本（TXT|markdown）  

### 数据清洗流程
三级组合式清洗：
1. 异常检测与处理  
2. 隐私信息脱敏  
3. 文本过滤与标准化  

### 智能数据标注
基于LLM+Prompt的分布解耦实现：
- 持续生成预标注数据集  
- 为模型微调提供优化训练数据  


## 安装指南(重点)
依赖包括 libreoffice、datamax、MinerU
## 1. 安装libreoffice依赖
请注意： 如果不安装datamax不支持.doc文件
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

## 2.安装MinerU依赖
如果不安装MinerU 将无法支持PDF更强大的OCR解析
### 创建虚拟环境，安装基础依赖
```bash
conda create -n mineru python=3.10
conda activate mineru
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple
```


### 安装模型权重文件 
https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_zh_cn.md
```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```

### 修改配置文件 magic-pdf.json （用户目录下, 以下为模板预览）
```json
{
    "models-dir": "path\\to\\folder\\PDF-Extract-Kit-1___0\\models",
    "layoutreader-model-dir": "path\\to\\folder\\layoutreader",
    "device-mode": "cpu",
    ...
}
```


##  3.安装datamax基本依赖
1. 克隆仓库到本地：
   ```bash
   git clone 
   ```
2. 安装依赖到conda中：
   ```bash
   cd datamax
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```


## 功能特性
- **多格式支持**: 能够处理PDF、HTML、DOCX和TXT等不同类型的文本文件。
- **内容提取**: 提供强大的内容提取功能，可以准确地从复杂的文档结构中提取所需信息。
- **数据转换**: 支持将处理后的数据转换为markdown格式，方便进一步的数据分析。
- **批量处理**: 可以一次性处理大量文件，提高工作效率。
- **自定义配置**: 用户可以根据需要调整处理参数，满足不同的业务需求。
- **跨平台兼容**: 本SDK可在Windows、MacOS和Linux等多种操作系统上运行。


## 技术栈
- **编程语言**: Python >= 3.10
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
        from datamax import DataMax
        ##  处理单个文件的两种方式
        # 1.长度为1的列表 
        data = DataMax(file_path=[r"docx_files_example/船视宝概述.doc"])
        data = data.get_data()
        # 2.字符串
        data = DataMax(file_path=r"docx_files_example/船视宝概述.doc")
        data = data.get_data()
      
        ## 处理多个文件
        ## 1.长度为n的列表
        data = DataMax(file_path=[r"docx_files_example/船视宝概述1.doc", r"docx_files_example/船视宝概述2.doc"])
        data = data.get_data()
    
        ## 2.传递文件夹字符串
        data = DataMax(file_path=r"docx_files_example/")
        data = data.get_data()
      
        # 文件清洗
        """
        具体清洗规则可以从 datamax/utils/data_cleaner.py 查看 
        abnormal: 异常清洗
        private: 隐私处理
        filter： 文本过滤
        """
        # 直接使用：支持对text参数中的文本内容进行直接清洗,返回字符串
        dm = DataMax()
        data = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
        
        # 过程使用：支持在get_data()后使用,即可返回完整的数据结构
        dm = DataMax(file_path=r"C:\Users\cykro\Desktop\数据库开发手册.pdf", use_ocr=True)
        data2 = dm.get_data()
        cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
      
        # 大模型预标注 支持用OpenAI SDK的方式调用模型
        data = DataMax(
              file_path=r"path\to\xxx.docx"
          )
        parsed_data = data.get_data()
        # 如果不传递自定义messages，则使用sdk中默认的messages
        messages=[
                  {'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': '你是谁？'}
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


## 示例
```python
  ## docx | doc | epub | html | txt | ppt | pptx | xls | xlsx
  from datamax import DataMax
  data = DataMax(file_path=r"docx_files_example/船视宝概述.doc", to_markdown=True)
  """
  参数： 
  file_path: 文件相对路径 / 文件绝对路径
  to_markdown: 是否转换markdown(默认值False，直接返回文本) 该参数只支持word文件 （doc | docx）
  """
  
  
  ## jpg | jpeg | png | ...(图片类型)
  data = DataMax(file_path=r"image.jpg", use_mineru=True)
  """
  参数：
  file_path: 文件相对路径 / 文件绝对路径
  use_mineru: 是否使用MinerU增强
  """
  
  
  ## pdf
  from datamax import DataMax
  data = DataMax(file_path=r"docx_files_example/船视宝概述.pdf", use_mineru=True)
  """
  参数： 
  file_path: 文件相对路径 / 文件绝对路径
  use_mineru: 是否使用MinerU增强
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
├── examples        # 示例代码
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


    

    