# UNO API 高并发文档转换使用指南

## 概述

本项目已支持使用 LibreOffice UNO (Universal Network Objects) API 进行高并发文档转换，相比传统的命令行调用方式，性能提升显著。

## 主要优势

1. **真并行处理**：支持多个文档同时转换，而不是串行处理
2. **性能提升**：避免重复启动 LibreOffice 进程的开销
3. **资源优化**：共享单个 LibreOffice 服务实例
4. **稳定性提升**：更好的错误处理和服务管理

## 安装要求

### 1. 安装 LibreOffice

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libreoffice libreoffice-script-provider-python python3-uno

# CentOS/RHEL
sudo yum install libreoffice libreoffice-pyuno

# macOS
brew install libreoffice

# Windows
# 下载并安装 LibreOffice: https://www.libreoffice.org/download/
```

### 2. 配置 Python UNO

确保 Python 能够导入 UNO 模块：

```python
# 测试 UNO 是否可用
import uno
print("UNO 已安装并可用")
```

如果导入失败，可能需要：

```bash
# 设置 Python 路径（Linux）
export PYTHONPATH=/usr/lib/libreoffice/program:$PYTHONPATH

# 或者使用 LibreOffice 自带的 Python
/usr/lib/libreoffice/program/python
```

## 使用方法

### 1. 基本使用

```python
from datamax.parser.doc_parser import DocParser
from datamax.parser.docx_parser import DocxParser
from datamax.parser.ppt_parser import PPtParser

# 创建解析器（自动检测并使用 UNO）
doc_parser = DocParser("document.doc")
result = doc_parser.parse("document.doc")

# 明确指定使用 UNO
docx_parser = DocxParser("document.docx", use_uno=True)
result = docx_parser.parse("document.docx")

# 禁用 UNO，使用传统方式
ppt_parser = PPtParser("presentation.ppt", use_uno=False)
result = ppt_parser.parse("presentation.ppt")
```

### 2. 并行处理多个文档

```python
import concurrent.futures
from datamax.utils import get_uno_manager

# 预先连接 UNO 服务
manager = get_uno_manager()

def convert_document(file_path):
    """转换单个文档"""
    if file_path.endswith('.doc'):
        parser = DocParser(file_path, use_uno=True)
    elif file_path.endswith('.docx'):
        parser = DocxParser(file_path, use_uno=True)
    elif file_path.endswith('.ppt'):
        parser = PPtParser(file_path, use_uno=True)
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")

    return parser.parse(file_path)

# 并行转换多个文档
file_paths = ["doc1.doc", "doc2.docx", "ppt1.ppt", "doc3.doc"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(convert_document, fp) for fp in file_paths]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

### 3. 异步处理

```python
import asyncio
from datamax.utils import get_uno_manager

async def async_convert(file_path):
    """异步转换文档"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, convert_document, file_path)

async def batch_convert_async(file_paths):
    """批量异步转换"""
    # 预先连接服务
    manager = get_uno_manager()

    # 创建异步任务
    tasks = [async_convert(fp) for fp in file_paths]

    # 并发执行
    results = await asyncio.gather(*tasks)
    return results

# 运行异步转换
file_paths = ["doc1.doc", "doc2.docx", "ppt1.ppt"]
results = asyncio.run(batch_convert_async(file_paths))
```

## 配置选项

### UnoManager 配置

```python
from datamax.utils.uno_handler import UnoManager

# 自定义配置
manager = UnoManager(
    host="localhost",  # LibreOffice 服务主机
    port=2002,        # 服务端口（默认 2002）
    timeout=30        # 连接超时时间（秒）
)

# 手动管理服务生命周期
manager.connect()  # 连接服务
# ... 执行转换操作 ...
manager.disconnect()  # 断开连接
manager.stop_service()  # 停止服务
```

### 性能调优

1. **工作线程数**：根据 CPU 核心数和文档大小调整
   ```python
   # CPU 密集型任务
   max_workers = os.cpu_count()

   # I/O 密集型任务
   max_workers = os.cpu_count() * 2
   ```

2. **服务端口**：如果默认端口被占用，可以使用其他端口
   ```python
   manager = UnoManager(port=2003)
   ```

3. **超时设置**：处理大文档时可能需要增加超时时间
   ```python
   manager = UnoManager(timeout=60)  # 60 秒超时
   ```

## 故障排除

### 1. UNO 模块导入失败

```bash
# 检查 LibreOffice 安装
which soffice

# 查找 uno.py 位置
find /usr -name "uno.py" 2>/dev/null

# 添加到 Python 路径
export PYTHONPATH=/usr/lib/libreoffice/program:$PYTHONPATH
```

### 2. 服务连接失败

```bash
# 检查端口是否被占用
netstat -tuln | grep 2002

# 手动启动 LibreOffice 服务
soffice --headless --invisible --nocrashreport --nodefault \
        --nofirststartwizard --nologo --norestore \
        "--accept=socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
```

### 3. 转换失败

- 确保文档格式正确且未损坏
- 检查文件权限
- 查看日志获取详细错误信息

## 性能对比

使用提供的示例脚本进行性能测试：

```bash
python examples/uno_conversion_example.py
```

典型性能提升：
- **串行处理**：10 个文档需要 50 秒
- **并行处理（4线程）**：10 个文档需要 15 秒
- **并行处理（8线程）**：10 个文档需要 10 秒

## 注意事项

1. **内存使用**：LibreOffice 服务会占用一定内存，建议监控内存使用情况
2. **服务稳定性**：长时间运行可能需要定期重启服务
3. **文档兼容性**：某些特殊格式的文档可能需要特殊处理
4. **并发限制**：虽然支持并发，但过高的并发数可能导致性能下降

## 示例项目

参见 `examples/uno_conversion_example.py` 获取完整的使用示例。
