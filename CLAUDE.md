
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataMax 是一个强大的多格式文件解析、数据清洗和AI标注工具包。该项目是一个Python库，支持PDF、DOCX/DOC、PPT/PPTX、XLS/XLSX、HTML、EPUB、TXT、图片等多种格式的文件解析，并提供智能清洗和AI自动标注功能。

## 核心架构

### 主要模块结构
- `datamax/parser/` - 文件解析器模块，包含各种格式的解析器
  - `core.py` - 核心解析引擎和DataMax主类
  - `base.py` - 解析器基类
  - `*_parser.py` - 各种格式的具体解析器实现
- `datamax/loader/` - 数据加载器模块，支持本地文件、OSS、MinIO等数据源
- `datamax/utils/` - 工具模块
  - `data_cleaner.py` - 数据清洗工具
  - `qa_generator.py` - QA生成器
  - `domain_tree.py` - 领域树处理
  - `tokenizer.py` - 分词器

### 核心设计模式
- 工厂模式：`ParserFactory`根据文件扩展名创建对应的解析器
- 策略模式：不同的数据清洗策略（异常清洗、隐私脱敏、文本过滤）
- 缓存机制：支持TTL缓存避免重复解析

## 常用开发命令

### 安装和环境设置
```bash
# 本地开发安装
pip install -r requirements.txt
python setup.py install

# 开发者模式安装（推荐）
pip install -e .
```

### 测试命令
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_core.py

# 运行测试并生成覆盖率报告
pytest --cov=datamax --cov-report=html:htmlcov --cov-fail-under=80
```

### 代码质量检查
```bash
# 使用内置的代码格式化脚本
python scripts/format_code.py

# 仅检查不修复
python scripts/format_code.py --check-only

# 手动运行各个工具
black datamax/                           # 代码格式化
isort datamax/ --profile black           # 导入排序
flake8 datamax/ --max-line-length=88 --extend-ignore=E203,W503  # 代码检查
```

### 模型下载（可选）
```bash
# 下载MinerU所需模型
python scripts/download_models.py
```

## 开发规范

### 代码风格
- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序（配置为与Black兼容）
- 遵循 flake8 规范，最大行长度88字符
- 使用 pytest 进行测试，要求覆盖率不低于80%

### 文件命名约定
- 解析器文件：`{format}_parser.py`（如 `pdf_parser.py`）
- 测试文件：`test_{module}.py`
- 工具文件：使用描述性命名

### 重要配置文件
- `setup.py` - 包配置和依赖管理
- `requirements.txt` - 依赖列表
- `pytest.ini` - 测试配置
- `magic-pdf.template.json` - MinerU配置模板

## 特殊依赖和功能

### 可选依赖
- **LibreOffice**: DOC文件支持
- **MinerU**: 高级PDF解析，需要额外模型下载
- **OCR引擎**: 图片文本识别

### 关键功能模块
- **数据生命周期追踪**: 通过 `lifecycle_types.py` 实现解析、清洗、标注过程的生命周期管理
- **领域树功能**: 支持自定义领域分类树用于更精确的AI标注
- **多格式支持**: 通过 `ParserFactory` 统一管理各种文件格式解析器
- **智能缓存**: TTL缓存机制避免重复解析大文件

## 常见开发任务

### 添加新的文件格式解析器
1. 在 `datamax/parser/` 下创建 `{format}_parser.py`
2. 继承 `BaseParser` 类
3. 在 `ParserFactory.create_parser()` 中注册新格式
4. 添加对应的测试文件

### 扩展数据清洗功能
- 在 `data_cleaner.py` 中添加新的清洗策略
- 更新 `DataMax.clean_data()` 方法支持新策略

### 修改AI标注逻辑
- 主要逻辑在 `qa_generator.py` 中
- `DataMax.get_pre_label()` 提供高级接口

## 调试和故障排除

### 常见问题
- MinerU配置问题：检查 `magic-pdf.template.json` 配置
- 文件解析失败：查看日志，可能是缺少相关依赖
- 测试失败：确保所有依赖已正确安装

### 日志系统
项目使用 `loguru` 进行日志记录，关键操作都有相应的日志输出。
