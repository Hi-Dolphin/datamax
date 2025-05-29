# 变更日志 / Changelog

本文档记录了 PyDataMax 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布] - Unreleased

### 计划新增
- [ ] 更多文件格式支持（RTF、CSV等）
- [ ] 批量处理性能优化
- [ ] Web界面支持
- [ ] Docker容器化支持

## [0.1.14] - 2025-05-28

### 新增
- 🚀 将包名正式更改为 `pydatamax`
- ✨ 优化 DOC 文件解析支持
- ✨ 增强 DOCX 文件解析能力
- ✨ 改进 XLSX 文件解析性能
- 📚 添加本地手动发布流程文档

### 改进
- 🔧 更新依赖项版本范围，提高兼容性
- 📖 完善部署指南文档
- ⚡ 优化文件解析性能

### 修复
- 🐛 修复某些 DOC 文件解析失败的问题
- 🐛 修复 XLSX 文件中特殊字符处理问题

## [0.1.13] - 2025-05-28

### 新增
- 📦 更新 PyPI 发布配置

### 改进
- 🔧 优化依赖管理
- 📚 更新文档和示例代码

### 技术改进
- 🛠️ 重构部分核心模块
- 🧪 增强测试覆盖率

## [0.1.11] - 2025-05-26

### 新增
- 🎯 GPU 版本支持发布
- ⚡ 增强多格式文件解析能力
- 🤖 完善 AI 标注功能

### 核心功能
- 📄 **多格式支持**: PDF、DOCX/DOC、PPT/PPTX、XLS/XLSX、HTML、EPUB、TXT、图片等
- 🧹 **智能清洗**: 三层清洗机制，包含异常检测、隐私保护、文本过滤
- 🤖 **AI 标注**: 基于 LLM 的自动数据标注和预标记
- ⚡ **批量处理**: 高效的多文件并行处理
- 🎯 **易于集成**: 简洁的 API 设计，开箱即用

### 解析器支持
- 📝 **文档类**: `.pdf`, `.docx`, `.doc` - 支持 OCR，Markdown 转换
- 📊 **表格类**: `.xlsx`, `.xls` - 结构化数据提取
- 🎨 **演示类**: `.pptx`, `.ppt` - 幻灯片内容提取
- 🌐 **网页类**: `.html`, `.epub` - 标签解析
- 🖼️ **图像类**: `.jpg`, `.png`, `.jpeg` - OCR 文字识别
- 📄 **文本类**: `.txt` - 自动编码检测

### 高级功能
- 🔬 **MinerU 集成**: 高级 PDF 解析能力
- 📝 **Markdown 转换**: Word 文档转 Markdown
- 👁️ **OCR 支持**: 图像文字识别
- 🧼 **数据清洗**: 异常数据处理、隐私信息脱敏、文本过滤规范化
- 🏷️ **智能标注**: 自定义标注任务，支持分块处理和并发

### 环境支持
- 🐍 Python >= 3.10
- 🖥️ 支持 Windows、macOS、Linux
- 📦 可选依赖：LibreOffice（DOC 支持）、MinerU（高级 PDF 解析）

---

## 版本说明

- **主版本号（Major）**: 不兼容的 API 修改
- **次版本号（Minor）**: 向下兼容的功能性新增
- **修订号（Patch）**: 向下兼容的问题修复

## 贡献指南

如果您发现了 bug 或有新功能建议，欢迎：

1. 🐛 提交 [Issue](https://github.com/Hi-Dolphin/datamax/issues)
2. 🔧 提交 [Pull Request](https://github.com/Hi-Dolphin/datamax/pulls)
3. 📧 联系我们：cy.kron@foxmail.com

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

📌 **说明**: 
- 版本号遵循 [语义化版本规范](https://semver.org/)
- 每个版本都包含新增、改进、修复等分类说明 