# 扩展指南

## 新文件解析器
1) 在 `datamax/parser` 新增 `xxx_parser.py`，实现类如 `XxxParser(BaseLife)`，输出使用 `MarkdownOutputVo`
2) 在 `ParserFactory` 注册扩展名到模块/类的映射（`datamax/core.py`）
3) 保持统一生命周期事件（开始/完成/失败）

## 新爬虫引擎
1) 继承 `BaseCrawler` 实现爬取逻辑与（可选）异步版本
2) 在 `crawler_factory` 注册类型与 URL 模式
3) 通过 `StorageAdapter` 统一落盘

## 存储适配器
1) 参考 `LocalStorageAdapter` 实现 `save/load/list/delete/exists`
2) 在 `create_storage_adapter` 中挂接自定义 provider（S3/GCS/Azure 等）

## 清洗/评估自定义
- 在 `cleaner/evaluator` 增加类与组合；保持“单向数据流”与生命周期事件

## 最佳实践
- 解析 → 清洗 → 标注 → 评估 分层清晰、彼此解耦
- 所有阶段补齐 `lifecycle`，便于审计与可追溯
