# 爬虫

## 引擎与模式
- 引擎：`web`（通用网页）、`arxiv`（学术论文），或 `engine="auto"` 聚合多引擎
- 模式：支持同步/异步（内部已封装聚合接口）

## 统一调用（一行式）
```python
from datamax.crawler import crawl

# 指定引擎
data = crawl("https://example.com", engine="web")

# 聚合所有可用引擎
mix = crawl("航运", engine="auto")
```

## 结果解析（转 Markdown）
```python
from datamax.parser import CrawlerParser

md_vo = CrawlerParser(file_path="crawl_result.json").parse()
print(md_vo.to_dict()["content"])  # 结构化 Markdown
```

## 存储适配
- 本地：JSON/YAML（`LocalStorageAdapter`）
- 云：`CloudStorageAdapter` 占位（可扩展 S3/GCS/Azure）

## 小贴士
- `engine="auto"` 会并发调用已注册的引擎并汇总成功/失败结果
- 可结合解析/清洗/标注形成完整链路（爬取 → 解析 → 清洗 → 标注）

## 示例脚本

--8<-- "examples/scripts/crawl_web.py"
