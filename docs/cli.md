# CLI 参考

## 启动
```bash
python -m datamax.cli.main --help
python -m datamax.cli.main status
```

## 常用命令（示例）

### 解析单文件
```bash
python -m datamax.cli.main parse \
  --input a.pdf --to-markdown --use-mineru \
  --output out.json
```

### 批量解析
```bash
python -m datamax.cli.main batch \
  --input-dir ./docs --pattern "*.pdf" \
  --to-markdown --use-mineru \
  --output ./out
```

### 清洗
```bash
python -m datamax.cli.main clean \
  --text "原始文本" --pipeline abnormal,filter,private
```

### 文本 QA 生成
```bash
python -m datamax.cli.main qa \
  --input a.txt \
  --api-key $DASHSCOPE_API_KEY \
  --base-url $DASHSCOPE_BASE_URL \
  --model qwen-max \
  --question-number 10
```

### 多模态 QA 生成
```bash
python -m datamax.cli.main multimodal \
  --input with_images.md \
  --api-key $OPENAI_API_KEY \
  --model gpt-4o
```

### 爬虫
```bash
python -m datamax.cli.main crawler crawl "航运" --engine auto -o result.json
```

## 日志
- `-v` 启用调试日志
- `-q` 静默模式

