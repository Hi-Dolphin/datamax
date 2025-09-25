# 评估

## 文本质量评估
- 指标：BERTScore、ROUGE、BLEU、Self‑CIDEr（语义多样性）

```python
from datamax.evaluator import TextQualityEvaluator

e = TextQualityEvaluator(lang="zh")
bertscore = e.evaluate_bertscore(["生成句子"], ["参考句子"])  # {precision, recall, f1}
```

> 需要安装相应依赖（如 `bert-score`, `rouge-score`, `sacrebleu`, `pycocoevalcap`）。

## 多模态一致性
- 指标：CLIPScore（DashScope 多模态嵌入）、VQA（OpenAI 兼容接口）

```python
from datamax.evaluator import MultimodalConsistencyEvaluator as MCE

m = MCE(
  clip_model_name="qwen-vl-clip",
  vqa_model_name="qwen-vl-max",
  dashscope_api_key="${DASHSCOPE_API_KEY}"
)
score = m.evaluate_clip_score("img.png", "这张图描述了...")
```

## 端到端筛选（范式）
- 解析 PDF → 输出含绝对图片路径的 Markdown
- 生成多模态 QA → 计算 CLIPScore/VQA 分数
- 设阈值过滤低质量样本 → 导出

## 示例脚本

--8<-- "examples/scripts/evaluate_text.py"
