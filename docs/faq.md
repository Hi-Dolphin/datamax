# FAQ

**DOC/DOCX/PPT 解析失败？**
- 请安装 LibreOffice 与 `python-uno`，确保 `soffice` 可执行；缺失时会回退到子进程方案，但稳定性与速度较差。

**PDF 高保真解析？**
- 启用 `use_mineru=True`，并执行 `scripts/download_models.py` 下载/配置模型。

**OCR/视觉模型调用失败或无响应？**
- 校验 `DASHSCOPE_API_KEY`/`OPENAI_API_KEY` 与 `base_url`；`base_url` 未包含 `/chat/completions` 时会自动补全。

**多模态 QA 无法找到图片？**
- 确保 Markdown 中图片路径可访问；评估流水线会将图片路径替换为绝对路径。

**如何导出训练集？**
- 使用 `save_label_data(..., "train")` 生成 `.jsonl`；或自定义写出格式。

**领域树必须启用吗？**
- 可选。可提供 `custom_domain_tree` 或关闭 `use_tree_label`，流程将回退为纯文本生成策略。

