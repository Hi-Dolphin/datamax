# 最佳实践：文档文本模态 QA 生成 Pipeline

`examples/scripts/generate_qa.py` 提供了一个端到端的文本模态 QA 数据生成流程。本指南总结了实战中最常见的配置与操作步骤，帮助你快速完成环境准备、数据输入与结果落盘。

## 背景说明

该脚本会：

- 从本地或华为 OBS 拉取待标注的文档；
- 调用 DataMax 的 `get_pre_label` 能力批量生成问答；
- 将输出保存到 `train/` 目录下，便于直接用于微调或评估。

默认根目录由环境变量 `DATAMAX_ROOT` 控制（默认 `/mnt/f/datamax`），所有数据路径都基于该根目录解析。

## 环境变量配置

在运行脚本前，请准备好以下环境变量：

| 变量 | 说明 | 必需 | 备注 |
| --- | --- | --- | --- |
| `DASHSCOPE_API_KEY` | 达摩盘推理服务的 API Key | 是 | 若留空将无法调用模型 |
| `DASHSCOPE_BASE_URL` | 模型服务地址 | 是 | 请根据实际部署填写 |
| `QA_MODEL` | 模型名称或版本 | 是 | 例如 `qwen-turbo` |
| `DATAMAX_ROOT` | DataMax 项目根路径 | 否 | 相对路径会被自动转换为绝对路径 |
| `QA_INPUT_SOURCE` | 数据来源：`local` 或 `obs` | 否 | 默认 `local` |
| `OBS_ENDPOINT` | OBS 访问地址 | 当 `QA_INPUT_SOURCE=obs` 时必填 | |
| `OBS_ACCESS_KEY_ID` / `OBS_ACCESS_KEY_SECRET` | OBS 访问凭证 | 当 `QA_INPUT_SOURCE=obs` 时必填 | |
| `OBS_BUCKET_NAME` | OBS Bucket 名称 | 当 `QA_INPUT_SOURCE=obs` 时必填 | |
| `OBS_DOWNLOAD_DIR` | OBS 数据下载目录 | 否 | 相对路径基于 `DATAMAX_ROOT` |
| `OBS_PREFIX` | OBS 对象前缀过滤 | 否 | 为空时拉取整个 Bucket |

> ✅ 建议使用 `.env` 或 CI/CD 系统的机密变量统一管理上述配置。

## 本地文件组织

当 `QA_INPUT_SOURCE` 为 `local` 时，脚本默认从 `${DATAMAX_ROOT}/data/Step1/` 目录递归搜索文件。你可以按照如下结构组织数据：

```
${DATAMAX_ROOT}/
├─ data/
│  └─ Step1/
│     ├─ contract/
│     │  ├─ doc1.md
│     │  └─ doc2.docx
│     └─ report/
│        └─ q1.pdf
└─ examples/scripts/generate_qa.py
```

如需自定义目录，可通过修改脚本顶部的常量 `local_dataset_dir`。

## 运行步骤

```bash
python examples/scripts/generate_qa.py
```

脚本将自动：

1. 创建 `${DATAMAX_ROOT}/train/` 目录；
2. 逐个遍历输入文件并生成预标注 QA；
3. 将输出保存为 `<原文件名>_train.json`（或同名目录下的 JSON 序列）。

默认仅处理首个文件，可根据需求移除脚本中的 `break` 语句以处理全部文件。

## OBS 数据拉取策略

当设置 `QA_INPUT_SOURCE=obs` 时：

- 下载目录优先使用 `OBS_DOWNLOAD_DIR`，否则落在 `${DATAMAX_ROOT}/obs_downloads/`；
- `OBS_PREFIX` 可用于限定某一子目录或命名空间；
- OBS 凭证缺失时脚本会立即报错退出，确保配置安全可靠。

下载完成后，脚本会继续使用统一的 DataMax 流程生成 QA。

## 结果验证与常见问题

- **生成结果为空**：确认输入文件包含文本内容，或者调高 `question_number`。
- **无法解析根目录**：检查 `DATAMAX_ROOT` 是否为合法路径，特别是在 Windows 环境下建议使用绝对路径。
- **OBS 下载失败**：核对 Endpoint、Bucket 名称和访问凭证，必要时使用 `datamax.loader.core.DataLoader` 的独立脚本进行连通性测试。

## 进一步集成

- 可将脚本与定时任务结合，实现每日自动增量生成 QA。
- 结合 CI/CD，在合并新文档时触发脚本，生成新的训练样本。
- 若需自定义输出结构，参照 `DataMax.save_label_data` 编写自定义持久化逻辑。
