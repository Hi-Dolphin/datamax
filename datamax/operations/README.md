# Data Factory Operations API

后端基于 FastAPI + SQLAlchemy，提供数据工厂操作与统计看板所需的真实接口能力。
目前默认使用 sqlite+aiosqlite:///./operations.db，通过 DATAMAX_POSTGRES_DSN 可切换至任意 Async SQLAlchemy 兼容数据库（Postgres、MySQL 等）。
启动阶段会自动建表并注入一批示例数据，便于前端联调。

## 启动方式
`ash
pip install -e .[dev]
uvicorn datamax.operations.main:app --reload
`

### 关键环境变量
| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| DATAMAX_ENVIRONMENT | 环境标识（development/test/prod） | development |
| DATAMAX_API_V1_PREFIX | API 前缀 | /api/v1 |
| DATAMAX_POSTGRES_DSN | SQLAlchemy Async DSN，缺省为 SQLite 文件 | sqlite+aiosqlite:///./operations.db |
| DATAMAX_REDIS_URL | 预留缓存/队列连接 | 空 |

## 主要接口
auth 免鉴权，全部走 JSON：
- GET /api/v1/datasources：数据源列表，支持创建/编辑/删除
- GET /api/v1/datasets：数据集 CRUD & 刷新
- GET /api/v1/pipelines：流程编排，包含阶段配置与统计
- GET /api/v1/tasks：任务监控、运行记录
- GET /api/v1/metrics/dashboard：看板聚合指标（Totals + Stage Stats + Series）
- GET /api/v1/alerts：告警查询与确认
- GET /api/v1/system/config：系统配置占位

## 数据模型
位于 datamax/operations/models/，采用 SQLAlchemy ORM：
- DataSource：连接信息、标签、健康状态
- Dataset：关联数据源、刷新统计
- Pipeline/PipelineStage：流程及阶段定义，阶段内置 stats 字段供看板使用
- Task/TaskRun：任务调度与执行结果，TaskRun 保存 tokens/qpm 等指标
- Alert、MetricSummary：告警与历史指标缓存（可选）

## 指标聚合
services/metrics.py 会基于 TaskRun 与 PipelineStage.stats 计算看板所需 totals、阶段数据、吞吐序列以及 Prompt/Completion 占比。后续只需把真实任务运行数据入库即可驱动前端看板。

## 测试
`ash
pytest tests/test_operations_api.py
`
测试使用 SQLite 文件数据库并自动清理，覆盖健康检查、数据源 CRUD、数据集刷新、指标接口与告警确认等核心路径。

## 后续计划
- 接入 Alembic 迁移管理
- 补充 Redis/消息队列、gRPC 调用封装
- 引入鉴权与操作审计
- 根据业务落地更多筛选与分页能力
