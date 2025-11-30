# 数据工厂操作后端概览

本说明聚焦新版（非 Mock）实现，涵盖数据库模型、接口能力与后续演进方向。

## 1. 模块结构
`
datamax/operations/
├── api/             # FastAPI 路由拆分 (v1)
├── core/            # 配置、日志、数据库会话
├── models/          # SQLAlchemy ORM 定义
├── repositories/    # CRUD 封装
├── services/        # 统计聚合、数据初始化
├── main.py          # 应用入口（启动建表 + 演示数据）
└── README.md        # 使用说明
`

## 2. 已提供能力
| 模块 | 端点 | 说明 |
| --- | --- | --- |
| 数据源 | GET/POST/PUT/DELETE /api/v1/datasources | 保存连接信息、标签、健康状态，密码字段默认脱敏返回 |
| 数据集 | GET/POST/PUT/DELETE /api/v1/datasets<br>POST /api/v1/datasets/{id}/refresh | 关联数据源，维护 Schema、记录数、刷新时间 |
| 流程编排 | GET/POST/PUT/DELETE /api/v1/pipelines | 阶段信息存入 pipeline_stages.stats，供看板展示 |
| 任务监控 | GET/POST/PUT/DELETE /api/v1/tasks<br>GET /api/v1/tasks/{id}/runs | 记录任务、运行明细，Run 中包含 tokens / requests / duration 等指标 |
| 指标看板 | GET /api/v1/metrics/dashboard | 汇总 Tokens、QPM、吞吐序列、阶段统计、占比等数据 |
| 告警 | GET /api/v1/alerts、POST /api/v1/alerts/{id}/ack | 查询并确认告警 |
| 系统配置 | GET/PUT /api/v1/system/config | 当前为内存态配置占位 |
| 健康检查 | GET /health | 基础存活检查 |

## 3. 数据来源
- 默认使用 SQLite (sqlite+aiosqlite:///./operations.db)，方便快速起服；设置 DATAMAX_POSTGRES_DSN 可切换至 Postgres/MySQL 等。
- 启动阶段自动建表并调用 services.seed_demo_data() 注入示例数据，满足前端联调所需的真实结构。
- 指标聚合 (services/metrics.compute_dashboard_payload) 会读取 TaskRun 数值与 PipelineStage.stats 字段，产出看板 totals / stages / series / token split，前端无需再 Mock。

## 4. 测试
`
pytest tests/test_operations_api.py
`
- 使用独立的 SQLite 文件 	est_operations.db，每个用例前创建、结束后清理。
- 覆盖路由：健康检查、数据源 CRUD、数据集创建与刷新、指标计算、告警确认等。

## 5. 后续演进建议
1. 加入 Alembic 迁移，按环境管理表结构。
2. 扩展 Datasource 配置（超时、连接池、指标 SQL 模板等），实现真正的跨库汇聚。
3. 引入 Redis/消息队列 & gRPC 客户端，打通任务分发与模型服务联调。
4. 做好权限与审计（角色、操作日志、变更历史）。
5. 为列表接口补充查询条件、分页、排序，匹配前端 TODO。

最新接口/模型以代码为准，可结合 README 及 PRD（docs/prd-data-factory-dashboard.md）继续落地。
