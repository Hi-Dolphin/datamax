# DataMax 自动发布配置指南

本指南将帮助您配置GitHub Action自动发布流程，实现一键发布Python包到PyPI并创建GitHub Release。

## 📋 前置要求

### 1. PyPI账号设置

#### 创建PyPI API Token
1. 登录 [PyPI官网](https://pypi.org/)
2. 进入 Account Settings → API tokens
3. 点击 "Add API token"
4. 设置Token名称（例如：`datamax-github-action`）
5. 选择Scope为 "Entire account" 或指定项目
6. 复制生成的Token（格式：`pypi-xxxxx`）

#### 创建Test PyPI API Token（可选，用于测试）
1. 登录 [Test PyPI](https://test.pypi.org/)
2. 重复上述步骤创建Test PyPI Token

### 2. GitHub仓库设置

#### 配置Secrets
在GitHub仓库中设置以下Secrets：

1. 进入仓库 Settings → Secrets and variables → Actions
2. 添加以下Repository secrets：

| Secret名称 | 值 | 说明 |
|-----------|---|------|
| `PYPI_API_TOKEN` | `pypi-xxxxx` | PyPI API Token |
| `TEST_PYPI_API_TOKEN` | `pypi-xxxxx` | Test PyPI API Token（可选） |

#### 配置Environment（推荐）
为了增强安全性，建议创建Environment：

1. 进入仓库 Settings → Environments
2. 创建名为 `pypi` 的Environment
3. 设置保护规则（可选）：
   - Required reviewers（需要审核者）
   - Deployment branches（指定分支）
4. 在Environment中添加secrets

## 🚀 使用方法

### 方法1：手动触发发布

1. 进入GitHub仓库的 Actions 页面
2. 选择 "发布到PyPI和创建标签" 工作流
3. 点击 "Run workflow"
4. 选择参数：
   - **版本类型**：`major`（主版本）、`minor`（次版本）、`patch`（补丁版本）、`prerelease`（预发布版本）
   - **自定义版本号**：可选，如果提供则忽略版本类型

### 方法2：通过Git标签触发

```bash
# 创建标签
git tag v1.2.3

# 推送标签
git push origin v1.2.3
```

## 🔧 版本管理策略

### 语义化版本控制
项目采用 [Semantic Versioning](https://semver.org/) 规范：

- **MAJOR**（主版本）：不兼容的API修改
- **MINOR**（次版本）：向后兼容的功能性新增
- **PATCH**（补丁版本）：向后兼容的问题修正
- **PRERELEASE**（预发布）：预发布版本（如：1.0.0-rc.1234567890）

### 版本号规则
- 格式：`MAJOR.MINOR.PATCH[-PRERELEASE]`
- 示例：`1.2.3`、`2.0.0-rc.1234567890`

## 📦 发布流程详解

### 1. 验证和测试阶段
- ✅ 检出代码并设置Python环境
- ✅ 安装依赖和构建工具
- ✅ 自动版本管理（基于输入参数或Git标签）
- ✅ 代码质量检查
- ✅ 构建Python包
- ✅ 验证包的完整性和可安装性
- ✅ 检查PyPI版本冲突

### 2. PyPI发布阶段
- ✅ 预发布版本自动发布到Test PyPI
- ✅ 正式版本发布到PyPI
- ✅ 详细的发布日志

### 3. GitHub Release创建阶段
- ✅ 自动提交版本更新
- ✅ 创建Git标签
- ✅ 生成更新日志（基于Git提交记录）
- ✅ 创建GitHub Release
- ✅ 上传构建产物

### 4. 通知阶段
- ✅ 发布成功/失败通知
- ✅ 提供PyPI和GitHub链接

## 🛡️ 安全考虑

### 1. Token权限最小化
- PyPI Token设置为项目级别而非账户级别（如果可能）
- 定期轮换API Token

### 2. 环境保护
- 使用GitHub Environment保护生产环境
- 设置审核机制（可选）

### 3. 分支保护
- 保护主分支，要求PR审核
- 只允许特定分支触发发布

## ⚠️ 注意事项

### 1. 版本冲突处理
- 工作流会自动检查PyPI版本冲突
- 如果版本已存在，将跳过发布

### 2. 依赖问题
- 确保 `requirements.txt` 包含所有必要依赖
- 某些依赖可能需要系统级安装（如 LibreOffice）

### 3. 构建失败排查
- 检查 `setup.py` 配置
- 确认包结构正确
- 验证依赖兼容性

### 4. 权限问题
- 确保GitHub Token有足够权限创建Release
- 检查PyPI Token权限和有效性

## 🔧 自定义配置

### 修改Python版本
编辑 `.github/workflows/publish.yml`：

```yaml
env:
  PYTHON_VERSION: '3.11'  # 修改为所需版本
```

### 添加额外测试
在验证阶段添加更多测试步骤：

```yaml
- name: 运行单元测试
  run: |
    pytest tests/
```

### 自定义通知
修改通知阶段以集成Slack、邮件等：

```yaml
- name: Slack通知
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🖥️ 本地手动发布流程（每次迭代）

下表列出了 **每次版本迭代** 时，从代码冻结到 PyPI 发布需要执行的全部步骤和对应命令。建议在 *虚拟环境* 中操作，并确保已在 `~/.pypirc` 或环境变量中配置好 `TWINE_USERNAME/__token__` 与 `TWINE_PASSWORD/pypi-***`。

```
python3.11 -m venv venv311
source venv311/bin/activate
pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install build -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python scripts/bump_version.py patch
python -m build
pip install dist/pydatamax-*.whl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -c "import datamax;"
deactivate
```

| 阶段 | 目的 | 典型命令 |
|------|------|----------|
| 1. 代码冻结 | 合并功能分支，进入发布流程 | `git checkout main && git pull` |
| 2. 更新日志 | 编写 / 更新 `CHANGELOG.md` | 手动编辑文件 |
| 3. 质量检查 | 运行全部静态检查和单测 | `pre-commit run --all-files`<br>`pytest -q` |
| 4. 版本号递增 | 使用脚本递增版本（或手动指定） | `python scripts/bump_version.py patch`<br>可选 `major` / `minor` / 具体版本号 |
| 5. 提交&打标签 | 持久化版本变动 | `git add .`<br>`git commit -m "bump: 发布 vX.Y.Z" --no-verify`<br>`git tag vX.Y.Z` |
| 6. 构建包 | 生成 sdist 和 wheel | `python -m build` |
| 7. TestPyPI 验证 | 先上传到测试仓库并安装验证 | `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`<br>`pip install --index-url https://test.pypi.org/simple/ pydatamax==X.Y.Z --no-cache-dir -U` |
| 8. 正式发布 | 上传到 PyPI | `twine upload dist/*` |
| 9. 推送代码 | 同步远程仓库和标签 | `git push origin main --follow-tags` |
|10. 发布Release | 创建GitHub Release并上传构建产物 | 在GitHub网页端创建Release<br>或使用 `gh release create vX.Y.Z dist/* --title "vX.Y.Z" --notes-from-tag` |
|11. 清理工作 | 保持工作区整洁 | `rm -rf build dist *.egg-info` |

> 💡 **提示**
> • 若使用 **SSH** 方式避免输入 PAT，可运行 `git remote set-url origin git@github.com:<user>/<repo>.git`。
> • `twine` 会优先读取 `~/.pypirc`，也可通过环境变量：
>   ```bash
>   export TWINE_USERNAME="__token__"
>   export TWINE_PASSWORD="pypi-xxxxxxxxxxxxxxxxxxxx"
>   ```
> • 上传前可执行 `twine check dist/*` 确认元数据完整。
> • 注意：本项目在 PyPI 上发布的包名为 **pydatamax**，而非 datamax。

## 📚 相关资源

- [PyPI官方文档](https://packaging.python.org/)
- [GitHub Actions文档](https://docs.github.com/en/actions)
- [Python打包指南](https://packaging.python.org/tutorials/packaging-projects/)
- [Semantic Versioning](https://semver.org/)

## 🆘 故障排除

### 常见问题

#### 1. PyPI发布失败
```
错误：403 Forbidden
```
**解决方案**：
- 检查PyPI API Token是否正确
- 确认Token权限范围
- 验证包名称是否冲突

#### 2. 版本号格式错误
```
错误：Invalid version number
```
**解决方案**：
- 检查版本号格式是否符合PEP 440
- 确认setup.py中版本号更新正确

#### 3. 依赖安装失败
```
错误：Could not find a version that satisfies the requirement
```
**解决方案**：
- 检查requirements.txt中的依赖版本
- 考虑使用更宽松的版本约束
- 检查Python版本兼容性

## 📞 支持

如果您在配置过程中遇到问题，请：

1. 检查GitHub Actions日志获取详细错误信息
2. 参考本指南的故障排除部分
3. 在项目仓库创建Issue描述问题

---

**祝您发布顺利！** 🎉
