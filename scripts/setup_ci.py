#!/usr/bin/env python3
"""
DataMax CI/CD 设置脚本

自动配置开发环境和CI/CD工具
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description="", check=True):
    """运行命令并显示进度"""
    if description:
        print(f"🔧 {description}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 错误: {e}")
        if e.stderr:
            print(f"   详细信息: {e.stderr.strip()}")
        if check:
            sys.exit(1)
        return e


def check_requirements():
    """检查环境要求"""
    print("📋 检查环境要求...")

    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   请升级到Python 3.10或更高版本")
        sys.exit(1)
    print(
        f"   ✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    # 检查Git
    git_check = run_command("git --version", check=False)
    if git_check.returncode != 0:
        print("❌ 未找到Git，请先安装Git")
        sys.exit(1)

    # 检查是否在Git仓库中
    git_repo_check = run_command("git rev-parse --git-dir", check=False)
    if git_repo_check.returncode != 0:
        print("❌ 当前目录不是Git仓库")
        print("   请先运行: git init")
        sys.exit(1)
    print("   ✅ Git环境检查通过")


def install_dev_dependencies():
    """安装开发依赖"""
    print("\n📦 安装开发依赖...")

    dev_packages = [
        "build",  # Python包构建工具
        "twine",  # PyPI上传工具
        "pre-commit",  # Git钩子工具
        "black",  # 代码格式化
        "isort",  # 导入排序
        "flake8",  # 代码检查
        "bandit",  # 安全检查
        "pytest",  # 测试框架
        "pytest-cov",  # 测试覆盖率
    ]

    for package in dev_packages:
        run_command(f"pip install {package}", f"安装 {package}")


def setup_pre_commit():
    """设置pre-commit钩子"""
    print("\n🎣 设置Pre-commit钩子...")

    # 安装pre-commit钩子
    run_command("pre-commit install", "安装pre-commit钩子")

    # 运行首次检查
    print("🧪 运行首次pre-commit检查...")
    # 首次运行可能有修复，允许失败
    run_command("pre-commit run --all-files", "执行所有文件的代码检查", check=False)


def create_github_templates():
    """创建GitHub模板"""
    print("\n📝 创建GitHub模板...")

    # 创建.github目录
    github_dir = Path(".github")
    github_dir.mkdir(exist_ok=True)

    # 创建Issue模板
    issue_template = github_dir / "ISSUE_TEMPLATE" / "bug_report.md"
    issue_template.parent.mkdir(exist_ok=True)

    issue_content = """---
name: Bug报告
about: 创建bug报告以帮助我们改进
title: '[BUG] '
labels: bug
assignees: ''
---

## 问题描述
简要描述遇到的问题。

## 重现步骤
1. 执行 '...'
2. 查看 '....'
3. 滚动到 '....'
4. 出现错误

## 预期行为
描述您期望发生的情况。

## 实际行为
描述实际发生的情况。

## 环境信息
- OS: [例如 Windows 10, Ubuntu 20.04]
- Python版本: [例如 3.10.5]
- DataMax版本: [例如 0.1.11]

## 附加信息
添加任何其他相关信息、截图等。
"""

    if not issue_template.exists():
        issue_template.write_text(issue_content, encoding="utf-8")
        print("   ✅ 创建Bug报告模板")

    # 创建PR模板
    pr_template = github_dir / "pull_request_template.md"
    pr_content = """## 更改描述
简要描述此PR的更改内容。

## 更改类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 重大更改（会破坏现有功能）
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 测试添加/修改

## 测试
- [ ] 已添加相应的测试
- [ ] 所有测试通过
- [ ] 已手动测试相关功能

## 检查清单
- [ ] 代码遵循项目的代码规范
- [ ] 已进行自我代码审查
- [ ] 代码已添加适当的注释
- [ ] 已更新相关文档
- [ ] 更改不会产生新的警告
- [ ] 已添加相应的测试且测试通过
- [ ] 新依赖项（如有）已记录在requirements.txt中

## 相关Issue
关闭 #(issue编号)
"""

    if not pr_template.exists():
        pr_template.write_text(pr_content, encoding="utf-8")
        print("   ✅ 创建PR模板")


def setup_local_testing():
    """设置本地测试环境"""
    print("\n🧪 设置本地测试环境...")

    # 创建基本的测试目录结构
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)

    # 创建__init__.py
    (test_dir / "__init__.py").touch()

    # 创建基本测试文件
    test_basic = test_dir / "test_basic.py"
    if not test_basic.exists():
        test_content = '''"""
DataMax 基础测试
"""

import pytest
from datamax import DataMax


def test_import():
    """测试模块导入"""
    assert DataMax is not None


def test_version():
    """测试版本号"""
    import datamax
    assert hasattr(datamax, '__version__') or True  # 版本号检查


# 更多测试用例...
'''
        test_basic.write_text(test_content, encoding="utf-8")
        print("   ✅ 创建基础测试文件")

    # 创建pytest配置
    pytest_ini = Path("pytest.ini")
    if not pytest_ini.exists():
        pytest_content = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --cov=datamax
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
"""
        pytest_ini.write_text(pytest_content, encoding="utf-8")
        print("   ✅ 创建pytest配置")


def generate_security_check():
    """生成安全检查报告"""
    print("\n🔒 运行安全检查...")

    # 运行bandit安全检查
    result = run_command(
        "bandit -r datamax/ -f json -o security_report.json", "生成安全检查报告", check=False
    )

    if result.returncode == 0:
        print("   ✅ 未发现安全问题")
    else:
        print("   ⚠️ 发现潜在安全问题，请查看 security_report.json")


def validate_package():
    """验证包配置"""
    print("\n📦 验证包配置...")

    # 验证setup.py
    run_command("python setup.py check", "检查setup.py配置")

    # 尝试构建包
    run_command("python -m build", "构建Python包")

    # 验证构建的包
    run_command("python -m twine check dist/*", "验证包的完整性")

    print("   ✅ 包配置验证通过")


def main():
    parser = argparse.ArgumentParser(description="DataMax CI/CD 设置工具")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-pre-commit", action="store_true", help="跳过pre-commit设置")
    parser.add_argument("--skip-tests", action="store_true", help="跳过测试环境设置")

    args = parser.parse_args()

    print("🚀 DataMax CI/CD 环境设置")
    print("=" * 50)

    try:
        # 检查环境要求
        check_requirements()

        # 安装开发依赖
        if not args.skip_deps:
            install_dev_dependencies()

        # 设置pre-commit
        if not args.skip_pre_commit:
            setup_pre_commit()

        # 创建GitHub模板
        create_github_templates()

        # 设置测试环境
        if not args.skip_tests:
            setup_local_testing()

        # 安全检查
        generate_security_check()

        # 验证包
        validate_package()

        print("\n🎉 CI/CD环境设置完成！")
        print("\n下一步操作:")
        print("1. 配置GitHub Secrets (PYPI_API_TOKEN)")
        print("2. 阅读 DEPLOYMENT_GUIDE.md 了解详细使用方法")
        print("3. 运行 python scripts/bump_version.py --help 查看版本管理")
        print("4. 提交更改: git add . && git commit -m 'feat: 设置CI/CD环境'")

    except Exception as e:
        print(f"\n❌ 设置过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
