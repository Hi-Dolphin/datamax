#!/usr/bin/env python3
"""
DataMax 版本管理脚本

用法:
    python scripts/bump_version.py patch      # 增加补丁版本 (0.1.11 -> 0.1.12)
    python scripts/bump_version.py minor      # 增加次版本 (0.1.11 -> 0.2.0)
    python scripts/bump_version.py major      # 增加主版本 (0.1.11 -> 1.0.0)
    python scripts/bump_version.py 1.2.3      # 设置为指定版本
"""

import argparse
import re
import sys
from pathlib import Path


def get_current_version():
    """从setup.py获取当前版本"""
    setup_py = Path(__file__).parent.parent / "setup.py"

    if not setup_py.exists():
        raise FileNotFoundError("找不到 setup.py 文件")

    content = setup_py.read_text(encoding="utf-8")
    version_match = re.search(r"version=['\"]([^'\"]+)['\"]", content)

    if not version_match:
        raise ValueError("无法在 setup.py 中找到版本号")

    return version_match.group(1)


def update_version(new_version):
    """更新setup.py中的版本号"""
    setup_py = Path(__file__).parent.parent / "setup.py"
    content = setup_py.read_text(encoding="utf-8")

    # 替换版本号
    new_content = re.sub(
        r"version=['\"]([^'\"]+)['\"]", f"version='{new_version}'", content
    )

    setup_py.write_text(new_content, encoding="utf-8")
    print(f"✅ 已更新 setup.py 中的版本号: {new_version}")


def parse_version(version_str):
    """解析版本号为(major, minor, patch)元组"""
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"版本号格式错误: {version_str}")

    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        raise ValueError(f"版本号必须是数字: {version_str}")


def bump_version(current_version, bump_type):
    """根据类型增加版本号"""
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        # 直接设置版本号
        parse_version(bump_type)  # 验证格式
        return bump_type


def main():
    parser = argparse.ArgumentParser(description="DataMax 版本管理工具")
    parser.add_argument("version", help="版本类型 (major/minor/patch) 或具体版本号 (如: 1.2.3)")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要更新的版本，不实际修改文件")

    args = parser.parse_args()

    try:
        # 获取当前版本
        current_version = get_current_version()
        print(f"📦 当前版本: {current_version}")

        # 计算新版本
        new_version = bump_version(current_version, args.version)
        print(f"🚀 新版本: {new_version}")

        if args.dry_run:
            print("🔍 试运行模式，未实际修改文件")
        else:
            # 更新版本
            update_version(new_version)
            print("✨ 版本更新完成！")
            print("\n下一步操作:")
            print(
                "1. 提交更改: git add setup.py && "
                f"git commit -m 'bump: 版本更新至 v{new_version}'"
            )
            print(f"2. 创建标签: git tag v{new_version}")
            print(f"3. 推送标签: git push origin v{new_version}")
            print("4. 或者使用 GitHub Actions 手动触发发布")

    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
