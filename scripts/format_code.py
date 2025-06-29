#!/usr/bin/env python3
"""
代码自动格式化和检查脚本
使用方法: python scripts/format_code.py [--check-only]
"""
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} - 失败")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} - 错误: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='代码格式化和检查工具')
    parser.add_argument('--check-only', action='store_true', 
                       help='仅检查，不自动修复')
    args = parser.parse_args()
    
    # 确保在项目根目录
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    print("🚀 DataMax 代码质量工具")
    print("=" * 50)
    
    success = True
    
    if args.check_only:
        # 仅检查模式
        commands = [
            ("black --check datamax/", "Black格式检查"),
            ("isort --check-only datamax/ --profile black", "导入排序检查"),
            ("flake8 datamax/ --max-line-length=88 --extend-ignore=E203,W503", "Flake8代码检查"),
            ("bandit -r datamax/ -f json", "安全检查"),
        ]
    else:
        # 自动修复模式
        commands = [
            ("black datamax/", "Black自动格式化"),
            ("isort datamax/ --profile black", "导入语句自动排序"),
            ("autopep8 --in-place --recursive datamax/", "AutoPEP8自动修复"),
            ("flake8 datamax/ --max-line-length=88 --extend-ignore=E203,W503", "Flake8最终检查"),
        ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有检查通过！代码质量良好。")
    else:
        print("⚠️ 发现问题，请查看上述输出并修复。")
        sys.exit(1)

if __name__ == "__main__":
    main() 