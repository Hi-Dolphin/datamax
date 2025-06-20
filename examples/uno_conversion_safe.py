"""
安全的 UNO API 文档转换示例

这个版本避免了 UNO 与其他库的导入冲突
"""

import logging
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

# 确保导入本地开发版本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def multiprocess_worker(file_path, use_uno, result_queue):
    """
    multiprocessing 工作函数（必须在顶层以支持 pickle）
    """
    try:
        # 在这个进程中才导入可能冲突的模块
        from datamax.parser.doc_parser import DocParser
        from datamax.parser.docx_parser import DocxParser
        from datamax.parser.ppt_parser import PPtParser

        if file_path.endswith(".doc"):
            parser = DocParser(file_path, use_uno=use_uno)
        elif file_path.endswith(".docx"):
            parser = DocxParser(file_path, use_uno=use_uno)
        elif file_path.endswith(".ppt"):
            parser = PPtParser(file_path, use_uno=use_uno)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")

        result = parser.parse(file_path)
        result_queue.put(("success", result))

    except Exception as e:
        result_queue.put(("error", str(e)))


def convert_document_subprocess(file_path: str, use_uno: bool = False):
    """
    在子进程中转换文档，避免 UNO 导入冲突
    """
    # 创建一个新的 Python 脚本来执行转换
    script = f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')

from datamax.parser.doc_parser import DocParser
from datamax.parser.docx_parser import DocxParser
from datamax.parser.ppt_parser import PPtParser

file_path = '{file_path}'
use_uno = {use_uno}

try:
    if file_path.endswith('.doc'):
        parser = DocParser(file_path, use_uno=use_uno)
    elif file_path.endswith('.docx'):
        parser = DocxParser(file_path, use_uno=use_uno)
    elif file_path.endswith('.ppt'):
        parser = PPtParser(file_path, use_uno=use_uno)
    else:
        raise ValueError(f"Unsupported file type: {{file_path}}")

    result = parser.parse(file_path)
    print("CONVERSION_SUCCESS")
except Exception as e:
    print(f"CONVERSION_ERROR: {{str(e)}}")
    sys.exit(1)
"""

    # 运行子进程
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    if "CONVERSION_SUCCESS" in result.stdout:
        return True
    else:
        error_msg = result.stderr or result.stdout
        logger.error(f"转换失败: {error_msg}")
        return False


def convert_document_multiprocess(file_path: str, use_uno: bool = True):
    """
    使用 multiprocessing 在独立进程中转换文档
    """
    # 使用 spawn 方法创建全新的进程
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    process = ctx.Process(
        target=multiprocess_worker, args=(file_path, use_uno, result_queue)
    )
    process.start()
    process.join(timeout=60)  # 60秒超时

    if process.is_alive():
        process.terminate()
        process.join()
        return None, "转换超时"

    if not result_queue.empty():
        status, result = result_queue.get()
        return (result, None) if status == "success" else (None, result)
    else:
        return None, "未知错误"


def test_safe_conversion():
    """测试安全的文档转换"""
    # 测试文件列表
    test_files = [
        "examples/00b33cb2-3cce-40a1-95b7-de7d6935bf66.docx",
        "examples/EAM资产管理系统应急预案2020-02(新EAM).docx",
        "examples/中远海运科技_会议纪要_开尔唯OCP&BMS项目_20230523_BMS财务部应收会计调研.docx",
        "examples/远海码头官网应急预案2020-2.docx",
    ]

    # 找到存在的测试文件
    available_files = [f for f in test_files if os.path.exists(f)]

    if not available_files:
        logger.warning("没有找到测试文件")
        return

    logger.info(f"找到 {len(available_files)} 个测试文件")

    # 测试方法1: 使用 subprocess
    logger.info("\n" + "=" * 60)
    logger.info("方法1: 使用 subprocess 隔离")
    logger.info("=" * 60)

    for file_path in available_files[:1]:  # 只测试第一个文件
        logger.info(f"\n转换文件: {file_path}")
        start_time = time.time()

        success = convert_document_subprocess(file_path, use_uno=True)

        elapsed = time.time() - start_time
        if success:
            logger.info(f"✅ 转换成功 (耗时: {elapsed:.2f}秒)")
        else:
            logger.error(f"❌ 转换失败 (耗时: {elapsed:.2f}秒)")

    # 测试方法2: 使用 multiprocessing
    logger.info("\n" + "=" * 60)
    logger.info("方法2: 使用 multiprocessing 隔离")
    logger.info("=" * 60)

    for file_path in available_files[:1]:  # 只测试第一个文件
        logger.info(f"\n转换文件: {file_path}")
        start_time = time.time()

        result, error = convert_document_multiprocess(file_path, use_uno=True)

        elapsed = time.time() - start_time
        if result:
            logger.info(f"✅ 转换成功 (耗时: {elapsed:.2f}秒)")
            logger.info(f"   标题: {result.get('title', 'N/A')}")
            logger.info(f"   内容长度: {len(result.get('content', ''))} 字符")
        else:
            logger.error(f"❌ 转换失败: {error} (耗时: {elapsed:.2f}秒)")


def test_traditional_conversion():
    """测试传统方式（不使用 UNO）"""
    logger.info("\n" + "=" * 60)
    logger.info("测试传统方式（不使用 UNO）")
    logger.info("=" * 60)

    # 直接导入，因为不使用 UNO 就不会有冲突
    from datamax.parser.docx_parser import DocxParser

    test_file = None
    for f in ["examples/00b33cb2-3cce-40a1-95b7-de7d6935bf66.docx", "test.docx"]:
        if os.path.exists(f):
            test_file = f
            break

    if test_file:
        logger.info(f"测试文件: {test_file}")
        start_time = time.time()

        try:
            parser = DocxParser(test_file, use_uno=False)
            result = parser.parse(test_file)

            elapsed = time.time() - start_time
            logger.info(f"✅ 转换成功 (耗时: {elapsed:.2f}秒)")
            logger.info(f"   标题: {result.get('title', 'N/A')}")
            logger.info(f"   内容长度: {len(result.get('content', ''))} 字符")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ 转换失败: {str(e)} (耗时: {elapsed:.2f}秒)")
    else:
        logger.warning("没有找到测试文件")


if __name__ == "__main__":
    # 检查 UNO 可用性
    from datamax.utils.uno_handler import HAS_UNO

    logger.info("文档转换测试（避免 UNO 导入冲突）")
    logger.info(f"UNO 可用性: {'✅ 可用' if HAS_UNO else '❌ 不可用'}")

    # 测试传统方式
    test_traditional_conversion()

    # 测试安全的 UNO 转换
    if HAS_UNO:
        test_safe_conversion()
    else:
        logger.warning("\n跳过 UNO 测试，因为 UNO 不可用")
