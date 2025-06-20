#!/usr/bin/env python3
"""
文档转换性能对比测试

对比传统 soffice 命令行方式和 UNO API 方式的性能
"""

from loguru import logger
import os
import sys
import time
from pathlib import Path

# 确保导入本地开发版本
sys.path.insert(0, os.path.abspath("."))


def test_traditional_method(files):
    """测试传统方式（soffice 命令行）"""
    logger.info("🔄 测试传统方式（soffice 命令行）")
    logger.info("=" * 50)

    from datamax.parser.docx_parser import DocxParser

    times = []
    for file_path in files:
        if not os.path.exists(file_path):
            continue

        logger.info(f"转换文件: {Path(file_path).name}")
        start_time = time.time()

        try:
            parser = DocxParser(file_path, use_uno=False)
            result = parser.parse(file_path)

            elapsed = time.time() - start_time
            times.append(elapsed)

            content_length = len(result.get("content", ""))
            logger.info(f"✅ 成功 - 耗时: {elapsed:.2f}秒, 内容: {content_length} 字符")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ 失败 - 耗时: {elapsed:.2f}秒, 错误: {str(e)}")

    if times:
        avg_time = sum(times) / len(times)
        logger.info(f"📊 传统方式平均耗时: {avg_time:.2f}秒")
        return avg_time
    return 0


def test_uno_method_if_available(files):
    """测试 UNO 方式（如果可用）"""
    logger.info("\n🚀 测试 UNO 方式")
    logger.info("=" * 50)

    try:
        from datamax.utils.uno_handler import HAS_UNO, get_uno_manager

        if not HAS_UNO:
            logger.warning("❌ UNO 不可用，跳过测试")
            return 0

        # 检查 LibreOffice 服务是否运行
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 2002))
        sock.close()

        if result != 0:
            logger.warning("❌ LibreOffice 服务未运行，跳过测试")
            logger.info("提示: 先启动服务:")
            logger.info(
                'soffice --headless --invisible --accept="socket,host=localhost,port=2002;urp;StarOffice.ComponentContext" &'
            )
            return 0

        # 连接 UNO 管理器
        manager = get_uno_manager()
        logger.info("✅ UNO 服务连接成功")

        from datamax.parser.docx_parser import DocxParser

        times = []
        for file_path in files:
            if not os.path.exists(file_path):
                continue

            logger.info(f"转换文件: {Path(file_path).name}")
            start_time = time.time()

            try:
                parser = DocxParser(file_path, use_uno=True)
                result = parser.parse(file_path)

                elapsed = time.time() - start_time
                times.append(elapsed)

                content_length = len(result.get("content", ""))
                logger.info(f"✅ 成功 - 耗时: {elapsed:.2f}秒, 内容: {content_length} 字符")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ 失败 - 耗时: {elapsed:.2f}秒, 错误: {str(e)}")

        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"📊 UNO 方式平均耗时: {avg_time:.2f}秒")
            return avg_time

    except Exception as e:
        logger.error(f"❌ UNO 测试失败: {str(e)}")

    return 0


def main():
    """主测试函数"""
    logger.info("📋 文档转换性能对比测试")
    logger.info("=" * 60)

    # 测试文件列表
    test_files = [
        "examples/00b33cb2-3cce-40a1-95b7-de7d6935bf66.docx",
        "examples/EAM资产管理系统应急预案2020-02(新EAM).docx",
        "examples/中远海运科技_会议纪要_开尔唯OCP&BMS项目_20230523_BMS财务部应收会计调研.docx",
        "examples/远海码头官网应急预案2020-2.docx",
    ]

    # 过滤存在的文件
    available_files = [f for f in test_files if os.path.exists(f)]

    if not available_files:
        logger.error("❌ 没有找到测试文件")
        return

    logger.info(f"📁 找到 {len(available_files)} 个测试文件:")
    for f in available_files:
        size = os.path.getsize(f)
        logger.info(f"   {Path(f).name} ({size:,} 字节)")

    # 测试传统方式
    traditional_time = test_traditional_method(available_files)

    # 测试 UNO 方式
    uno_time = test_uno_method_if_available(available_files)

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 性能对比总结")
    logger.info("=" * 60)

    if traditional_time > 0:
        logger.info(f"传统方式平均耗时: {traditional_time:.2f}秒")

    if uno_time > 0:
        logger.info(f"UNO 方式平均耗时: {uno_time:.2f}秒")

        if traditional_time > 0:
            if uno_time < traditional_time:
                speedup = traditional_time / uno_time
                logger.info(f"🚀 UNO 方式快了 {speedup:.1f}x")
            else:
                slowdown = uno_time / traditional_time
                logger.info(f"⚠️  UNO 方式慢了 {slowdown:.1f}x")

    logger.info("\n💡 使用建议:")
    if traditional_time > 0 and traditional_time < 1.0:
        logger.info("   传统方式已经很快（<1秒），建议直接使用")
        logger.info("   parser = DocxParser(file_path, use_uno=False)")
    elif uno_time > 0 and uno_time < traditional_time:
        logger.info("   UNO 方式性能更好，建议用于高并发场景")
        logger.info("   parser = DocxParser(file_path, use_uno=True)")
    else:
        logger.info("   根据实际情况选择，两种方式都可用")


if __name__ == "__main__":
    main()
