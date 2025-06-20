#!/usr/bin/env python3
"""
LibreOffice 并行处理测试 - 验证是否存在假多线程问题

这个测试专门验证传统的 subprocess 调用 soffice 命令是否存在假多线程问题：
1. 测试多个soffice进程是否能真正并行运行
2. 检查进程锁和资源竞争情况
3. 对比不同并发数下的性能表现
4. 监控系统资源使用情况
"""

import concurrent.futures
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from loguru import logger

# 确保导入本地开发版本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def monitor_soffice_processes():
    """监控系统中的soffice进程"""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "soffice" in proc.info["name"].lower():
                processes.append(
                    {
                        "pid": proc.info["pid"],
                        "name": proc.info["name"],
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes


def kill_all_soffice():
    """终止所有soffice进程"""
    killed_count = 0
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if "soffice" in proc.info["name"].lower():
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed_count > 0:
        logger.info(f"🔪 已终止 {killed_count} 个 soffice 进程")
        time.sleep(2)  # 等待进程清理


def convert_with_monitoring(file_path: str, thread_id: int):
    """转换文件并监控进程"""
    logger.info(f"🚀 [线程{thread_id}] 开始转换: {os.path.basename(file_path)}")

    start_time = time.time()
    initial_processes = len(monitor_soffice_processes())

    try:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = f'soffice --headless --convert-to txt "{file_path}" --outdir "{temp_dir}"'

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            time.sleep(0.1)  # 等待进程启动
            mid_processes = len(monitor_soffice_processes())

            stdout, stderr = process.communicate()

            end_time = time.time()
            final_processes = len(monitor_soffice_processes())

            success = process.returncode == 0
            duration = end_time - start_time

            logger.info(
                f"{'✅' if success else '❌'} [线程{thread_id}] "
                f"{os.path.basename(file_path)} (耗时: {duration:.2f}s, "
                f"进程: {initial_processes}→{mid_processes}→{final_processes})"
            )

            return {
                "thread_id": thread_id,
                "file": os.path.basename(file_path),
                "success": success,
                "duration": duration,
                "proc_change": mid_processes - initial_processes,
            }

    except Exception as e:
        logger.error(f"💥 [线程{thread_id}] 异常: {str(e)}")
        return {"thread_id": thread_id, "success": False, "error": str(e)}


def test_parallel_performance():
    """测试并行性能"""
    test_files = [
        "examples/00b33cb2-3cce-40a1-95b7-de7d6935bf66.docx",
        "examples/EAM资产管理系统应急预案2020-02(新EAM).docx",
        "examples/中远海运科技_会议纪要_开尔唯OCP&BMS项目_20230523_BMS财务部应收会计调研.docx",
        "examples/远海码头官网应急预案2020-2.docx",
    ]

    existing_files = [f for f in test_files if os.path.exists(f)]
    if not existing_files:
        logger.error("❌ 没有找到测试文件")
        return

    logger.info("=" * 80)
    logger.info("🔬 LibreOffice 假多线程问题分析")
    logger.info("=" * 80)

    # 测试不同线程数
    thread_counts = [1, 2, 4, 8]
    results = {}

    for max_workers in thread_counts:
        logger.info(f"\n🧪 测试 {max_workers} 线程并行转换...")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(convert_with_monitoring, file_path, i + 1)
                for i, file_path in enumerate(existing_files)
            ]

            thread_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    thread_results.append(result)
                except Exception as e:
                    logger.error(f"线程异常: {e}")

        total_time = time.time() - start_time
        successful = [r for r in thread_results if r.get("success", False)]

        if successful:
            avg_duration = sum(r["duration"] for r in successful) / len(successful)
            efficiency = (avg_duration * len(existing_files)) / total_time

            results[max_workers] = {
                "total_time": total_time,
                "efficiency": efficiency,
                "successful_count": len(successful),
            }

            logger.info(
                f"📊 {max_workers}线程结果: 总时间={total_time:.2f}s, 效率={efficiency:.2f}x"
            )

    # 分析结果
    logger.info("\n" + "=" * 80)
    logger.info("📈 并行性能分析")
    logger.info("=" * 80)

    for workers, data in results.items():
        efficiency = data["efficiency"]
        ideal_efficiency = workers
        efficiency_ratio = efficiency / ideal_efficiency if ideal_efficiency > 0 else 0

        logger.info(
            f"{workers}线程: 效率={efficiency:.2f}x, 理想={ideal_efficiency}x, "
            f"达成率={efficiency_ratio*100:.1f}%"
        )

        if efficiency_ratio < 0.3:
            logger.warning(f"⚠️  {workers}线程存在严重性能问题（可能假多线程）")
        elif efficiency_ratio < 0.7:
            logger.warning(f"⚠️  {workers}线程存在性能损失")
        else:
            logger.info(f"✅ {workers}线程性能正常")


if __name__ == "__main__":
    test_parallel_performance()
