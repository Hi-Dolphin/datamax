"""
文档转换性能对比示例 - UNO API vs 传统方式

这个示例对比了两种文档转换方式的性能差异：

1. 传统方式（use_uno=False）：
   - 直接调用 soffice --headless --convert-to 命令
   - 每次调用都是独立进程
   - 无状态管理，开销相对较小

2. UNO API方式（use_uno=True）：
   - 使用LibreOffice的UNO API
   - 维护长连接和服务状态
   - 支持精细的文档控制，但有额外开销

性能差异原因：
- UNO方式慢：LibreOffice重量级、IPC通信、服务管理开销
- 传统方式快：直接命令调用、无中间层、独立进程

运行此示例会进行详细的性能测试和分析。
"""

import os
import sys

# 确保导入本地开发版本而不是已安装的版本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import concurrent.futures
import time
from pathlib import Path

# 配置日志
from loguru import logger

from datamax.parser.doc_parser import DocParser
from datamax.parser.docx_parser import DocxParser
from datamax.parser.ppt_parser import PPtParser
from datamax.utils import (
    HAS_UNO, 
    cleanup_uno_managers, 
    get_uno_manager, 
    pre_create_uno_managers, 
    warmup_uno_managers,
    release_uno_manager,
    uno_manager_context,
    get_uno_pool
)


def warmup_thread_pool(executor: concurrent.futures.ThreadPoolExecutor, num_tasks: int = None):
    """预热线程池，让所有线程真正启动起来"""
    if num_tasks is None:
        # 默认为线程池的最大工作线程数
        num_tasks = executor._max_workers
    
    def dummy_task(x):
        """简单的占位任务"""
        time.sleep(0.001)  # 短暂休眠
        return x * 2
    
    # 提交任务让线程启动
    futures = [executor.submit(dummy_task, i) for i in range(num_tasks)]
    
    # 等待所有任务完成
    for future in concurrent.futures.as_completed(futures):
        _ = future.result()
    
    logger.debug(f"   ⚡ 线程池预热完成，{num_tasks}个线程已就绪")


def convert_document(file_path: str, use_uno: bool = True):
    """转换单个文档"""
    file_path = Path(file_path)

    start_time = time.time()

    try:
        if file_path.suffix.lower() == ".doc":
            parser = DocParser(str(file_path), use_uno=use_uno)
            result = parser.parse(str(file_path))
        elif file_path.suffix.lower() == ".docx":
            parser = DocxParser(str(file_path), use_uno=use_uno)
            result = parser.parse(str(file_path))
        elif file_path.suffix.lower() == ".ppt":
            parser = PPtParser(str(file_path), use_uno=use_uno)
            result = parser.parse(str(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        elapsed_time = time.time() - start_time
        logger.info(f"✅ 转换成功: {file_path.name} (耗时: {elapsed_time:.2f}秒)")
        
        # 释放UNO管理器（如果使用）
        if use_uno:
            release_uno_manager()
            
        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ 转换失败: {file_path.name} - {str(e)} (耗时: {elapsed_time:.2f}秒)")
        # 确保释放管理器
        if use_uno:
            release_uno_manager()
        raise


def batch_convert_sequential(file_paths: list, use_uno: bool = False):
    """顺序转换多个文档（传统方式）"""
    logger.info(f"🔄 开始顺序转换 {len(file_paths)} 个文档...")
    start_time = time.time()

    results = []
    for file_path in file_paths:
        try:
            result = convert_document(file_path, use_uno=use_uno)
            results.append(result)
        except Exception as e:
            logger.error(f"转换失败: {file_path} - {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"⏱️ 顺序转换完成，总耗时: {total_time:.2f}秒")
    return results


def batch_convert_parallel(
    file_paths: list, max_workers: int = 4, use_uno: bool = True
):
    """并行转换多个文档（使用UNO API）"""
    if not HAS_UNO and use_uno:
        logger.warning("⚠️ UNO API 不可用，将使用传统方式")
        use_uno = False

    logger.info(f"🚀 开始并行转换 {len(file_paths)} 个文档 (工作线程: {max_workers})...")
    start_time = time.time()

    # 如果使用 UNO，预先连接服务
    if use_uno:
        manager = get_uno_manager()
        logger.info("📡 UNO 服务已连接")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(convert_document, file_path, use_uno): file_path
            for file_path in file_paths
        }

        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"转换失败: {file_path} - {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"⏱️ 并行转换完成，总耗时: {total_time:.2f}秒")
    return results


def batch_convert_parallel_with_executor(
    file_paths: list, executor: concurrent.futures.ThreadPoolExecutor, use_uno: bool = True
):
    """使用提供的线程池执行并行转换（避免重复创建线程池）"""
    if not HAS_UNO and use_uno:
        logger.warning("⚠️ UNO API 不可用，将使用传统方式")
        use_uno = False

    logger.info(f"🚀 使用预创建的线程池转换 {len(file_paths)} 个文档...")
    start_time = time.time()

    # 如果使用 UNO，预先连接服务
    if use_uno:
        manager = get_uno_manager()
        logger.info("📡 UNO 服务已连接")

    results = []
    # 提交所有任务
    future_to_file = {
        executor.submit(convert_document, file_path, use_uno): file_path
        for file_path in file_paths
    }

    # 收集结果
    for future in concurrent.futures.as_completed(future_to_file):
        file_path = future_to_file[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            logger.error(f"转换失败: {file_path} - {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"⏱️ 转换完成，耗时: {total_time:.2f}秒")
    return results


async def async_convert_document(file_path: str, use_uno: bool = True):
    """异步转换文档"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, convert_document, file_path, use_uno)


async def batch_convert_async(file_paths: list, use_uno: bool = True):
    """异步批量转换文档"""
    if not HAS_UNO and use_uno:
        logger.warning("⚠️ UNO API 不可用，将使用传统方式")
        use_uno = False

    logger.info(f"⚡ 开始异步转换 {len(file_paths)} 个文档...")
    start_time = time.time()

    # 如果使用 UNO，预先连接服务
    if use_uno:
        manager = get_uno_manager()
        logger.info("📡 UNO 服务已连接")

    # 创建所有任务
    tasks = [async_convert_document(file_path, use_uno) for file_path in file_paths]

    # 执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"转换失败: {file_paths[i]} - {str(result)}")
        else:
            successful_results.append(result)

    total_time = time.time() - start_time
    logger.info(f"⏱️ 异步转换完成，总耗时: {total_time:.2f}秒")
    return successful_results


def performance_comparison(file_paths: list):
    """详细性能对比测试"""
    logger.info("=" * 80)
    logger.info("📊 开始详细性能对比测试")
    logger.info("=" * 80)

    # 过滤存在的文件
    existing_files = [f for f in file_paths if os.path.exists(f)]
    if not existing_files:
        logger.error("❌ 没有找到可用的测试文件")
        return

    logger.info(f"📁 测试文件数量: {len(existing_files)}")
    for i, file_path in enumerate(existing_files, 1):
        file_size = os.path.getsize(file_path) / 1024  # KB
        logger.info(f"   {i}. {os.path.basename(file_path)} ({file_size:.1f} KB)")

    results = {}

    # 1. 测试传统方式（顺序，不使用UNO）
    logger.info("\n" + "=" * 60)
    logger.info("🐌 1️⃣ 传统方式 - 顺序处理（不使用UNO）")
    logger.info("=" * 60)
    start_time = time.time()
    try:
        batch_convert_sequential(existing_files, use_uno=False)
        sequential_time = time.time() - start_time
        results["sequential_no_uno"] = sequential_time
        logger.info(f"📊 传统顺序方式总耗时: {sequential_time:.2f}秒")
        logger.info(f"📊 平均每文件: {sequential_time/len(existing_files):.2f}秒")
    except Exception as e:
        logger.error(f"❌ 传统方式测试失败: {str(e)}")
        results["sequential_no_uno"] = None

    # 2. 测试传统方式（并行，不使用UNO）
    logger.info("\n" + "=" * 60)
    logger.info("⚡ 2️⃣ 传统方式 - 并行处理（不使用UNO，4线程）")
    logger.info("=" * 60)
    start_time = time.time()
    try:
        batch_convert_parallel(existing_files, max_workers=4, use_uno=False)
        parallel_no_uno_time = time.time() - start_time
        results["parallel_no_uno"] = parallel_no_uno_time
        logger.info(f"📊 传统并行方式总耗时: {parallel_no_uno_time:.2f}秒")
        logger.info(f"📊 平均每文件: {parallel_no_uno_time/len(existing_files):.2f}秒")
        if "sequential_no_uno" in results and results["sequential_no_uno"]:
            speedup = results["sequential_no_uno"] / parallel_no_uno_time
            logger.info(f"📊 相比顺序提升: {speedup:.2f}x")
    except Exception as e:
        logger.error(f"❌ 传统并行方式测试失败: {str(e)}")
        results["parallel_no_uno"] = None

    if HAS_UNO:
        # 3. 测试UNO方式（并行）
        logger.info("\n" + "=" * 60)
        logger.info("🚀 3️⃣ UNO API - 并行处理（4线程）")
        logger.info("=" * 60)
        start_time = time.time()
        try:
            batch_convert_parallel(existing_files, max_workers=4, use_uno=True)
            uno_parallel_time = time.time() - start_time
            results["uno_parallel"] = uno_parallel_time
            logger.info(f"📊 UNO并行方式总耗时: {uno_parallel_time:.2f}秒")
            logger.info(f"📊 平均每文件: {uno_parallel_time/len(existing_files):.2f}秒")
        except Exception as e:
            logger.error(f"❌ UNO并行方式测试失败: {str(e)}")
            results["uno_parallel"] = None

        # 4. 测试UNO方式（高并发）
        logger.info("\n" + "=" * 60)
        logger.info("🔥 4️⃣ UNO API - 高并发处理（8线程）")
        logger.info("=" * 60)
        start_time = time.time()
        try:
            batch_convert_parallel(existing_files, max_workers=8, use_uno=True)
            uno_high_parallel_time = time.time() - start_time
            results["uno_high_parallel"] = uno_high_parallel_time
            logger.info(f"📊 UNO高并发方式总耗时: {uno_high_parallel_time:.2f}秒")
            logger.info(f"📊 平均每文件: {uno_high_parallel_time/len(existing_files):.2f}秒")
        except Exception as e:
            logger.error(f"❌ UNO高并发方式测试失败: {str(e)}")
            results["uno_high_parallel"] = None
    else:
        logger.warning("⚠️ UNO API 不可用，跳过 UNO 性能测试")

    # 性能总结
    logger.info("\n" + "=" * 80)
    logger.info("📈 性能对比总结")
    logger.info("=" * 80)

    # 构建结果表格
    methods = [
        ("sequential_no_uno", "传统顺序方式"),
        ("parallel_no_uno", "传统并行方式(4线程)"),
        ("uno_parallel", "UNO并行方式(4线程)"),
        ("uno_high_parallel", "UNO高并发方式(8线程)"),
    ]

    logger.info(f"{'方法':<25} {'总时间(秒)':<12} {'平均时间(秒)':<15} {'相对性能':<12}")
    logger.info("-" * 70)

    baseline_time = None
    for key, name in methods:
        if key in results and results[key] is not None:
            total_time = results[key]
            avg_time = total_time / len(existing_files)

            if baseline_time is None:
                baseline_time = total_time
                relative = "1.00x (基准)"
            else:
                ratio = baseline_time / total_time
                relative = f"{ratio:.2f}x"

            logger.info(
                f"{name:<25} {total_time:<12.2f} {avg_time:<15.2f} {relative:<12}"
            )

    # 性能分析
    logger.info("\n" + "=" * 80)
    logger.info("🔍 性能分析")
    logger.info("=" * 80)

    if results.get("parallel_no_uno") and results.get("uno_parallel"):
        ratio = results["uno_parallel"] / results["parallel_no_uno"]
        if ratio > 1:
            logger.info(f"✅ 传统并行方式比UNO并行方式快 {ratio:.2f}x")
            logger.info("💡 建议在追求性能时使用传统方式（use_uno=False）")
        else:
            logger.info(f"✅ UNO并行方式比传统并行方式快 {1/ratio:.2f}x")
            logger.info("💡 UNO方式在这种情况下性能更优")

    logger.info("\n🧠 性能差异原因分析:")
    logger.info("  🐌 UNO慢的原因:")
    logger.info("     • LibreOffice是重量级应用，启动和运行开销大")
    logger.info("     • 需要进程间通信(IPC)，有网络延迟")
    logger.info("     • 需要完整加载文档到内存")
    logger.info("     • 服务连接和管理有额外开销")
    logger.info("  ⚡ 传统方式快的原因:")
    logger.info("     • 直接调用soffice命令，无中间层")
    logger.info("     • 每次调用都是独立进程，无状态管理")
    logger.info("     • 命令行调用开销相对较小")
    logger.info("  🤔 何时使用UNO:")
    logger.info("     • 需要精细控制文档转换过程")
    logger.info("     • 需要复杂的文档操作和格式化")
    logger.info("     • 在长时间运行的服务中，可以复用连接")

    logger.info("\n" + "=" * 80)
    logger.info("✅ 性能对比测试完成")
    logger.info("=" * 80)


def convert_document_with_manager_info(file_path: str, use_uno: bool = True):
    """转换单个文档并显示UNO管理器信息"""
    import threading

    thread_id = threading.current_thread().ident

    start_time = time.time()

    try:
        if file_path.lower().endswith(".doc"):
            parser = DocParser(str(file_path), use_uno=use_uno)
        elif file_path.lower().endswith(".docx"):
            parser = DocxParser(str(file_path), use_uno=use_uno)
        elif file_path.lower().endswith(".ppt"):
            parser = PPtParser(str(file_path), use_uno=use_uno)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        # 如果使用UNO，显示管理器信息
        if use_uno:
            manager = get_uno_manager()
            logger.info(f"🎯 [线程{thread_id}] 使用UnoManager (端口: {manager.port})")

        result = parser.parse(str(file_path))
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ [线程{thread_id}] 转换成功: {os.path.basename(file_path)} (耗时: {elapsed_time:.2f}秒)"
        )
        
        # 释放UNO管理器回到池中
        if use_uno:
            release_uno_manager()
            logger.debug(f"♻️ [线程{thread_id}] 已释放UnoManager")
        
        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            f"❌ [线程{thread_id}] 转换失败: {os.path.basename(file_path)} - {str(e)} (耗时: {elapsed_time:.2f}秒)"
        )
        # 确保释放管理器
        if use_uno:
            release_uno_manager()
        raise


def batch_convert_with_manager_info(
    file_paths: list, max_workers: int = 4, use_uno: bool = True
):
    """并行转换多个文档并显示管理器信息"""
    if not HAS_UNO and use_uno:
        logger.warning("⚠️ UNO API 不可用，将使用传统方式")
        use_uno = False

    logger.info(f"🚀 开始并行转换 {len(file_paths)} 个文档 (工作线程: {max_workers})...")
    start_time = time.time()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(
                convert_document_with_manager_info, file_path, use_uno
            ): file_path
            for file_path in file_paths
        }

        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"转换失败: {file_path} - {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"⏱️ 并行转换完成，总耗时: {total_time:.2f}秒")
    return results


def batch_convert_with_manager_info_with_executor(
    file_paths: list, executor: concurrent.futures.ThreadPoolExecutor, use_uno: bool = True
):
    """使用提供的线程池执行并行转换并显示管理器信息（避免重复创建线程池）"""
    if not HAS_UNO and use_uno:
        logger.warning("⚠️ UNO API 不可用，将使用传统方式")
        use_uno = False

    logger.info(f"🚀 使用预创建的线程池转换 {len(file_paths)} 个文档...")
    start_time = time.time()

    results = []
    # 提交所有任务
    future_to_file = {
        executor.submit(
            convert_document_with_manager_info, file_path, use_uno
        ): file_path
        for file_path in file_paths
    }

    # 收集结果
    for future in concurrent.futures.as_completed(future_to_file):
        file_path = future_to_file[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            logger.error(f"转换失败: {file_path} - {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"⏱️ 转换完成，耗时: {total_time:.2f}秒")
    return results


def traditional_stress_test(base_files: list, repeat_count: int = 3):
    """传统LibreOffice方式压力测试"""
    logger.info("=" * 100)
    logger.info("⚡ 传统LibreOffice方式并行性能压力测试")
    logger.info("=" * 100)

    # 扩充文件列表 - 重复调用相同文档
    expanded_files = []
    for i in range(repeat_count):
        for file_path in base_files:
            if os.path.exists(file_path):
                expanded_files.append(file_path)

    if not expanded_files:
        logger.error("❌ 没有找到可用的测试文件")
        return

    logger.info(f"📁 压测配置:")
    logger.info(f"   基础文件数: {len(base_files)}")
    logger.info(f"   重复次数: {repeat_count}")
    logger.info(f"   总文件数: {len(expanded_files)}")

    # 显示文件信息
    unique_files = list(set(expanded_files))
    total_size = 0
    for file_path in unique_files:
        file_size = os.path.getsize(file_path) / 1024  # KB
        total_size += file_size
        count = expanded_files.count(file_path)
        logger.info(f"   📄 {os.path.basename(file_path)}: {file_size:.1f}KB × {count}次")

    logger.info(f"   📊 总数据量: {total_size * repeat_count:.1f}KB")

    logger.info(f"🚀 传统LibreOffice并行架构:")
    logger.info(f"   🎯 每个线程使用独立的soffice进程")
    logger.info(f"   ⚡ 直接调用命令行，无中间层")
    logger.info(f"   🔄 无状态管理，进程独立运行")

    # 测试不同线程数的传统方式性能
    thread_configs = [1, 4, 8, 12]
    results = {}
    baseline_time = None
    
    # 提前创建所有线程池
    logger.info(f"\n🔧 预创建所有线程池...")
    executors = {}
    for workers in thread_configs:
        executors[workers] = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        logger.info(f"   ✅ 创建{workers}线程池完成")
        # 预热线程池
        warmup_thread_pool(executors[workers])
    
    logger.info(f"🎉 所有线程池准备就绪，开始测试...\n")

    try:
        for max_workers in thread_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"⚡ 测试传统方式 - {max_workers} 线程并行处理")
            logger.info(f"{'='*80}")

            start_time = time.time()

            try:
                # 使用预创建的线程池进行并行转换
                converted_results = batch_convert_parallel_with_executor(
                    expanded_files, executor=executors[max_workers], use_uno=False
                )

                total_time = time.time() - start_time
                successful_count = len([r for r in converted_results if r is not None])

                # 计算性能指标
                avg_time_per_file = total_time / len(expanded_files)
                throughput = len(expanded_files) / total_time  # 文件/秒

                # 理论最优时间（基于单线程时间）
                if max_workers == 1:
                    baseline_time = total_time
                    efficiency = 1.0
                else:
                    theoretical_time = baseline_time / max_workers
                    efficiency = theoretical_time / total_time if total_time > 0 else 0

                results[max_workers] = {
                    "total_time": total_time,
                    "successful_count": successful_count,
                    "avg_time_per_file": avg_time_per_file,
                    "throughput": throughput,
                    "efficiency": efficiency,
                    "files_processed": len(expanded_files),
                }

                logger.info(f"📊 性能统计:")
                logger.info(f"   总耗时: {total_time:.2f}秒")
                logger.info(f"   成功转换: {successful_count}/{len(expanded_files)}")
                logger.info(f"   平均时间: {avg_time_per_file:.2f}秒/文件")
                logger.info(f"   吞吐量: {throughput:.2f}文件/秒")
                if max_workers > 1:
                    logger.info(f"   并行效率: {efficiency:.2f}x (理想: {max_workers}x)")
                    efficiency_percentage = (efficiency / max_workers) * 100
                    logger.info(f"   效率百分比: {efficiency_percentage:.1f}%")

            except Exception as e:
                logger.error(f"❌ {max_workers}线程测试失败: {str(e)}")
                results[max_workers] = {
                    "error": str(e),
                    "total_time": 0,
                    "successful_count": 0,
                }
    finally:
        # 清理所有线程池
        logger.info(f"\n🧹 清理线程池...")
        for workers, executor in executors.items():
            executor.shutdown(wait=True)
            logger.info(f"   ✅ {workers}线程池已关闭")
        logger.info(f"🎉 所有线程池已清理")

    # 综合性能分析
    logger.info(f"\n{'='*100}")
    logger.info("📈 传统LibreOffice压力测试综合分析")
    logger.info(f"{'='*100}")

    # 性能对比表格
    logger.info(
        f"{'线程数':<8} {'总时间(秒)':<12} {'成功率':<10} {'吞吐量(文件/秒)':<18} {'并行效率':<12} {'效率百分比':<12}"
    )
    logger.info("-" * 85)

    baseline_throughput = None
    for max_workers in sorted(results.keys()):
        result = results[max_workers]
        if "error" in result:
            logger.info(
                f"{max_workers:<8} {'ERROR':<12} {'N/A':<10} {'N/A':<18} {'N/A':<12} {'N/A':<12}"
            )
            continue

        total_time = result["total_time"]
        success_rate = f"{result['successful_count']}/{result['files_processed']}"
        throughput = result["throughput"]
        efficiency = result.get("efficiency", 1.0)
        efficiency_percentage = (
            (efficiency / max_workers) * 100 if max_workers > 1 else 100
        )

        if baseline_throughput is None:
            baseline_throughput = throughput

        logger.info(
            f"{max_workers:<8} {total_time:<12.2f} {success_rate:<10} "
            f"{throughput:<18.2f} {efficiency:<12.2f} {efficiency_percentage:<12.1f}%"
        )

    # 性能分析和建议
    logger.info(f"\n🔍 传统方式性能分析:")

    # 找出最佳配置
    valid_results = {
        k: v
        for k, v in results.items()
        if "error" not in v and v["successful_count"] > 0
    }
    if valid_results:
        best_throughput = max(v["throughput"] for v in valid_results.values())
        best_config = [
            k for k, v in valid_results.items() if v["throughput"] == best_throughput
        ][0]

        logger.info(f"   🎯 最佳性能配置: {best_config}线程 (吞吐量: {best_throughput:.2f}文件/秒)")

        # 分析扩展性
        if len(valid_results) >= 3:
            efficiency_4 = valid_results.get(4, {}).get("efficiency", 0)
            efficiency_8 = valid_results.get(8, {}).get("efficiency", 0)
            efficiency_12 = valid_results.get(12, {}).get("efficiency", 0)

            logger.info(f"   📊 扩展性分析:")
            if efficiency_4 > 0:
                logger.info(
                    f"      4线程效率: {efficiency_4:.2f}x ({(efficiency_4/4)*100:.1f}%)"
                )
            if efficiency_8 > 0:
                logger.info(
                    f"      8线程效率: {efficiency_8:.2f}x ({(efficiency_8/8)*100:.1f}%)"
                )
            if efficiency_12 > 0:
                logger.info(
                    f"      12线程效率: {efficiency_12:.2f}x ({(efficiency_12/12)*100:.1f}%)"
                )

            # 分析LibreOffice假多线程问题
            if efficiency_4 < 2.0:
                logger.warning("   ⚠️  传统方式存在明显的并行瓶颈")
                if efficiency_4 < 1.5:
                    logger.warning("      🔴 严重瓶颈：可能是LibreOffice全局锁")
                else:
                    logger.warning("      🟡 中等瓶颈：可能是I/O或资源竞争")
            elif efficiency_8 < 4.0:
                logger.warning("   ⚠️  8线程以上存在性能递减")
                logger.info("      🟡 建议使用4线程以内获得最佳性价比")
            else:
                logger.info("   ✅ 传统方式并行性能良好")

            # 判断最佳线程数
            if efficiency_4 > efficiency_8 and efficiency_4 > efficiency_12:
                logger.info("   💡 4线程是最佳选择")
            elif efficiency_8 > efficiency_12:
                logger.info("   💡 8线程是合理上限")

    logger.info(f"\n💡 传统方式优化建议:")
    if valid_results:
        best_efficiency_ratio = max(
            (v.get("efficiency", 0) / k) for k, v in valid_results.items() if k > 1
        )

        if best_efficiency_ratio > 0.7:
            logger.info("   ✅ 传统方式并行效率良好")
        elif best_efficiency_ratio > 0.5:
            logger.info("   🟡 传统方式并行效率中等")
        else:
            logger.info("   🔴 传统方式并行效率较低")

        logger.info(f"   🎯 推荐配置: {best_config}线程 用于传统LibreOffice处理")
        logger.info(f"   📊 预期性能: {best_throughput:.2f}文件/秒")

        # 与理想并行的对比
        ideal_speedup = best_config
        actual_speedup = valid_results[best_config]["efficiency"]
        parallel_loss = (1 - actual_speedup / ideal_speedup) * 100
        logger.info(
            f"   📉 并行损失: {parallel_loss:.1f}% (理想{ideal_speedup}x vs 实际{actual_speedup:.2f}x)"
        )

    return results


def uno_stress_test(base_files: list, repeat_count: int = 3):
    """UNO API 压力测试"""
    logger.info("=" * 100)
    logger.info("🔥 UNO API 并行性能压力测试 - 多UNO服务版本")
    logger.info("=" * 100)

    # 清理之前的UNO管理器
    if cleanup_uno_managers:
        cleanup_uno_managers()
        logger.info("🧹 已清理之前的UNO管理器")
        time.sleep(2)  # 等待服务完全关闭

    # 扩充文件列表 - 重复调用相同文档
    expanded_files = []
    for i in range(repeat_count):
        for file_path in base_files:
            if os.path.exists(file_path):
                expanded_files.append(file_path)

    if not expanded_files:
        logger.error("❌ 没有找到可用的测试文件")
        return

    logger.info(f"📁 压测配置:")
    logger.info(f"   基础文件数: {len(base_files)}")
    logger.info(f"   重复次数: {repeat_count}")
    logger.info(f"   总文件数: {len(expanded_files)}")

    # 显示文件信息
    unique_files = list(set(expanded_files))
    total_size = 0
    for file_path in unique_files:
        file_size = os.path.getsize(file_path) / 1024  # KB
        total_size += file_size
        count = expanded_files.count(file_path)
        logger.info(f"   📄 {os.path.basename(file_path)}: {file_size:.1f}KB × {count}次")

    logger.info(f"   📊 总数据量: {total_size * repeat_count:.1f}KB")

    if not HAS_UNO:
        logger.error("❌ UNO API 不可用，无法进行压测")
        return

    logger.info(f"🏊 多UNO服务并行架构:")
    logger.info(f"   🎯 每个线程将使用独立的UNO服务实例")
    logger.info(f"   🔌 UNO服务端口范围: 2002-2009")
    logger.info(f"   🚀 支持真正的并行处理")
    
    # 显示连接池配置信息
    pool = get_uno_pool()
    logger.info(f"   📊 连接池最大管理器数: {pool.max_managers}")
    logger.info(f"   ♻️  支持管理器复用，提高性能")

    # 测试不同线程数的UNO性能
    thread_configs = [1, 4, 8, 12]
    results = {}
    baseline_time = None
    
    # 预创建所有UNO管理器
    max_uno_managers = max(thread_configs)
    logger.info(f"\n🔧 预创建 {max_uno_managers} 个UNO管理器...")
    start_time = time.time()
    created_count = pre_create_uno_managers(max_uno_managers)
    creation_time = time.time() - start_time
    logger.info(f"✅ 成功创建 {created_count} 个UNO管理器，耗时 {creation_time:.2f}秒")
    
    # 预热UNO管理器
    logger.info(f"\n⚡ 预热所有UNO管理器...")
    start_time = time.time()
    warmup_uno_managers()
    warmup_time = time.time() - start_time
    logger.info(f"✅ 预热完成，耗时 {warmup_time:.2f}秒")
    
    # 提前创建所有线程池
    logger.info(f"\n🔧 预创建所有线程池...")
    executors = {}
    for workers in thread_configs:
        executors[workers] = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        logger.info(f"   ✅ 创建{workers}线程池完成")
        # 预热线程池
        warmup_thread_pool(executors[workers])
    
    logger.info(f"🎉 所有资源准备就绪，开始测试...\n")

    try:
        for max_workers in thread_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"🚀 测试 UNO API - {max_workers} 线程并行处理")
            logger.info(f"{'='*80}")

            start_time = time.time()

            try:
                # 使用预创建的线程池进行UNO并行转换，显示管理器信息
                converted_results = batch_convert_with_manager_info_with_executor(
                    expanded_files, executor=executors[max_workers], use_uno=True
                )

                total_time = time.time() - start_time
                successful_count = len([r for r in converted_results if r is not None])

                # 计算性能指标
                avg_time_per_file = total_time / len(expanded_files)
                throughput = len(expanded_files) / total_time  # 文件/秒

                # 理论最优时间（基于单线程时间）
                if max_workers == 1:
                    baseline_time = total_time
                    efficiency = 1.0
                else:
                    theoretical_time = baseline_time / max_workers
                    efficiency = theoretical_time / total_time if total_time > 0 else 0

                results[max_workers] = {
                    "total_time": total_time,
                    "successful_count": successful_count,
                    "avg_time_per_file": avg_time_per_file,
                    "throughput": throughput,
                    "efficiency": efficiency,
                    "files_processed": len(expanded_files),
                }

                logger.info(f"📊 性能统计:")
                logger.info(f"   总耗时: {total_time:.2f}秒")
                logger.info(f"   成功转换: {successful_count}/{len(expanded_files)}")
                logger.info(f"   平均时间: {avg_time_per_file:.2f}秒/文件")
                logger.info(f"   吞吐量: {throughput:.2f}文件/秒")
                if max_workers > 1:
                    logger.info(f"   并行效率: {efficiency:.2f}x (理想: {max_workers}x)")
                    efficiency_percentage = (efficiency / max_workers) * 100
                    logger.info(f"   效率百分比: {efficiency_percentage:.1f}%")

            except Exception as e:
                logger.error(f"❌ {max_workers}线程测试失败: {str(e)}")
                results[max_workers] = {
                    "error": str(e),
                    "total_time": 0,
                    "successful_count": 0,
                }
    finally:
        # 清理所有线程池
        logger.info(f"\n🧹 清理线程池...")
        for workers, executor in executors.items():
            executor.shutdown(wait=True)
            logger.info(f"   ✅ {workers}线程池已关闭")
        logger.info(f"🎉 所有线程池已清理")

    # 综合性能分析
    logger.info(f"\n{'='*100}")
    logger.info("📈 UNO API 压力测试综合分析")
    logger.info(f"{'='*100}")

    # 性能对比表格
    logger.info(
        f"{'线程数':<8} {'总时间(秒)':<12} {'成功率':<10} {'吞吐量(文件/秒)':<18} {'并行效率':<12} {'效率百分比':<12}"
    )
    logger.info("-" * 85)

    baseline_throughput = None
    for max_workers in sorted(results.keys()):
        result = results[max_workers]
        if "error" in result:
            logger.info(
                f"{max_workers:<8} {'ERROR':<12} {'N/A':<10} {'N/A':<18} {'N/A':<12} {'N/A':<12}"
            )
            continue

        total_time = result["total_time"]
        success_rate = f"{result['successful_count']}/{result['files_processed']}"
        throughput = result["throughput"]
        efficiency = result.get("efficiency", 1.0)
        efficiency_percentage = (
            (efficiency / max_workers) * 100 if max_workers > 1 else 100
        )

        if baseline_throughput is None:
            baseline_throughput = throughput

        logger.info(
            f"{max_workers:<8} {total_time:<12.2f} {success_rate:<10} "
            f"{throughput:<18.2f} {efficiency:<12.2f} {efficiency_percentage:<12.1f}%"
        )

    # 性能分析和建议
    logger.info(f"\n🔍 性能分析:")

    # 找出最佳配置
    valid_results = {
        k: v
        for k, v in results.items()
        if "error" not in v and v["successful_count"] > 0
    }
    if valid_results:
        best_throughput = max(v["throughput"] for v in valid_results.values())
        best_config = [
            k for k, v in valid_results.items() if v["throughput"] == best_throughput
        ][0]

        logger.info(f"   🎯 最佳性能配置: {best_config}线程 (吞吐量: {best_throughput:.2f}文件/秒)")

        # 分析扩展性
        if len(valid_results) >= 3:
            efficiency_4 = valid_results.get(4, {}).get("efficiency", 0)
            efficiency_8 = valid_results.get(8, {}).get("efficiency", 0)
            efficiency_12 = valid_results.get(12, {}).get("efficiency", 0)

            logger.info(f"   📊 扩展性分析:")
            if efficiency_4 > 0:
                logger.info(
                    f"      4线程效率: {efficiency_4:.2f}x ({(efficiency_4/4)*100:.1f}%)"
                )
            if efficiency_8 > 0:
                logger.info(
                    f"      8线程效率: {efficiency_8:.2f}x ({(efficiency_8/8)*100:.1f}%)"
                )
            if efficiency_12 > 0:
                logger.info(
                    f"      12线程效率: {efficiency_12:.2f}x ({(efficiency_12/12)*100:.1f}%)"
                )

            # 判断UNO的并行瓶颈
            if efficiency_8 < 4.0:
                logger.warning("   ⚠️  UNO API存在明显的并行瓶颈")
                if efficiency_4 < 2.0:
                    logger.warning("      🔴 严重瓶颈：可能是全局锁或资源竞争")
                else:
                    logger.warning("      🟡 中等瓶颈：建议使用4线程以内")
            else:
                logger.info("   ✅ UNO API并行性能良好")

    # 与传统方式对比提示
    logger.info(f"\n💡 优化建议:")
    if valid_results:
        best_efficiency_ratio = max(
            (v.get("efficiency", 0) / k) for k, v in valid_results.items() if k > 1
        )

        if best_efficiency_ratio < 0.3:
            logger.info("   📝 UNO API并行效率较低，考虑使用传统方式")
        elif best_efficiency_ratio < 0.6:
            logger.info("   📝 UNO API有一定并行能力，但不如传统方式")
        else:
            logger.info("   📝 UNO API并行性能可接受")

        logger.info(f"   🎯 推荐配置: {best_config}线程 用于UNO API并行处理")
        logger.info(f"   📊 预期性能: {best_throughput:.2f}文件/秒")

    # 清理UNO管理器
    if cleanup_uno_managers:
        cleanup_uno_managers()
        logger.info("🧹 压测完成，已清理所有UNO管理器")

    return results


def comprehensive_stress_test(base_files: list, repeat_count: int = 3):
    """综合压力测试 - 传统方式 vs UNO方式"""
    logger.info("=" * 120)
    logger.info("🏆 LibreOffice 综合性能压力测试 - 传统方式 vs UNO方式")
    logger.info("=" * 120)

    # 1. 传统方式压力测试
    logger.info("\n🚀 第一阶段：传统LibreOffice方式压力测试")
    traditional_results = traditional_stress_test(base_files, repeat_count)

    # 等待一下，避免资源冲突
    time.sleep(3)

    # 2. UNO方式压力测试（如果可用）
    if HAS_UNO:
        logger.info("\n🚀 第二阶段：UNO API方式压力测试")
        uno_results = uno_stress_test(base_files, repeat_count)
    else:
        logger.warning("\n⚠️ UNO API不可用，跳过UNO压力测试")
        uno_results = {}

    # 3. 综合对比分析
    logger.info(f"\n{'='*120}")
    logger.info("🏆 综合性能对比分析")
    logger.info(f"{'='*120}")

    if traditional_results and uno_results:
        # 找出各自最佳配置
        traditional_best = max(
            ((k, v) for k, v in traditional_results.items() if "error" not in v),
            key=lambda x: x[1]["throughput"],
        )
        uno_best = max(
            ((k, v) for k, v in uno_results.items() if "error" not in v),
            key=lambda x: x[1]["throughput"],
        )

        traditional_threads, traditional_perf = traditional_best
        uno_threads, uno_perf = uno_best

        logger.info(f"📊 最佳性能对比:")
        logger.info(
            f"   传统方式: {traditional_threads}线程, {traditional_perf['throughput']:.2f}文件/秒"
        )
        logger.info(f"   UNO方式:  {uno_threads}线程, {uno_perf['throughput']:.2f}文件/秒")

        # 性能比较
        speed_ratio = traditional_perf["throughput"] / uno_perf["throughput"]
        if speed_ratio > 1.2:
            logger.info(f"   🏆 传统方式胜出: 快 {speed_ratio:.2f}x")
        elif speed_ratio < 0.8:
            logger.info(f"   🏆 UNO方式胜出: 快 {1/speed_ratio:.2f}x")
        else:
            logger.info(f"   🤝 两种方式性能接近 (比率: {speed_ratio:.2f})")

        # 并行效率对比
        logger.info(f"\n📈 并行效率对比:")
        for threads in [1, 4, 8, 12]:
            if threads in traditional_results and threads in uno_results:
                trad_eff = traditional_results[threads].get("efficiency", 0)
                uno_eff = uno_results[threads].get("efficiency", 0)
                trad_pct = (trad_eff / threads) * 100 if threads > 1 else 100
                uno_pct = (uno_eff / threads) * 100 if threads > 1 else 100

                logger.info(f"   {threads}线程: 传统{trad_pct:.1f}% vs UNO{uno_pct:.1f}%")

    logger.info(f"\n💡 最终建议:")
    if traditional_results:
        best_traditional = max(
            v["throughput"] for v in traditional_results.values() if "error" not in v
        )
        if HAS_UNO and uno_results:
            best_uno = max(
                v["throughput"] for v in uno_results.values() if "error" not in v
            )
            if best_traditional > best_uno * 1.2:
                logger.info("   🎯 推荐使用传统LibreOffice方式 (use_uno=False)")
                logger.info("   ✅ 传统方式在性能上有明显优势")
            elif best_uno > best_traditional * 1.2:
                logger.info("   🎯 推荐使用UNO API方式 (use_uno=True)")
                logger.info("   ✅ UNO方式在性能上有明显优势")
            else:
                logger.info("   🤝 两种方式性能相近，可根据功能需求选择")
                logger.info("   📝 简单转换用传统方式，复杂操作用UNO方式")
        else:
            logger.info("   🎯 推荐使用传统LibreOffice方式 (use_uno=False)")
            logger.info("   📝 UNO不可用或性能未测试")

    return {"traditional": traditional_results, "uno": uno_results}


if __name__ == "__main__":
    # 基础测试文件列表
    base_test_files = [
        "examples/datamax.doc",
        "examples/datamax.docx"
    ]

    # 检查 UNO 可用性
    if HAS_UNO:
        logger.info("✅ UNO API 可用")
    else:
        logger.info("❌ UNO API 不可用，将使用传统方式")

    # 提供测试选择
    import sys

    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()

        if test_mode == "traditional":
            logger.info("\n⚡ 开始传统LibreOffice方式压力测试...")
            traditional_stress_test(base_test_files, repeat_count=10)

        elif test_mode == "uno":
            if HAS_UNO:
                logger.info("\n🔥 开始UNO API并行性能压力测试...")
                uno_stress_test(base_test_files, repeat_count=10)
            else:
                logger.error("❌ UNO API不可用，无法进行UNO压力测试")

        elif test_mode == "comprehensive":
            logger.info("\n🏆 开始综合性能对比测试...")
            comprehensive_stress_test(base_test_files, repeat_count=10)
            
        elif test_mode == "context":
            # 演示使用上下文管理器
            if HAS_UNO:
                logger.info("\n🎯 演示使用 uno_manager_context 上下文管理器...")
                
                for file_path in base_test_files:
                    if os.path.exists(file_path):
                        # 使用上下文管理器自动管理UNO资源
                        with uno_manager_context() as manager:
                            logger.info(f"📁 使用管理器 (端口: {manager.port}) 转换文件: {file_path}")
                            
                            output_path = f"{file_path}.converted.txt"
                            manager.convert_document(file_path, output_path, "txt")
                            
                            logger.info(f"✅ 转换完成: {output_path}")
                            # 管理器会自动释放回池中
                            
                logger.info("🎉 所有文件转换完成，管理器已自动释放")
            else:
                logger.error("❌ UNO API不可用")

        else:
            logger.error(f"❌ 未知的测试模式: {test_mode}")
            logger.info("可用模式: traditional, uno, comprehensive, context")
    else:
        # 默认进行传统方式压力测试
        logger.info("\n💡 使用参数指定测试模式:")
        logger.info(
            "   python examples/uno_conversion_example.py traditional    # 仅测试传统方式"
        )
        logger.info(
            "   python examples/uno_conversion_example.py uno            # 仅测试UNO方式"
        )
        logger.info(
            "   python examples/uno_conversion_example.py comprehensive  # 综合对比测试"
        )
        logger.info(
            "   python examples/uno_conversion_example.py context        # 演示上下文管理器"
        )
        logger.info("")
        logger.info("⚡ 默认运行传统LibreOffice方式压力测试...")
        traditional_stress_test(base_test_files, repeat_count=10)
