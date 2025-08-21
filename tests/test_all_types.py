#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataMax服务全格式文件测试脚本

该脚本用于测试DataMax服务对各种文件格式的处理能力，包括：
- 同步接口测试
- 异步接口测试
- 多种文件格式支持
- 测试结果统计和报告

支持的文件格式：
- 文档类：doc, docx, pdf, ppt, pptx
- 图片类：png, jpg, jpeg, webp
- 表格类：xls, xlsx, csv
- 文本类：txt, md, html, json, py, log, epub
"""

import os
import json
import requests
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from download_file_from_url import download_file
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 测试文件URL配置
doc_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/doc/CI-D01-020%20%E8%88%B9%E9%95%BF%E5%91%BD%E4%BB%A4%E7%AE%A1%E7%90%86%E8%A7%84%E5%AE%9A%281.0%29.doc?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634385&Signature=VG8NcVBgsUGA4cZgVcsA2C%2BfBpA%3D"
docx_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/docx/%E8%BF%9C%E6%B5%B7%E7%A0%81%E5%A4%B4%E5%AE%98%E7%BD%91%E5%BA%94%E6%80%A5%E9%A2%84%E6%A1%882020-2.docx?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634414&Signature=C5/S/RaFKHaQmIJDC/dZLbGTxBU%3D"
pdf_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/pdf/%E7%94%B3%E6%8A%A5%E8%87%AA%E6%88%91%E6%89%BF%E8%AF%BA%E4%B9%A6.pdf?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634494&Signature=E2h6RXV8A6DljpOz3n99V1oQYAQ%3D"
ppt_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/pdf/%E5%8E%A6%E9%97%A8%E8%BF%9C%E6%B5%B7EDI%E7%B3%BB%E7%BB%9F%E4%BB%8B%E7%BB%8D.ppt?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634554&Signature=PUzaUY80ye4hYopJdg0Qse4hNqg%3D"
pptx_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/02%E6%B3%A2%E6%B5%AA%E7%90%86%E8%AE%BA.pptx?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634747&Signature=03ssV9IHl0TWF/ZgFyYMEDZn5Mk%3D"
png_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/png/image1.png?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634627&Signature=D2CWJrwzfNCqBIFNCp2PISUlO%2BY%3D"
jpg_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_1750227418847.jpg?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634834&Signature=di9h4CPb53zCqEjFEtZxqw8iqAc%3D"
jpeg_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/mmexportc1cb5a081cf7d159678ef87c9300dfd0_1692165857046%281%29.jpeg?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634856&Signature=CVv/le/bTPvrzV1tNWwMR2NQecQ%3D"
webp_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/DALL%C2%B7E%202024-09-14%2012.06.49%20-%20An%20advanced%20shipping%20agent%20from%20the%20Hi-Dolphin%20model%20with%20autonomous%20thinking%20abilities.%20The%20agent%20is%20shown%20in%20a%20futuristic%20maritime%20environment%2C%20navi.webp?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634958&Signature=tmS26f7NELl1PUSlY9r4RyhDrWE%3D"
xls_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/ocr-test/pdf/%E8%88%B9%E8%88%B6%E6%8A%80%E6%9C%AF%E8%A7%84%E8%8C%8320250427.xls?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634648&Signature=04I9KIIX0kbGLvIuIU4gxO0i2cc%3D"
xlsx_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/20230615%EF%BC%88%E7%AC%AC%E5%8D%81%E4%B8%80%E5%91%A8%EF%BC%89-%E4%B8%AD%E8%BF%9C%E6%B5%B7%E8%BF%90%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD%E6%B5%8B%E8%AF%95%E5%86%85%E5%AE%B9%E5%8B%BE%E9%80%89%E8%A1%A8--%E5%90%84%E5%8E%82%E5%95%86%E8%BF%9B%E5%BA%A6%E5%8F%8D%E9%A6%88.xlsx?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634774&Signature=nCqY8gH/9F6zGBPRermyz73AYtM%3D"
csv_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/%E5%AD%A6%E4%B9%A0%E9%A2%98%E5%BA%93.csv?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634903&Signature=Ic4hmKBLbO90f2rnUtZrbnzf79Q%3D"
html_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/1f8v.html?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634922&Signature=9nkBlA3n/c2Fm2Wg3liqAdANqj4%3D"
epub_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/%E4%B8%9C%E9%87%8E%E5%9C%AD%E5%90%BE-%E3%80%8A%E5%AB%8C%E7%96%91%E4%BA%BAX%E7%9A%84%E7%8C%AE%E8%BA%AB%E3%80%8B.epub?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634884&Signature=6hBPq6fxoLnxQ0NGAHZWWITmilU%3D"
md_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/2015-01-30.md?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634934&Signature=spR9fVgQQMa4i/3OnIf/DpXiBeA%3D"
py_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/idp_pdf.py?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786634984&Signature=BRmVUzL%2BCIOHLUEZ9cmW%2B/ZRUhU%3D"
log_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/err.log?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786635000&Signature=ZL22bu67UteuyqGZ9lL3B7vfyog%3D"
txt_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/LICENSE.txt?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786635015&Signature=bbsqcgBEzc/lRC30rv1eh%2Bw0Dbo%3D"
json_example_file_url = "https://hi-dolphin-prod.obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com:443/debug/step-2f36bb90-c078-400f-9c77-09dd467aaa4c.json?AccessKeyId=AQGMV9H38SJGSUCN4QSZ&Expires=1786635031&Signature=x18G%2BcQwXKP76Hh%2B/4xgaITmxSA%3D"


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FileType(Enum):
    """文件类型枚举"""
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    PPT = "ppt"
    PPTX = "pptx"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEBP = "webp"
    XLS = "xls"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    PY = "py"
    LOG = "log"
    EPUB = "epub"


@dataclass
class TestResult:
    """测试结果数据类"""
    file_type: str
    test_type: str  # 'sync' or 'async'
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    response_data: Optional[Dict] = None
    error_message: Optional[str] = None
    task_id: Optional[str] = None
    file_size_bytes: Optional[int] = None
    processing_time_ms: Optional[int] = None

    @property
    def is_success(self) -> bool:
        return self.status == TestStatus.SUCCESS

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.duration_ms is not None:
            return self.duration_ms / 1000.0
        return None


@dataclass
class TestSummary:
    """测试汇总数据类"""
    total_tests: int = 0
    success_tests: int = 0
    failed_tests: int = 0
    timeout_tests: int = 0
    total_duration_ms: int = 0
    results: List[TestResult] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.success_tests / self.total_tests) * 100

    @property
    def average_duration_ms(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.total_duration_ms / self.total_tests


# 测试文件配置映射
TEST_FILES = {
    FileType.DOC.value: doc_example_file_url,
    FileType.DOCX.value: docx_example_file_url,
    FileType.PDF.value: pdf_example_file_url,
    FileType.PPT.value: ppt_example_file_url,
    FileType.PPTX.value: pptx_example_file_url,
    FileType.PNG.value: png_example_file_url,
    FileType.JPG.value: jpg_example_file_url,
    FileType.JPEG.value: jpeg_example_file_url,
    FileType.WEBP.value: webp_example_file_url,
    FileType.XLS.value: xls_example_file_url,
    FileType.XLSX.value: xlsx_example_file_url,
    FileType.CSV.value: csv_example_file_url,
    FileType.TXT.value: txt_example_file_url,
    FileType.MD.value: md_example_file_url,
    FileType.HTML.value: html_example_file_url,
    FileType.JSON.value: json_example_file_url,
    FileType.PY.value: py_example_file_url,
    # FileType.LOG.value: log_example_file_url,
    FileType.EPUB.value: epub_example_file_url,
}


class DataMaxTester:
    """DataMax服务测试器"""
    def test_datamax_sdk(self, file_types: Optional[List[str]] = None) -> TestSummary:
        """
        测试DataMax SDK
        """
        if file_types is None:
            file_types = list(TEST_FILES.keys())

        summary = TestSummary()
        
        logger.info(f"开始SDK批量测试，共 {len(file_types)} 个文件类型")
        summary.total_tests = len(file_types)
        
        for file_type in file_types:
            if file_type not in TEST_FILES:
                logger.warning(f"跳过未知文件类型: {file_type}")
                continue
                
            download_url = TEST_FILES[file_type]
            res = download_file(download_url)
            if res.get("success"):
                try:
                    file_path = res.get("file_path")
                    from datamax import DataMax
                    dm = DataMax(
                        file_path=file_path,
                        use_mineru = True
                    )
                    result = dm.get_data()

                    if result:
                        result['file_type'] = file_type
                        summary.results.append(result)
                        summary.success_tests += 1
                    time.sleep(1)
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"处理文件 {file_type} 失败: {e}")
                    summary.results.append({
                        "file_type": file_type,
                        "test_type": "sync",
                        "status": TestStatus.FAILED,
                        "error_message": str(e),
                    })
                    summary.failed_tests += 1
                    time.sleep(1)

            else:
                logger.error(f"下载文件 {file_type} 失败: {res.get('error_message')}")
                summary.results.append({
                    "file_type": file_type,
                    "test_type": "sync",
                    "status": TestStatus.FAILED,
                    "error_message": res.get('error_message'),
                })
                summary.failed_tests += 1
                time.sleep(1)


            
        logger.info(f"SDK测试完成 - 成功率: {summary.success_rate:.1f}%")
        return summary

    
    def generate_report(self, sync_summary: TestSummary) -> str:
        """
        生成测试报告
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("DataMax服务全格式文件测试报告")
        report_lines.append("="*80)
        report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 总体统计
        total_tests = sync_summary.total_tests
        total_success = sync_summary.success_tests
        total_failed = sync_summary.failed_tests
        total_timeout = sync_summary.timeout_tests
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        report_lines.append("📊 总体统计")
        report_lines.append("-" * 40)
        report_lines.append(f"总测试数量: {total_tests}")
        report_lines.append(f"成功: {total_success} ({overall_success_rate:.1f}%)")
        report_lines.append(f"失败: {total_failed}")
        report_lines.append(f"超时: {total_timeout}")
        report_lines.append("")
        report_lines.append("="*80)

        if sync_summary.results:
            report_lines.append("详细结果:")
            for result in sync_summary.results:
                
                status_icon = "✅" if result.get("content") else "❌"
                report_lines.append(f"  {status_icon} {result.get('file_type', '')}")

                if result.get('error_message'):
                    report_lines.append(f"      错误: {result.get('error_message')}")
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report_to_file(self, report: str, filename: str = None) -> str:
        """
        保存测试报告到文件
        
        Args:
            report: 测试报告内容
            filename: 文件名，如果为None则自动生成
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"datamax_test_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"测试报告已保存到: {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存测试报告失败: {str(e)}")
            return ""


def run_datamax_sdk_test(file_types: Optional[List[str]] = None):
    """
    运行DataMax SDK测试
    """
    tester = DataMaxTester()
    sync_summary = tester.test_datamax_sdk(file_types)
    report = tester.generate_report(sync_summary)
    print("\n" + report)
    # 保存报告文件
    report_file = tester.save_report_to_file(report)
    logger.info(f"\n✅ 测试完成！")
    if report_file:
        logger.info(f"📄 报告文件: {report_file}")

    
def api_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DataMax服务全格式文件测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                示例用法:
                python test_all_format.py                          # 测试所有格式，同步+异步
                python test_all_format.py --types pdf,docx,png     # 仅测试指定格式
                python test_all_format.py --test_sdk               # 测试SDK
                ......
        """
    )
    
    parser.add_argument(
        '--types',
        help='要测试的文件类型，用逗号分隔 (例如: pdf,docx,png)'
    )
    
    parser.add_argument(
        '--test_sdk',
        action='store_true',
        help='仅测试SDK'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细日志输出'
    )
    
    parser.add_argument(
        '--list-types',
        action='store_true',
        help='列出所有支持的文件类型'
    )
    
    args = parser.parse_args()
    # 列出支持的文件类型
    if args.list_types:
        print("支持的文件类型:")
        for file_type in sorted(TEST_FILES.keys()):
            print(f"  - {file_type}")
        return

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 解析文件类型
    file_types = None
    if args.types:
        file_types = [t.strip().lower() for t in args.types.split(',')]
        # 验证文件类型
        invalid_types = [t for t in file_types if t not in TEST_FILES]
        if invalid_types:
            logger.error(f"不支持的文件类型: {invalid_types}")
            logger.info(f"支持的文件类型: {list(TEST_FILES.keys())}")
            return
            
    try:
        run_datamax_sdk_test(
            file_types=file_types
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  测试被用户中断")
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    # api_main()
    from datamax import DataMax
    res = download_file(pdf_example_file_url)
    print(res.get("file_path"))
    dm = DataMax(
        file_path=res.get("file_path"),
        use_mineru=True
    )

    print(dm.get_data())

