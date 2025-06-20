#!/usr/bin/env python3
"""
DOCX文件格式分析和演示脚本

这个脚本详细分析DOCX文件可能遇到的各种格式和存储方式，
帮助理解为什么需要多种内容提取策略。
"""

import sys
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_docx_structure(docx_path: str):
    """分析DOCX文件的内部结构"""
    
    print("=" * 80)
    print(f"📄 DOCX文件分析: {docx_path}")
    print("=" * 80)
    
    if not os.path.exists(docx_path):
        print(f"❌ 文件不存在: {docx_path}")
        return
    
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx:
            print("\n📁 文件结构:")
            file_list = docx.namelist()
            for filename in sorted(file_list):
                size = docx.getinfo(filename).file_size
                print(f"  📄 {filename} ({size} bytes)")
            
            # 分析不同的内容存储方式
            analyze_content_types(docx)
            
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")

def analyze_content_types(docx_zip: zipfile.ZipFile):
    """分析DOCX中的各种内容类型"""
    
    print("\n🔍 内容类型分析:")
    
    # 1. 检查altChunk (嵌入HTML/MHT)
    if check_altchunk(docx_zip):
        print("  ✅ 检测到 altChunk 格式 (嵌入HTML/MHT内容)")
    
    # 2. 检查标准document.xml
    if check_standard_document(docx_zip):
        print("  ✅ 检测到 标准document.xml格式")
    
    # 3. 检查页眉页脚
    headers_footers = check_headers_footers(docx_zip)
    if headers_footers:
        print(f"  ✅ 检测到 页眉页脚: {len(headers_footers)} 个文件")
    
    # 4. 检查注释
    if check_comments(docx_zip):
        print("  ✅ 检测到 注释和批注")
    
    # 5. 检查文本框
    textboxes = check_textboxes(docx_zip)
    if textboxes:
        print(f"  ✅ 检测到 文本框: {textboxes} 个")
    
    # 6. 检查嵌入对象
    embedded = check_embedded_objects(docx_zip)
    if embedded:
        print(f"  ✅ 检测到 嵌入对象: {len(embedded)} 个")
    
    # 7. 检查图表和图形
    charts_shapes = check_charts_shapes(docx_zip)
    if charts_shapes:
        print(f"  ✅ 检测到 图表和图形: {len(charts_shapes)} 个")

def check_altchunk(docx_zip: zipfile.ZipFile) -> bool:
    """检查是否包含altChunk"""
    try:
        if 'word/document.xml' in docx_zip.namelist():
            doc_xml = docx_zip.read('word/document.xml').decode('utf-8', errors='replace')
            if 'altChunk' in doc_xml:
                # 查找相关的MHT或HTML文件
                mht_files = [f for f in docx_zip.namelist() if f.endswith('.mht') and 'word/' in f]
                html_files = [f for f in docx_zip.namelist() if f.endswith('.html') and 'word/' in f]
                
                print(f"    📎 MHT文件: {mht_files}")
                print(f"    📎 HTML文件: {html_files}")
                
                # 分析MHT文件内容
                for mht_file in mht_files:
                    mht_content = docx_zip.read(mht_file).decode('utf-8', errors='replace')
                    print(f"    📋 {mht_file} 内容预览:")
                    print(f"         大小: {len(mht_content)} 字符")
                    if 'Content-Type: text/html' in mht_content:
                        print("         包含HTML内容")
                    if 'quoted-printable' in mht_content:
                        print("         使用quoted-printable编码")
                
                return True
        return False
    except Exception as e:
        print(f"    ❌ 检查altChunk失败: {str(e)}")
        return False

def check_standard_document(docx_zip: zipfile.ZipFile) -> bool:
    """检查标准document.xml格式"""
    try:
        if 'word/document.xml' in docx_zip.namelist():
            doc_xml = docx_zip.read('word/document.xml').decode('utf-8', errors='replace')
            
            # 分析document.xml结构
            text_elements = doc_xml.count('<w:t>')
            paragraph_elements = doc_xml.count('<w:p>')
            run_elements = doc_xml.count('<w:r>')
            
            print(f"    📊 文档统计:")
            print(f"         段落数: {paragraph_elements}")
            print(f"         文本运行数: {run_elements}")
            print(f"         文本元素数: {text_elements}")
            
            return text_elements > 0
        return False
    except Exception as e:
        print(f"    ❌ 检查标准文档失败: {str(e)}")
        return False

def check_headers_footers(docx_zip: zipfile.ZipFile) -> list:
    """检查页眉页脚"""
    try:
        headers_footers = []
        for filename in docx_zip.namelist():
            if ('word/header' in filename or 'word/footer' in filename) and filename.endswith('.xml'):
                headers_footers.append(filename)
                
                # 分析内容
                content = docx_zip.read(filename).decode('utf-8', errors='replace')
                text_count = content.count('<w:t>')
                print(f"    📄 {filename}: {text_count} 个文本元素")
        
        return headers_footers
    except Exception as e:
        print(f"    ❌ 检查页眉页脚失败: {str(e)}")
        return []

def check_comments(docx_zip: zipfile.ZipFile) -> bool:
    """检查注释"""
    try:
        if 'word/comments.xml' in docx_zip.namelist():
            comments_xml = docx_zip.read('word/comments.xml').decode('utf-8', errors='replace')
            comment_count = comments_xml.count('<w:comment')
            print(f"    💬 注释数量: {comment_count}")
            return comment_count > 0
        return False
    except Exception as e:
        print(f"    ❌ 检查注释失败: {str(e)}")
        return False

def check_textboxes(docx_zip: zipfile.ZipFile) -> int:
    """检查文本框"""
    try:
        textbox_count = 0
        for filename in docx_zip.namelist():
            if 'word/' in filename and filename.endswith('.xml'):
                content = docx_zip.read(filename).decode('utf-8', errors='replace')
                textbox_count += content.count('<w:txbxContent')
        
        if textbox_count > 0:
            print(f"    📦 文本框数量: {textbox_count}")
        
        return textbox_count
    except Exception as e:
        print(f"    ❌ 检查文本框失败: {str(e)}")
        return 0

def check_embedded_objects(docx_zip: zipfile.ZipFile) -> list:
    """检查嵌入对象"""
    try:
        embedded = [f for f in docx_zip.namelist() if 'word/embeddings/' in f]
        
        for embed_file in embedded:
            info = docx_zip.getinfo(embed_file)
            print(f"    📎 嵌入对象: {embed_file} ({info.file_size} bytes)")
        
        return embedded
    except Exception as e:
        print(f"    ❌ 检查嵌入对象失败: {str(e)}")
        return []

def check_charts_shapes(docx_zip: zipfile.ZipFile) -> list:
    """检查图表和图形"""
    try:
        charts_shapes = []
        
        # 查找图表文件
        charts = [f for f in docx_zip.namelist() if 'word/charts/' in f]
        
        # 查找绘图文件
        drawings = [f for f in docx_zip.namelist() if 'word/drawings/' in f]
        
        # 查找媒体文件 (图片等)
        media = [f for f in docx_zip.namelist() if 'word/media/' in f]
        
        if charts:
            print(f"    📊 图表文件: {len(charts)} 个")
            charts_shapes.extend(charts)
        
        if drawings:
            print(f"    🎨 绘图文件: {len(drawings)} 个")
            charts_shapes.extend(drawings)
        
        if media:
            print(f"    🖼️ 媒体文件: {len(media)} 个")
            charts_shapes.extend(media)
        
        return charts_shapes
    except Exception as e:
        print(f"    ❌ 检查图表图形失败: {str(e)}")
        return []

def explain_docx_complexity():
    """解释DOCX文件格式的复杂性"""
    
    print("\n" + "=" * 80)
    print("📚 DOCX文件格式复杂性详解")
    print("=" * 80)
    
    explanations = [
        ("🔄 altChunk + MHT格式", [
            "用途: 嵌入来自外部应用的富文本内容",
            "场景: 从网页复制粘贴、插入HTML内容、保持原始格式",
            "存储: 内容存储在.mht文件中，使用MIME HTML格式",
            "挑战: 需要解析MIME格式和quoted-printable编码"
        ]),
        
        ("📄 标准XML格式", [
            "用途: Word原生文档内容",
            "场景: 直接在Word中编辑的文档",
            "存储: 内容存储在document.xml中，使用OpenXML标准",
            "挑战: 复杂的XML命名空间和嵌套结构"
        ]),
        
        ("📑 页眉页脚", [
            "用途: 页面顶部和底部的重复内容", 
            "场景: 文档标题、页码、版权信息",
            "存储: 独立的header*.xml和footer*.xml文件",
            "挑战: 可能包含重要的标题或联系信息"
        ]),
        
        ("💬 注释和批注", [
            "用途: 文档的审阅和评论信息",
            "场景: 协作编辑、审稿意见、修改建议", 
            "存储: comments.xml文件",
            "挑战: 注释可能包含重要的解释或补充信息"
        ]),
        
        ("📦 文本框和图形", [
            "用途: 独立定位的文本容器",
            "场景: 侧边栏、标注、图表标签",
            "存储: 嵌套在主文档XML中的txbxContent元素",
            "挑战: 文本框中的内容可能包含关键信息"
        ]),
        
        ("📎 嵌入对象", [
            "用途: 插入其他类型的文档",
            "场景: Excel表格、PDF文件、其他Word文档",
            "存储: word/embeddings/目录下的二进制文件",
            "挑战: 需要相应的解析器来处理不同类型的嵌入文档"
        ]),
        
        ("🖼️ 图表和媒体", [
            "用途: 图片、图表、图形等视觉元素",
            "场景: 数据可视化、插图、截图",
            "存储: word/media/和word/charts/目录",
            "挑战: 可能包含OCR可提取的文字信息"
        ])
    ]
    
    for title, items in explanations:
        print(f"\n{title}:")
        for item in items:
            print(f"  • {item}")

def demonstrate_extraction_strategies():
    """演示不同的提取策略"""
    
    print("\n" + "=" * 80)
    print("🛠️ 内容提取策略")
    print("=" * 80)
    
    strategies = [
        ("🎯 优先级策略", [
            "1. altChunk内容 (通常是主要内容)",
            "2. 标准document.xml内容",
            "3. 页眉页脚 (可能包含标题信息)",
            "4. 文本框 (重要的补充信息)",
            "5. 注释 (审阅和解释信息)",
            "6. 嵌入对象 (其他文档内容)"
        ]),
        
        ("🔄 回退机制", [
            "1. 尝试综合提取所有内容类型",
            "2. 如果失败，使用LibreOffice转换",
            "3. 如果还失败，尝试基础XML解析",
            "4. 最后使用python-docx库"
        ]),
        
        ("⚡ 性能优化", [
            "1. 按文件大小选择策略",
            "2. 缓存解析结果",
            "3. 并行处理多个文件",
            "4. 懒加载可选内容"
        ])
    ]
    
    for title, items in strategies:
        print(f"\n{title}:")
        for item in items:
            print(f"  {item}")

if __name__ == "__main__":
    # 分析示例文件
    docx_file = "examples/datamax.docx"
    if os.path.exists(docx_file):
        analyze_docx_structure(docx_file)
    else:
        print(f"❌ 示例文件不存在: {docx_file}")
    
    # 解释复杂性
    explain_docx_complexity()
    
    # 演示策略
    demonstrate_extraction_strategies()
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80) 