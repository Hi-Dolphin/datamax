#!/usr/bin/env python3
"""
测试增强的DOC解析器功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamax import DataMax
from datamax.parser.doc_parser import DocParser

def test_doc_parser():
    """测试DOC解析器的各种功能"""
    
    print("=" * 80)
    print("🧪 测试增强的DOC解析器")
    print("=" * 80)
    
    # 查找测试文件
    test_files = []
    for file in os.listdir("examples"):
        if file.endswith(".doc"):
            test_files.append(os.path.join("examples", file))
    
    if not test_files:
        print("⚠️ 没有找到.doc测试文件")
        print("📝 请在examples目录下放置一些.doc文件进行测试")
        return
    
    for doc_file in test_files:
        print(f"\n📄 测试文件: {doc_file}")
        print("-" * 40)
        
        try:
            # 1. 使用DataMax解析
            print("\n🔧 使用DataMax解析:")
            dm = DataMax(file_path=doc_file)
            data = dm.get_data()
            
            print(f"  ✅ 标题: {data.get('title', 'N/A')}")
            print(f"  ✅ 内容长度: {len(data.get('content', ''))} 字符")
            print(f"  ✅ 内容预览: {data.get('content', '')}...")
            
            # 2. 直接使用DocParser（启用高级功能）
            print("\n🔧 直接使用DocParser (高级模式):")
            parser = DocParser(file_path=doc_file, use_uno=True)
            result = parser.parse(doc_file)
            
            print(f"  ✅ 内容长度: {len(result.get('content', ''))} 字符")
            
            # 3. 检查是否有OLE支持
            try:
                import olefile
                print("\n🔍 OLE解析支持: ✅ 可用")
                
                # 检查文件的OLE结构
                with olefile.OleFileIO(doc_file) as ole:
                    print("  📂 OLE目录结构:")
                    for entry in ole.listdir()[:10]:  # 只显示前10个
                        print(f"    - {'/'.join(entry)}")
                    
                    if len(ole.listdir()) > 10:
                        print(f"    ... 还有 {len(ole.listdir()) - 10} 个条目")
                        
            except ImportError:
                print("\n🔍 OLE解析支持: ❌ 不可用 (请安装: pip install olefile)")
            except Exception as e:
                print(f"\n🔍 OLE解析支持: ❌ 错误 - {str(e)}")
            
        except Exception as e:
            print(f"  ❌ 解析失败: {str(e)}")
            import traceback
            traceback.print_exc()

def explain_doc_enhancements():
    """解释DOC解析器的增强功能"""
    
    print("\n" + "=" * 80)
    print("📚 DOC解析器增强功能说明")
    print("=" * 80)
    
    enhancements = [
        ("🔍 OLE文件格式解析", [
            "直接读取DOC文件的二进制结构",
            "提取WordDocument流中的文本",
            "支持多种编码格式（UTF-16LE、GBK、GB2312等）",
            "无需依赖外部转换工具"
        ]),
        
        ("📎 嵌入对象提取", [
            "检测并提取嵌入的文档",
            "支持OLE对象（Excel、PowerPoint等）",
            "提取Package对象中的内容",
            "处理复合文档结构"
        ]),
        
        ("🧹 智能文本清理", [
            "移除控制字符和格式化代码",
            "保持段落结构和可读性",
            "过滤无意义的短文本片段",
            "处理多种字符编码"
        ]),
        
        ("🔄 多层回退机制", [
            "优先使用OLE直接解析",
            "回退到UNO API转换",
            "最后使用命令行转换",
            "确保内容总能提取"
        ]),
        
        ("⚡ 性能优化", [
            "直接二进制解析更快",
            "减少外部依赖",
            "支持大文件处理",
            "内存使用优化"
        ])
    ]
    
    for title, items in enhancements:
        print(f"\n{title}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n💡 使用建议:")
    print("  1. 安装olefile以启用高级解析: pip install olefile")
    print("  2. 对于复杂的DOC文件，高级解析能提取更多内容")
    print("  3. 传统方式仍然可用，作为可靠的回退选项")

if __name__ == "__main__":
    test_doc_parser()
    explain_doc_enhancements()
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80) 