import os
import sys
from datamax import DataMax

def test_enhanced_qa_generation():
    """测试增强QA生成功能"""
    
    # 文件路径
    file_path = r"C:\Users\lysnoir\OneDrive\桌面\Transactions-instructions-only.md"
    
    # API配置
    api_key = "sk-bfddf3a562fb40b38bec21199dec6d82"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    model_name = "qwen2.5-7b-instruct"  # 使用通义千问模型
    
    print("=" * 60)
    print("增强QA生成功能测试")
    print("=" * 60)
    
    try:
        # 1. 初始化DataMax
        print("1. 初始化DataMax...")
        dm = DataMax(file_path=file_path)
        print(f"   文件路径: {file_path}")
        print(f"   文件存在: {os.path.exists(file_path)}")
        
        # 2. 获取解析数据
        print("\n2. 解析文件数据...")
        data = dm.get_data()
        content = data.get('content', '')
        print(f"   解析内容长度: {len(content)} 字符")
        print(f"   内容预览: {content[:200]}...")
        
        # 3. 测试增强QA生成（包含领域树集成和交互式编辑）
        print("\n3. 测试增强QA生成...")
        print("   启用功能:")
        print("   - use_tree_label=True (领域树集成)")
        print("   - interactive_tree=True (交互式树编辑)")
        
        qa_data = dm.get_pre_label(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            use_tree_label=True,      # 领域树集成参数
            interactive_tree=True,    # 在QA生成过程中启用交互式树编辑
            chunk_size=500,           # 文本块大小
            chunk_overlap=100,        # 重叠长度
            question_number=1,        # 每块生成问题数（减少到3个便于测试）
            max_workers=2             # 并发数（减少到2个便于测试）
        )
        
        # 4. 显示结果
        print("\n4. QA生成结果:")
        print(f"   生成状态: 成功")
        print(f"   结果类型: {type(qa_data)}")
        
        if isinstance(qa_data, dict):
            print(f"   结果键: {list(qa_data.keys())}")
            
            # 显示QA对数量
            if 'qa_pairs' in qa_data:
                qa_pairs = qa_data['qa_pairs']
                print(f"   生成的QA对数量: {len(qa_pairs)}")
                
                # 显示前3个QA对
                print("\n   前3个QA对:")
                for i, qa in enumerate(qa_pairs[:3]):
                    print(f"   {i+1}. 问题: {qa.get('question', 'N/A')}")
                    print(f"      答案: {qa.get('answer', 'N/A')[:100]}...")
                    print()
            
            # 显示领域树信息
            if 'domain_tree' in qa_data:
                domain_tree = qa_data['domain_tree']
                print(f"   领域树信息: {type(domain_tree)}")
                print(f"   领域树内容: {domain_tree}")
        
        # 5. 保存结果
        print("\n5. 保存结果...")
        dm.save_label_data(qa_data)
        print("   结果已保存")
        
        print("\n" + "=" * 60)
        print("测试完成！增强QA生成功能正常工作")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return False

def main():
    """主函数"""
    print("开始测试增强QA生成功能...")
    success = test_enhanced_qa_generation()
    
    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)

if __name__ == "__main__":
    main() 