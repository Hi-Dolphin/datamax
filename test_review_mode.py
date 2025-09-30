#!/usr/bin/env python3
"""
Test script for the review_mode functionality in DataMax.get_pre_label
"""

import os
import sys
import json

# Add the datamax directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from datamax.parser.core import DataMax

def test_review_mode():
    """Test the review_mode parameter in get_pre_label"""
    
    # Create a simple test content
    test_content = """
    这是一个测试文档，用于验证review_mode功能。
    该文档包含一些基本信息，可以用来生成问答对。
    测试内容应该足够简单，以便大模型能够正确评分。
    """
    
    # Initialize DataMax
    dm = DataMax()
    
    # Test parameters (you'll need to provide your own API credentials)
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    if api_key == "your-api-key-here":
        print("Please set OPENAI_API_KEY environment variable with your actual API key")
        return
    
    try:
        # Test without review mode
        print("Testing without review mode...")
        qa_pairs_normal = dm.get_pre_label(
            content=test_content,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            question_number=2,
            debug=True
        )
        
        print(f"Generated {len(qa_pairs_normal) if isinstance(qa_pairs_normal, list) else len(qa_pairs_normal.get('qa_pairs', []))} QA pairs without review mode")
        
        # Test with review mode
        print("\nTesting with review mode...")
        qa_pairs_reviewed = dm.get_pre_label(
            content=test_content,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            question_number=2,
            review_mode=True,
            debug=True
        )
        
        print(f"Generated {len(qa_pairs_reviewed) if isinstance(qa_pairs_reviewed, list) else len(qa_pairs_reviewed.get('qa_pairs', []))} QA pairs with review mode")
        
        print("Review mode test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_review_mode()
