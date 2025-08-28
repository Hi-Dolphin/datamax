#!/usr/bin/env python3
\"\"\"Test to verify storage adapter usage in one-line crawler example.\"\"\"

import sys
from pathlib import Path

def test_example_storage_adapter():
    \"\"\"Test that the example file includes storage adapter usage.\"\"\"
    print(\"Testing storage adapter usage in one-line crawler example...\")
    
    # Read the example file
    example_file_path = Path(__file__).parent / \"one_line_crawler_example.py\"
    
    try:
        with open(example_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if storage adapter is imported
        if 'from datamax.crawler.storage_adapter import create_storage_adapter' in content:
            print(\"[PASS] Found storage adapter import\")
        else:
            print(\"[FAIL] Storage adapter import not found\")
            return False
            
        # Check if storage adapter is created
        if 'storage_adapter = create_storage_adapter' in content:
            print(\"[PASS] Found storage adapter creation\")
        else:
            print(\"[FAIL] Storage adapter creation not found\")
            return False
            
        # Check if storage adapter is used for saving
        if 'storage_adapter.save' in content:
            print(\"[PASS] Found storage adapter save usage\")
        else:
            print(\"[FAIL] Storage adapter save usage not found\")
            return False
            
        # Check if environment variables are used for configuration
        if 'os.environ.get(' in content:
            print(\"[PASS] Found environment variable usage\")
        else:
            print(\"[FAIL] Environment variable usage not found\")
            return False
            
        # Check if there's a fallback to JSON
        if 'Falling back to JSON file saving' in content:
            print(\"[PASS] Found fallback to JSON saving\")
        else:
            print(\"[FAIL] Fallback to JSON saving not found\")
            return False
            
    except Exception as e:
        print(f\"[FAIL] Error reading example file: {e}\")
        return False
    
    return True

if __name__ == \"__main__\":
    success = test_example_storage_adapter()
    if success:
        print(\"\\nSUCCESS: Storage adapter usage has been successfully added to one-line crawler example.\")
    else:
        print(\"\\nFAILURE: Storage adapter implementation in example has issues.\")
        sys.exit(1)