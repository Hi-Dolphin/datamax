#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Request Logging - 测试MCP代理的请求体保存功能

This script tests the request body logging functionality of the MCP Agent.
It will save all LLM requests to files for debugging purposes.

Usage:
    python examples/test_request_logging.py

Author: DataMax Team
Date: 2024
License: MIT
"""

import asyncio
import os
import json
import glob
from datetime import datetime
from datamax.generator.mcp_agent_generator import (
    MCPAgent,
    MCPAgentConfig,
    ToolDefinition,
    ToolType
)


def create_test_agent_with_logging():
    """Create a test agent with request logging enabled"""
    # Use mock API key for testing
    config = MCPAgentConfig(
        name="Request-Logging-Test-Agent",
        api_key="test-api-key",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        enable_tool_calling=True,
        enable_memory=True,
        log_level="INFO"
    )
    
    agent = MCPAgent(config)
    
    # Add a simple test tool
    def test_calculator(expression: str) -> dict:
        """Simple calculator for testing"""
        try:
            # Safe evaluation for simple math
            allowed_chars = set('0123456789+-*/().')
            if all(c in allowed_chars or c.isspace() for c in expression):
                result = eval(expression)
                return {"success": True, "result": result, "expression": expression}
            else:
                return {"success": False, "error": "Invalid characters in expression"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Register the test tool
    calc_tool = ToolDefinition(
        name="calculator",
        description="Perform simple mathematical calculations",
        type=ToolType.FUNCTION,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                }
            }
        },
        required=["expression"],
        implementation=test_calculator
    )
    
    agent.register_tool(calc_tool)
    return agent


class MockLLMClient:
    """Mock LLM client for testing without real API calls"""
    
    def __init__(self):
        self.call_count = 0
    
    class ChatCompletions:
        def __init__(self, client):
            self.client = client
        
        def create(self, **kwargs):
            self.client.call_count += 1
            
            # Mock response with tool calls for testing
            class MockToolCall:
                def __init__(self):
                    self.id = f"call_test_{self.client.call_count}"
                    self.function = MockFunction()
            
            class MockFunction:
                def __init__(self):
                    self.name = "calculator"
                    self.arguments = '{"expression": "2 + 3"}'
            
            class MockChoice:
                def __init__(self, client):
                    self.message = MockMessage(client)
            
            class MockMessage:
                def __init__(self, client):
                    if client.call_count == 1:
                        # First call - return tool call
                        self.content = "I'll calculate that for you."
                        self.tool_calls = [MockToolCall()]
                    else:
                        # Second call - return final response
                        self.content = f"The calculation result is 5. This is response #{client.call_count}."
                        self.tool_calls = None
            
            class MockResponse:
                def __init__(self, client):
                    self.choices = [MockChoice(client)]
            
            return MockResponse(self.client)
    
    def __init__(self, *args, **kwargs):
        self.chat = self.ChatCompletions(self)


def clear_previous_logs():
    """Clear previous log files"""
    log_dir = "mcp_request_logs"
    if os.path.exists(log_dir):
        log_files = glob.glob(f"{log_dir}/*.json")
        for file in log_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove {file}: {e}")
        print(f"Cleared {len(log_files)} previous log files.")
    else:
        print("No previous log directory found.")


def analyze_saved_requests():
    """Analyze and display saved request files"""
    log_dir = "mcp_request_logs"
    if not os.path.exists(log_dir):
        print("❌ No log directory found!")
        return
    
    log_files = glob.glob(f"{log_dir}/*.json")
    if not log_files:
        print("❌ No log files found!")
        return
    
    print(f"\n📊 Found {len(log_files)} request log files:")
    print("=" * 60)
    
    for i, file_path in enumerate(sorted(log_files), 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            print(f"\n{i}. {filename}")
            print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
            print(f"   Agent: {data.get('agent_name', 'N/A')}")
            print(f"   Request Type: {data.get('request_type', 'N/A')}")
            print(f"   Session ID: {data.get('session_id', 'N/A')}")
            
            request_body = data.get('request_body', {})
            print(f"   Model: {request_body.get('model', 'N/A')}")
            print(f"   Temperature: {request_body.get('temperature', 'N/A')}")
            print(f"   Max Tokens: {request_body.get('max_tokens', 'N/A')}")
            print(f"   Messages Count: {len(request_body.get('messages', []))}")
            print(f"   Tools Count: {len(request_body.get('tools', []))}")
            
            # Show message types
            messages = request_body.get('messages', [])
            if messages:
                message_types = [msg.get('role', 'unknown') for msg in messages]
                print(f"   Message Roles: {', '.join(message_types)}")
            
            # Show tools if present
            tools = request_body.get('tools', [])
            if tools:
                tool_names = [tool.get('function', {}).get('name', 'unknown') for tool in tools]
                print(f"   Tool Names: {', '.join(tool_names)}")
            
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("💡 You can examine the complete request details in the JSON files.")


def show_sample_request_content():
    """Show detailed content of the first request file"""
    log_dir = "mcp_request_logs"
    log_files = glob.glob(f"{log_dir}/*.json")
    
    if not log_files:
        print("❌ No log files to display!")
        return
    
    # Show the first file content
    first_file = sorted(log_files)[0]
    print(f"\n📄 Sample Request Content ({os.path.basename(first_file)}):")
    print("=" * 60)
    
    try:
        with open(first_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Pretty print the JSON
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")


async def test_request_logging():
    """Test the request logging functionality"""
    print("🧪 Testing MCP Agent Request Logging")
    print("=" * 50)
    
    # Clear previous logs
    clear_previous_logs()
    
    # Create test agent
    print("\n1. Creating test agent...")
    agent = create_test_agent_with_logging()
    
    # Replace with mock client to avoid real API calls
    print("2. Setting up mock LLM client...")
    agent.llm_client = MockLLMClient()
    
    # Test messages that should trigger tool calls
    test_messages = [
        "Hello! Can you calculate 2 + 3 for me?",
        "What's 10 * 5?",
        "Please compute (15 + 25) / 2"
    ]
    
    print("\n3. Processing test messages...")
    for i, message in enumerate(test_messages, 1):
        print(f"\n   Processing message {i}: {message}")
        try:
            response = await agent.process_message(message)
            print(f"   ✅ Response: {response.content[:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n4. Analyzing saved request logs...")
    analyze_saved_requests()
    
    print("\n5. Showing sample request content...")
    show_sample_request_content()
    
    print("\n✅ Request logging test completed!")
    print("\n💡 Check the 'mcp_request_logs' directory for all saved requests.")


def main():
    """Main function"""
    print("🔍 DataMax MCP Agent - Request Logging Test")
    print("=" * 60)
    print("This test will demonstrate the request body logging functionality.")
    print("All LLM requests will be saved to JSON files for inspection.\n")
    
    try:
        # Run the test
        asyncio.run(test_request_logging())
        
        print("\n🎯 Next Steps:")
        print("1. Check the 'mcp_request_logs' directory")
        print("2. Open the JSON files to see complete request details")
        print("3. Use this for debugging LLM API calls")
        print("4. Analyze conversation patterns and tool usage")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()