#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Agent Test Script - Simple test to verify MCP Agent functionality

This script provides basic tests for the MCP Agent without requiring actual API keys.
It uses mock implementations to test the core functionality.

Author: DataMax Team
Date: 2024
License: MIT
"""

import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the MCP Agent components
try:
    from datamax.generator.mcp_agent_generator import (
        MCPAgent,
        MCPAgentConfig,
        ToolDefinition,
        ToolType,
        MessageRole,
        MCPMessage
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    exit(1)


class MockLLMClient:
    """Mock LLM client for testing without API calls"""
    
    def __init__(self):
        self.call_count = 0
    
    class ChatCompletions:
        def __init__(self, client):
            self.client = client
        
        def create(self, **kwargs):
            self.client.call_count += 1
            
            # Mock response structure
            class MockChoice:
                def __init__(self):
                    self.message = Mock()
                    self.message.content = f"Mock response #{self.client.call_count}: I understand your request and I'm here to help!"
                    self.message.tool_calls = None
            
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
            
            return MockResponse()
    
    def __init__(self, *args, **kwargs):
        self.chat = Mock()
        self.chat.completions = self.ChatCompletions(self)


def create_test_agent() -> MCPAgent:
    """Create a test MCP agent with mock LLM client"""
    config = MCPAgentConfig(
        name="Test-MCP-Agent",
        api_key="test-key",  # Mock API key
        model_name="test-model",
        temperature=0.7,
        enable_tool_calling=True,
        enable_memory=True,
        log_level="INFO"
    )
    
    agent = MCPAgent(config)
    
    # Replace the LLM client with mock
    agent.llm_client = MockLLMClient()
    
    return agent


def test_agent_initialization():
    """Test agent initialization"""
    print("🧪 Testing agent initialization...")
    
    agent = create_test_agent()
    
    # Check basic properties
    assert agent.config.name == "Test-MCP-Agent"
    assert len(agent.tools) > 0  # Should have default tools
    assert agent.session_id is not None
    
    # Check default tools
    expected_tools = ["process_data", "file_operation", "web_search"]
    for tool_name in expected_tools:
        assert tool_name in agent.tools, f"Missing default tool: {tool_name}"
    
    print("✅ Agent initialization test passed")
    return agent


def test_tool_registration():
    """Test custom tool registration"""
    print("🧪 Testing tool registration...")
    
    agent = create_test_agent()
    
    # Define a test tool
    def test_tool(message: str) -> Dict[str, Any]:
        return {"success": True, "echo": message}
    
    tool_def = ToolDefinition(
        name="test_echo",
        description="Echo test tool",
        type=ToolType.FUNCTION,
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            }
        },
        required=["message"],
        implementation=test_tool
    )
    
    # Register the tool
    initial_tool_count = len(agent.tools)
    agent.register_tool(tool_def)
    
    # Verify registration
    assert len(agent.tools) == initial_tool_count + 1
    assert "test_echo" in agent.tools
    assert agent.tools["test_echo"].implementation == test_tool
    
    # Test tool execution
    result = test_tool("Hello, World!")
    assert result["success"] is True
    assert result["echo"] == "Hello, World!"
    
    print("✅ Tool registration test passed")
    return agent


def test_message_handling():
    """Test message handling"""
    print("🧪 Testing message handling...")
    
    agent = create_test_agent()
    
    # Test adding messages
    initial_count = len(agent.conversation_history)
    
    message = agent.add_message(MessageRole.USER, "Test message")
    
    assert len(agent.conversation_history) == initial_count + 1
    assert message.role == MessageRole.USER
    assert message.content == "Test message"
    assert message.id is not None
    assert message.timestamp is not None
    
    print("✅ Message handling test passed")
    return agent


async def test_conversation_processing():
    """Test conversation processing with mock LLM"""
    print("🧪 Testing conversation processing...")
    
    agent = create_test_agent()
    
    # Test basic conversation
    response = await agent.process_message("Hello, how are you?")
    
    assert isinstance(response, MCPMessage)
    assert response.role == MessageRole.ASSISTANT
    assert response.content is not None
    assert len(response.content) > 0
    
    # Check conversation history
    assert len(agent.conversation_history) >= 2  # User message + Assistant response
    
    print("✅ Conversation processing test passed")
    return agent


def test_default_tools():
    """Test default tool implementations"""
    print("🧪 Testing default tools...")
    
    agent = create_test_agent()
    
    # Test data processing tool
    data_tool = agent.tools["process_data"].implementation
    result = data_tool("Hello\n\nWorld", "clean")
    assert result["success"] is True
    assert result["operation"] == "clean"
    
    result = data_tool("Hello World", "analyze")
    assert result["success"] is True
    assert "word_count" in result["result"]
    
    # Test web search tool (mock)
    search_tool = agent.tools["web_search"].implementation
    result = search_tool("test query", 3)
    assert result["success"] is True
    assert "result" in result
    assert len(result["result"]) <= 3
    
    print("✅ Default tools test passed")
    return agent


def test_file_operations():
    """Test file operation tool"""
    print("🧪 Testing file operations...")
    
    agent = create_test_agent()
    file_tool = agent.tools["file_operation"].implementation
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write("Test content for file operations")
        tmp_file_path = tmp_file.name
    
    try:
        # Test file read
        result = file_tool(tmp_file_path, "read")
        assert result["success"] is True
        assert "Test content" in result["result"]
        
        # Test file analyze
        result = file_tool(tmp_file_path, "analyze")
        assert result["success"] is True
        assert "size" in result["result"]
        assert "extension" in result["result"]
        
        # Test file write
        new_content = "New test content"
        result = file_tool(tmp_file_path, "write", new_content)
        assert result["success"] is True
        
        # Verify write worked
        result = file_tool(tmp_file_path, "read")
        assert result["success"] is True
        assert result["result"] == new_content
        
    finally:
        # Clean up
        os.unlink(tmp_file_path)
    
    print("✅ File operations test passed")
    return agent


def test_conversation_export():
    """Test conversation export functionality"""
    print("🧪 Testing conversation export...")
    
    agent = create_test_agent()
    
    # Add some test messages
    agent.add_message(MessageRole.USER, "Hello")
    agent.add_message(MessageRole.ASSISTANT, "Hi there!")
    agent.add_message(MessageRole.USER, "How are you?")
    agent.add_message(MessageRole.ASSISTANT, "I'm doing well, thank you!")
    
    # Test JSON export
    json_export = agent.export_conversation("json")
    assert isinstance(json_export, str)
    
    # Verify it's valid JSON
    parsed_json = json.loads(json_export)
    assert isinstance(parsed_json, list)
    assert len(parsed_json) == 4
    
    # Test Markdown export
    md_export = agent.export_conversation("markdown")
    assert isinstance(md_export, str)
    assert "# Conversation History" in md_export
    assert "Session ID" in md_export
    
    # Test conversation summary
    summary = agent.get_conversation_summary()
    assert "session_id" in summary
    assert "agent_name" in summary
    assert "message_count" in summary
    assert summary["message_count"] == 4
    
    print("✅ Conversation export test passed")
    return agent


def test_error_handling():
    """Test error handling"""
    print("🧪 Testing error handling...")
    
    agent = create_test_agent()
    
    # Test invalid tool operation
    data_tool = agent.tools["process_data"].implementation
    result = data_tool("test", "invalid_operation")
    assert result["success"] is False
    assert "error" in result
    
    # Test file operation with non-existent file
    file_tool = agent.tools["file_operation"].implementation
    result = file_tool("/non/existent/file.txt", "read")
    assert result["success"] is False
    assert "error" in result
    
    # Test invalid export format
    try:
        agent.export_conversation("invalid_format")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    print("✅ Error handling test passed")
    return agent


async def run_all_tests():
    """Run all tests"""
    print("🚀 Running MCP Agent Tests")
    print("=" * 50)
    
    try:
        # Run synchronous tests
        test_agent_initialization()
        test_tool_registration()
        test_message_handling()
        test_default_tools()
        test_file_operations()
        test_conversation_export()
        test_error_handling()
        
        # Run asynchronous tests
        await test_conversation_processing()
        
        print("\n🎉 All tests passed successfully!")
        print("\n📋 Test Summary:")
        print("✅ Agent initialization")
        print("✅ Tool registration")
        print("✅ Message handling")
        print("✅ Conversation processing")
        print("✅ Default tools")
        print("✅ File operations")
        print("✅ Conversation export")
        print("✅ Error handling")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("DataMax MCP Agent - Test Suite")
    print("This test suite verifies the core functionality without requiring API keys.")
    print()
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Set up your API keys (OPENAI_API_KEY or DASHSCOPE_API_KEY)")
        print("2. Run the full example: python examples/mcp_agent_example.py")
        print("3. Check the documentation: examples/README_MCP_Agent.md")
    else:
        print("\n🔧 Please fix the failing tests before proceeding.")
        exit(1)