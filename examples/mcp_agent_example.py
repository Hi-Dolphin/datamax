#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Agent Example - Demonstrates how to use the DataMax MCP Agent Generator

This example shows how to:
1. Create and configure an MCP agent
2. Use the agent with different LLM providers
3. Implement custom tools
4. Handle conversations and tool calling

Author: DataMax Team
Date: 2024
License: MIT
"""

import asyncio
import os
from typing import Dict, Any

# Import the MCP Agent components
from datamax.generator.mcp_agent_generator import (
    MCPAgent,
    MCPAgentConfig,
    ToolDefinition,
    ToolType,
    MessageRole
)


def create_openai_agent() -> MCPAgent:
    """Create an MCP agent using OpenAI API"""
    config = MCPAgentConfig(
        name="DataMax-OpenAI-Agent",
        api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        base_url="https://api.openai.com/v1",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        enable_tool_calling=True,
        enable_memory=True,
        log_level="INFO"
    )
    return MCPAgent(config)


def create_dashscope_agent() -> MCPAgent:
    """Create an MCP agent using DashScope API (Alibaba Cloud)"""
    config = MCPAgentConfig(
        name="DataMax-DashScope-Agent",
        llm_provider="dashscope",
        api_key=os.getenv("DASHSCOPE_API_KEY", "your-dashscope-api-key"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-turbo",
        temperature=0.7,
        enable_tool_calling=True,
        enable_memory=True,
        log_level="INFO"
    )
    return MCPAgent(config)


def create_custom_tool_agent() -> MCPAgent:
    """Create an MCP agent with custom tools"""
    # Create base agent
    agent = create_dashscope_agent()
    
    # Define a custom calculator tool
    def calculator_tool(expression: str) -> Dict[str, Any]:
        """Safe calculator tool implementation"""
        try:
            # Simple expression evaluation (be careful with eval in production!)
            # This is a simplified example - use a proper math parser in production
            allowed_chars = set('0123456789+-*/().')
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return {"success": False, "error": "Invalid characters in expression"}
            
            result = eval(expression)
            return {"success": True, "result": result, "expression": expression}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Register the custom tool
    calculator_tool_def = ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations",
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
        implementation=calculator_tool
    )
    
    agent.register_tool(calculator_tool_def)
    
    # Define a text analysis tool
    def text_analysis_tool(text: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """Text analysis tool implementation"""
        try:
            if analysis_type == "basic":
                analysis = {
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "sentence_count": len([s for s in text.split('.') if s.strip()]),
                    "paragraph_count": len([p for p in text.split('\n\n') if p.strip()])
                }
            elif analysis_type == "advanced":
                import re
                analysis = {
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "unique_words": len(set(text.lower().split())),
                    "sentence_count": len(re.findall(r'[.!?]+', text)),
                    "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                    "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
                    "has_urls": bool(re.search(r'https?://', text)),
                    "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
                }
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
            
            return {"success": True, "result": analysis, "analysis_type": analysis_type}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Register the text analysis tool
    text_analysis_tool_def = ToolDefinition(
        name="text_analysis",
        description="Analyze text content and provide statistics",
        type=ToolType.FUNCTION,
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["basic", "advanced"],
                    "default": "basic"
                }
            }
        },
        required=["text"],
        implementation=text_analysis_tool
    )
    
    agent.register_tool(text_analysis_tool_def)
    
    return agent


async def basic_conversation_example():
    """Basic conversation example"""
    print("\n🔹 Basic Conversation Example")
    print("=" * 40)
    
    agent = create_dashscope_agent()
    
    messages = [
        "Hello! What can you help me with?",
        "Can you analyze this text: 'The quick brown fox jumps over the lazy dog. This is a sample sentence for testing.'",
        "What tools do you have available?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n👤 User: {message}")
        response = await agent.process_message(message)
        print(f"🤖 Agent: {response.content}")
    
    return agent


async def custom_tools_example():
    """Custom tools example"""
    print("\n🔹 Custom Tools Example")
    print("=" * 40)
    
    agent = create_custom_tool_agent()
    
    messages = [
        "Can you calculate 15 * 7 + 23?",
        "Please analyze this text: 'DataMax is an advanced data crawling and processing framework. It supports multiple file formats including PDF, DOCX, and more. Visit https://github.com/Hi-Dolphin/datamax for more information.'",
        "What's the result of (100 - 25) / 5?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n👤 User: {message}")
        response = await agent.process_message(message)
        print(f"🤖 Agent: {response.content}")
    
    return agent


async def file_operations_example():
    """File operations example"""
    print("\n🔹 File Operations Example")
    print("=" * 40)
    
    agent = create_dashscope_agent()
    
    # Create a sample file
    sample_content = """# DataMax Sample Data

This is a sample file for testing the MCP agent's file operations.

## Features
- Data crawling
- File parsing
- Content analysis
- LLM integration

## Statistics
- Lines: 12
- Words: 25
- Characters: 150+
"""
    
    messages = [
        f"Can you write this content to a file called 'sample_data.md'? Content: {sample_content}",
        "Now read the file 'sample_data.md' and tell me what it contains",
        "Can you analyze the file 'sample_data.md'?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n👤 User: {message}")
        response = await agent.process_message(message)
        print(f"🤖 Agent: {response.content}")
    
    return agent


async def conversation_export_example():
    """Conversation export example"""
    print("\n🔹 Conversation Export Example")
    print("=" * 40)
    
    agent = create_dashscope_agent()
    
    # Have a short conversation
    messages = [
        "Hello, I'm testing the conversation export feature.",
        "Can you help me process some data?",
        "Thank you for your help!"
    ]
    
    for message in messages:
        await agent.process_message(message)
    
    # Export conversation
    print("\n📊 Conversation Summary:")
    summary = agent.get_conversation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Export to JSON
    json_export = agent.export_conversation("json")
    with open("conversation_export.json", "w", encoding="utf-8") as f:
        f.write(json_export)
    print("\n✅ Conversation exported to 'conversation_export.json'")
    
    # Export to Markdown
    md_export = agent.export_conversation("markdown")
    with open("conversation_export.md", "w", encoding="utf-8") as f:
        f.write(md_export)
    print("✅ Conversation exported to 'conversation_export.md'")
    
    return agent


async def main():
    """Main example runner"""
    print("🚀 DataMax MCP Agent Examples")
    print("=" * 50)
    print("\nThis example demonstrates various features of the MCP Agent:")
    print("1. Basic conversation handling")
    print("2. Custom tool integration")
    print("3. File operations")
    print("4. Conversation export")
    print("\nNote: Make sure to set your API keys in environment variables:")
    print("- OPENAI_API_KEY for OpenAI")
    print("- DASHSCOPE_API_KEY for DashScope")
    
    try:
        # Run examples
        await basic_conversation_example()
        await custom_tools_example()
        await file_operations_example()
        await conversation_export_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure you have:")
        print("1. Set the appropriate API keys")
        print("2. Installed required dependencies (openai, pydantic, loguru)")
        print("3. Network access to the LLM API")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())