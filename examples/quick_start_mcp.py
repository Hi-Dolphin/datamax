#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Guide for DataMax MCP Agent

This script provides a simple way to get started with the MCP Agent.
It includes both mock examples (no API key required) and real examples.

Usage:
    python examples/quick_start_mcp.py

Author: DataMax Team
Date: 2024
License: MIT
"""

import asyncio
import os
from datamax.generator.mcp_agent_generator import (
    MCPAgent,
    MCPAgentConfig,
    create_demo_agent
)


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 60)
    print("🚀 DataMax MCP Agent - Quick Start Guide")
    print("=" * 60)
    print("\nWelcome to the DataMax MCP (Model Context Protocol) Agent!")
    print("This intelligent agent combines LLM capabilities with tool calling.")
    print("\n📋 Features:")
    print("  • Natural language conversation")
    print("  • Dynamic tool calling")
    print("  • Data processing capabilities")
    print("  • File operations")
    print("  • Conversation memory")
    print("  • Export functionality")
    print()


def check_api_keys():
    """Check if API keys are available"""
    openai_key = os.getenv("OPENAI_API_KEY")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    
    print("🔑 API Key Status:")
    print(f"  OpenAI: {'✅ Available' if openai_key else '❌ Not set'}")
    print(f"  DashScope: {'✅ Available' if dashscope_key else '❌ Not set'}")
    
    if not openai_key and not dashscope_key:
        print("\n⚠️  No API keys found. Running in mock mode only.")
        print("\n💡 To use real LLM capabilities, set one of these environment variables:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("   export DASHSCOPE_API_KEY='your-dashscope-key'")
        return False
    
    return True


class MockAgent:
    """Mock agent for demonstration without API calls"""
    
    def __init__(self):
        self.conversation_count = 0
        self.tools = ["process_data", "file_operation", "web_search"]
    
    async def process_message(self, message: str) -> str:
        """Process message with mock responses"""
        self.conversation_count += 1
        
        # Simple rule-based responses for demo
        message_lower = message.lower()
        
        if "hello" in message_lower or "hi" in message_lower:
            return f"Hello! I'm the DataMax MCP Agent. I can help you with data processing, file operations, and more. This is conversation #{self.conversation_count}."
        
        elif "tools" in message_lower or "what can you do" in message_lower:
            return f"I have access to these tools: {', '.join(self.tools)}. I can process data, handle files, search the web, and much more!"
        
        elif "analyze" in message_lower or "process" in message_lower:
            return "I can analyze your data! For example, I can clean text, count words, extract patterns, and provide statistical insights. Just provide me with the data you'd like to analyze."
        
        elif "file" in message_lower:
            return "I can help with file operations! I can read files, write content to files, and analyze file properties like size and modification date."
        
        elif "search" in message_lower:
            return "I can perform web searches to find information for you. Just tell me what you'd like to search for!"
        
        elif "thank" in message_lower or "bye" in message_lower:
            return "You're welcome! It was great helping you. Feel free to come back anytime you need assistance with data processing or analysis!"
        
        else:
            return f"I understand your request: '{message}'. In a real scenario with API access, I would process this using my LLM capabilities and available tools. This is mock response #{self.conversation_count}."


async def mock_demo():
    """Run a mock demo without API calls"""
    print("\n🎭 Mock Demo (No API Required)")
    print("-" * 40)
    
    agent = MockAgent()
    
    demo_messages = [
        "Hello! What can you help me with?",
        "What tools do you have available?",
        "Can you help me analyze some text data?",
        "How about file operations?",
        "Can you search for information?",
        "Thank you for the demonstration!"
    ]
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n👤 User: {message}")
        response = await agent.process_message(message)
        print(f"🤖 Agent: {response}")
        
        # Add a small delay for better UX
        await asyncio.sleep(0.5)
    
    print("\n✅ Mock demo completed!")


async def real_demo_openai():
    """Run real demo with OpenAI API"""
    print("\n🌟 Real Demo with OpenAI")
    print("-" * 40)
    
    try:
        config = MCPAgentConfig(
            name="QuickStart-OpenAI-Agent",
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            enable_tool_calling=True,
            log_level="WARNING"  # Reduce log noise
        )
        
        agent = MCPAgent(config)
        
        demo_messages = [
            "Hello! I'm testing the DataMax MCP Agent.",
            "Can you analyze this text: 'DataMax is a powerful framework for data processing and analysis.'?",
            "What tools do you have available?"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"\n👤 User: {message}")
            response = await agent.process_message(message)
            print(f"🤖 Agent: {response.content}")
        
        print("\n✅ Real demo with OpenAI completed!")
        return agent
        
    except Exception as e:
        print(f"\n❌ Error in OpenAI demo: {e}")
        print("Please check your API key and internet connection.")
        return None


async def real_demo_dashscope():
    """Run real demo with DashScope API"""
    print("\n🌟 Real Demo with DashScope")
    print("-" * 40)
    
    try:
        config = MCPAgentConfig(
            name="QuickStart-DashScope-Agent",
            llm_provider="dashscope",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen-turbo",
            temperature=0.7,
            enable_tool_calling=True,
            log_level="WARNING"  # Reduce log noise
        )
        
        agent = MCPAgent(config)
        
        demo_messages = [
            "你好！我正在测试DataMax MCP代理。",
            "你能帮我分析这段文本吗：'DataMax是一个强大的数据处理和分析框架。'？",
            "你有哪些可用的工具？"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"\n👤 User: {message}")
            response = await agent.process_message(message)
            print(f"🤖 Agent: {response.content}")
        
        print("\n✅ Real demo with DashScope completed!")
        return agent
        
    except Exception as e:
        print(f"\n❌ Error in DashScope demo: {e}")
        print("Please check your API key and internet connection.")
        return None


def show_next_steps():
    """Show next steps for users"""
    print("\n" + "=" * 60)
    print("🎯 Next Steps")
    print("=" * 60)
    print("\n1. 📖 Read the documentation:")
    print("   examples/README_MCP_Agent.md")
    
    print("\n2. 🧪 Run the full example:")
    print("   python examples/mcp_agent_example.py")
    
    print("\n3. 🔧 Run tests:")
    print("   python examples/test_mcp_agent.py")
    
    print("\n4. 🚀 Build your own agent:")
    print("   from datamax.generator import MCPAgent, MCPAgentConfig")
    
    print("\n5. 🛠️ Customize tools:")
    print("   Create custom ToolDefinition objects")
    
    print("\n6. 📚 Learn more:")
    print("   • Check the source code in datamax/generator/mcp_agent_generator.py")
    print("   • Explore examples in the examples/ directory")
    print("   • Visit: https://github.com/Hi-Dolphin/datamax")
    
    print("\n💡 Tips:")
    print("   • Set API keys for full functionality")
    print("   • Customize agent configuration for your needs")
    print("   • Create domain-specific tools")
    print("   • Export conversations for analysis")
    print()


async def interactive_mode():
    """Interactive chat mode"""
    print("\n💬 Interactive Mode")
    print("-" * 40)
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Check if we can use real API
    has_api = check_api_keys()
    
    if has_api:
        try:
            if os.getenv("OPENAI_API_KEY"):
                config = MCPAgentConfig(
                    name="Interactive-Agent",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="gpt-3.5-turbo",
                    log_level="ERROR"  # Minimal logging
                )
                agent = MCPAgent(config)
                print("🌟 Using OpenAI API for real responses.")
            else:
                config = MCPAgentConfig(
                    name="Interactive-Agent",
                    llm_provider="dashscope",
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model_name="qwen-turbo",
                    log_level="ERROR"
                )
                agent = MCPAgent(config)
                print("🌟 Using DashScope API for real responses.")
        except Exception as e:
            print(f"⚠️  API setup failed: {e}")
            print("Falling back to mock mode.")
            agent = MockAgent()
            has_api = False
    else:
        agent = MockAgent()
        print("🎭 Using mock responses (no API key).")
    
    print()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Agent: Goodbye! Thanks for trying the DataMax MCP Agent!")
                break
            
            if not user_input:
                continue
            
            print("🤖 Agent: ", end="", flush=True)
            
            if has_api and isinstance(agent, MCPAgent):
                response = await agent.process_message(user_input)
                print(response.content)
            else:
                response = await agent.process_message(user_input)
                print(response)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


async def main():
    """Main function"""
    print_banner()
    
    # Check API availability
    has_api = check_api_keys()
    
    print("\n🎮 Choose a demo mode:")
    print("1. Mock Demo (No API required)")
    if has_api:
        print("2. Real Demo with API")
    print("3. Interactive Chat")
    print("4. Show Next Steps")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                await mock_demo()
            elif choice == "2" and has_api:
                if os.getenv("OPENAI_API_KEY"):
                    await real_demo_openai()
                elif os.getenv("DASHSCOPE_API_KEY"):
                    await real_demo_dashscope()
            elif choice == "3":
                await interactive_mode()
            elif choice == "4":
                show_next_steps()
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")