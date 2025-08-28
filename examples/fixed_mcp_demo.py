#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed MCP Agent Demo - Resolves API compatibility issues

This script provides a working demonstration of the MCP Agent with proper
error handling and API compatibility fixes.

Usage:
    python examples/fixed_mcp_demo.py

Author: DataMax Team
Date: 2024
License: MIT
"""

import asyncio
import os
import json
from typing import Dict, Any
from datamax.generator.mcp_agent_generator import (
    MCPAgent,
    MCPAgentConfig,
    ToolDefinition,
    ToolType,
    MessageRole
)


class FixedMCPAgent(MCPAgent):
    """Fixed MCP Agent with improved API compatibility"""
    
    def _prepare_messages_for_llm(self):
        """Prepare messages with better API compatibility"""
        messages = []
        
        # Add system message
        system_prompt = f"""
You are {self.config.name}, an advanced AI assistant powered by the DataMax framework.

You have access to the following tools:
{json.dumps([{"name": tool.name, "description": tool.description} for tool in self.tools.values()], indent=2)}

You can call these tools to help users with various tasks including data processing, file operations, and web searches.
Always be helpful, accurate, and provide detailed explanations of your actions.

Session ID: {self.session_id}
"""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history with careful filtering
        i = 0
        while i < len(self.conversation_history):
            msg = self.conversation_history[i]
            
            if msg.role == MessageRole.SYSTEM:
                i += 1
                continue
            
            if msg.role == MessageRole.USER:
                messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == MessageRole.ASSISTANT:
                # Check if this assistant message has tool calls
                if msg.metadata and "tool_calls" in msg.metadata:
                    # This is an assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": msg.metadata["tool_calls"]
                    })
                    
                    # Look for corresponding tool messages
                    j = i + 1
                    while j < len(self.conversation_history) and self.conversation_history[j].role == MessageRole.TOOL:
                        tool_msg = self.conversation_history[j]
                        if tool_msg.metadata and "tool_call_id" in tool_msg.metadata:
                            messages.append({
                                "role": "tool",
                                "content": tool_msg.content,
                                "tool_call_id": tool_msg.metadata["tool_call_id"]
                            })
                        j += 1
                    i = j - 1  # Skip the tool messages we just processed
                else:
                    # Regular assistant message
                    messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
            
            i += 1
        
        return messages
    
    async def process_message(self, user_message: str):
        """Process message with improved error handling"""
        try:
            # Add user message to history
            self.add_message(MessageRole.USER, user_message)
            
            # Prepare messages and tools for LLM
            messages = self._prepare_messages_for_llm()
            tools = self._prepare_tools_for_llm()
            
            # Call LLM with or without tools
            response_kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # Only add tools if they exist and tool calling is enabled
            if tools and self.config.enable_tool_calling:
                response_kwargs["tools"] = tools
                response_kwargs["tool_choice"] = "auto"
            
            response = self.llm_client.chat.completions.create(**response_kwargs)
            assistant_message = response.choices[0].message
            
            # Handle tool calls if present
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                return await self._handle_tool_calls(assistant_message)
            else:
                # No tool calls, just add the assistant response
                return self.add_message(MessageRole.ASSISTANT, assistant_message.content or "I understand your request.")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"抱歉，处理您的请求时遇到了错误。让我用其他方式来帮助您。"
            return self.add_message(MessageRole.ASSISTANT, error_response)
    
    async def _handle_tool_calls(self, assistant_message):
        """Handle tool calls separately for better error control"""
        try:
            # Convert tool calls to serializable format
            tool_calls_data = []
            for tc in assistant_message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            
            # Add assistant message with tool calls
            self.add_message(
                MessageRole.ASSISTANT,
                assistant_message.content or "我来使用工具帮助您处理这个请求。",
                metadata={"tool_calls": tool_calls_data}
            )
            
            # Execute tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if tool_name in self.tools:
                        tool_impl = self.tools[tool_name].implementation
                        if tool_impl:
                            result = tool_impl(**tool_args)
                            
                            # Add tool result message
                            self.add_message(
                                MessageRole.TOOL,
                                json.dumps(result, ensure_ascii=False),
                                metadata={"tool_call_id": tool_call.id, "tool_name": tool_name}
                            )
                except Exception as tool_error:
                    logger.error(f"Tool execution error: {tool_error}")
                    # Add error result
                    error_result = {"success": False, "error": f"工具执行错误: {str(tool_error)}"}
                    self.add_message(
                        MessageRole.TOOL,
                        json.dumps(error_result, ensure_ascii=False),
                        metadata={"tool_call_id": tool_call.id, "tool_name": tool_name}
                    )
            
            # Get final response after tool execution
            final_messages = self._prepare_messages_for_llm()
            final_response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=final_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            final_content = final_response.choices[0].message.content
            return self.add_message(MessageRole.ASSISTANT, final_content or "任务已完成。")
            
        except Exception as e:
            logger.error(f"Tool call handling error: {e}")
            error_response = "我在使用工具时遇到了一些问题，但我仍然可以为您提供帮助。请告诉我您需要什么。"
            return self.add_message(MessageRole.ASSISTANT, error_response)


def create_fixed_agent() -> FixedMCPAgent:
    """Create a fixed MCP agent with better configuration"""
    # Try different API configurations
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量")
    
    if os.getenv("OPENAI_API_KEY"):
        config = MCPAgentConfig(
            name="Fixed-OpenAI-Agent",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            enable_tool_calling=True,
            enable_memory=True,
            log_level="WARNING"
        )
    else:
        config = MCPAgentConfig(
            name="Fixed-DashScope-Agent",
            llm_provider="dashscope",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen-turbo",
            temperature=0.7,
            enable_tool_calling=False,  # Disable tool calling for DashScope to avoid issues
            enable_memory=True,
            log_level="WARNING"
        )
    
    return FixedMCPAgent(config)


def add_sample_data_tool(agent: FixedMCPAgent):
    """Add a sample data generation tool"""
    def generate_sales_data(record_count: int = 10, region: str = "全国") -> Dict[str, Any]:
        """Generate sample sales data"""
        import random
        from datetime import datetime, timedelta
        
        try:
            sales_data = []
            products = ["产品A", "产品B", "产品C", "产品D", "产品E"]
            regions = ["北京", "上海", "广州", "深圳", "杭州"] if region == "全国" else [region]
            
            for i in range(record_count):
                record = {
                    "订单ID": f"ORD{1000 + i}",
                    "产品名称": random.choice(products),
                    "销售数量": random.randint(1, 100),
                    "单价": round(random.uniform(10, 1000), 2),
                    "销售区域": random.choice(regions),
                    "销售日期": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                    "销售员": f"员工{random.randint(1, 20)}"
                }
                record["总金额"] = round(record["销售数量"] * record["单价"], 2)
                sales_data.append(record)
            
            # Calculate summary
            total_amount = sum(record["总金额"] for record in sales_data)
            total_quantity = sum(record["销售数量"] for record in sales_data)
            
            return {
                "success": True,
                "data": sales_data,
                "summary": {
                    "记录数": len(sales_data),
                    "总销售额": round(total_amount, 2),
                    "总销售数量": total_quantity,
                    "平均订单金额": round(total_amount / len(sales_data), 2),
                    "销售区域": region
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Register the tool
    sales_tool = ToolDefinition(
        name="generate_sales_data",
        description="生成示例销售数据记录",
        type=ToolType.FUNCTION,
        parameters={
            "type": "object",
            "properties": {
                "record_count": {
                    "type": "integer",
                    "description": "要生成的记录数量",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                },
                "region": {
                    "type": "string",
                    "description": "销售区域",
                    "default": "全国",
                    "enum": ["全国", "北京", "上海", "广州", "深圳", "杭州"]
                }
            }
        },
        required=[],
        implementation=generate_sales_data
    )
    
    agent.register_tool(sales_tool)


async def demo_conversation():
    """Run a demonstration conversation"""
    print("🚀 Fixed MCP Agent Demo")
    print("=" * 50)
    
    try:
        # Create agent
        agent = create_fixed_agent()
        
        # Add custom tools
        add_sample_data_tool(agent)
        
        print(f"✅ Agent created: {agent.config.name}")
        print(f"📊 Available tools: {list(agent.tools.keys())}")
        
        # Demo conversations
        demo_messages = [
            "你好！你能帮我生成一些销售数据吗？",
            "请生成10条北京地区的销售记录",
            "能分析一下这些数据的特点吗？"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"\n👤 User {i}: {message}")
            response = await agent.process_message(message)
            print(f"🤖 Agent: {response.content}")
            print("-" * 30)
        
        # Show conversation summary
        print("\n📊 Conversation Summary:")
        summary = agent.get_conversation_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n💡 请确保:")
        print("1. 设置了正确的API密钥")
        print("2. 网络连接正常")
        print("3. API配额充足")
        return None


async def interactive_chat():
    """Interactive chat mode"""
    print("\n💬 Interactive Chat Mode")
    print("输入 'quit' 或 'exit' 结束对话\n")
    
    try:
        agent = create_fixed_agent()
        add_sample_data_tool(agent)
        
        print(f"🤖 {agent.config.name} 已准备就绪！")
        print(f"📊 可用工具: {', '.join(agent.tools.keys())}\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', '退出', '再见']:
                    print("🤖 Agent: 再见！感谢使用 DataMax MCP Agent！")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Agent: ", end="", flush=True)
                response = await agent.process_message(user_input)
                print(response.content)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                print("请重试。\n")
                
    except Exception as e:
        print(f"❌ 无法启动聊天: {e}")
        print("请检查API配置。")


async def main():
    """Main function"""
    print("🔧 DataMax MCP Agent - Fixed Demo")
    print("=" * 50)
    print("这个演示解决了API兼容性问题，提供更稳定的体验。\n")
    
    # Check API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        print("⚠️  未找到API密钥")
        print("请设置以下环境变量之一:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export DASHSCOPE_API_KEY='your-dashscope-key'")
        return
    
    print("🎮 选择模式:")
    print("1. 演示对话")
    print("2. 交互聊天")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (0-2): ").strip()
            
            if choice == "0":
                print("\n👋 再见！")
                break
            elif choice == "1":
                await demo_conversation()
            elif choice == "2":
                await interactive_chat()
            else:
                print("❌ 无效选择，请重试。")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 再见！")