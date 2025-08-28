#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Agent Generator - A comprehensive example of MCP (Model Context Protocol) + LLM integration

This module demonstrates how to create an intelligent agent that combines:
1. MCP (Model Context Protocol) for structured communication
2. LLM (Large Language Model) for natural language processing
3. Tool calling capabilities for enhanced functionality

Author: DataMax Team
Date: 2024
License: MIT
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import uuid

import openai
from pydantic import BaseModel, Field
from loguru import logger


class MessageRole(str, Enum):
    """Message roles in MCP protocol"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolType(str, Enum):
    """Available tool types"""
    FUNCTION = "function"
    RETRIEVAL = "retrieval"
    CODE_INTERPRETER = "code_interpreter"
    WEB_SEARCH = "web_search"
    FILE_OPERATION = "file_operation"


@dataclass
class MCPMessage:
    """MCP Protocol Message Structure"""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['role'] = MessageRole(data['role'])
        return cls(**data)


class ToolDefinition(BaseModel):
    """Tool definition for MCP agent"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    type: ToolType = Field(..., description="Tool type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")
    required: List[str] = Field(default_factory=list, description="Required parameters")
    implementation: Optional[Callable] = Field(None, description="Tool implementation function")

    class Config:
        arbitrary_types_allowed = True


class MCPAgentConfig(BaseModel):
    """Configuration for MCP Agent"""
    name: str = Field("DataMax-MCP-Agent", description="Agent name")
    version: str = Field("1.0.0", description="Agent version")
    description: str = Field("Advanced MCP Agent with LLM integration", description="Agent description")
    
    # LLM Configuration
    llm_provider: str = Field("openai", description="LLM provider (openai, dashscope, etc.)")
    api_key: str = Field(..., description="API key for LLM service")
    base_url: Optional[str] = Field(None, description="Custom base URL for LLM API")
    model_name: str = Field("gpt-3.5-turbo", description="LLM model name")
    temperature: float = Field(0.7, description="LLM temperature")
    max_tokens: int = Field(2000, description="Maximum tokens for LLM response")
    
    # MCP Configuration
    max_conversation_length: int = Field(50, description="Maximum conversation history length")
    enable_tool_calling: bool = Field(True, description="Enable tool calling capabilities")
    enable_memory: bool = Field(True, description="Enable conversation memory")
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field(None, description="Log file path")


class MCPAgent:
    """MCP Agent with LLM integration"""
    
    def __init__(self, config: MCPAgentConfig):
        self.config = config
        self.conversation_history: List[MCPMessage] = []
        self.tools: Dict[str, ToolDefinition] = {}
        self.session_id = str(uuid.uuid4())
        
        # Setup logging
        self._setup_logging()
        
        # Initialize LLM client
        self._setup_llm_client()
        
        # Register default tools
        self._register_default_tools()
        
        logger.info(f"MCP Agent '{self.config.name}' initialized with session ID: {self.session_id}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            self.config.log_file or "mcp_agent.log",
            level=self.config.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="10 MB"
        )
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
        )
    
    def _setup_llm_client(self):
        """Setup LLM client based on provider"""
        if self.config.llm_provider.lower() == "openai":
            self.llm_client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        else:
            # For other providers, use OpenAI-compatible interface
            self.llm_client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.openai.com/v1"
            )
    
    def _register_default_tools(self):
        """Register default tools for the agent"""
        # Data processing tool
        self.register_tool(ToolDefinition(
            name="process_data",
            description="Process and analyze data using DataMax capabilities",
            type=ToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Data to process"},
                    "operation": {"type": "string", "description": "Operation to perform", "enum": ["clean", "parse", "analyze"]}
                }
            },
            required=["data", "operation"],
            implementation=self._process_data_tool
        ))
        
        # File operation tool
        self.register_tool(ToolDefinition(
            name="file_operation",
            description="Perform file operations like read, write, or analyze files",
            type=ToolType.FILE_OPERATION,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "operation": {"type": "string", "description": "Operation to perform", "enum": ["read", "write", "analyze"]},
                    "content": {"type": "string", "description": "Content to write (for write operation)"}
                }
            },
            required=["file_path", "operation"],
            implementation=self._file_operation_tool
        ))
        
        # Web search tool
        self.register_tool(ToolDefinition(
            name="web_search",
            description="Search the web for information",
            type=ToolType.WEB_SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results", "default": 5}
                }
            },
            required=["query"],
            implementation=self._web_search_tool
        ))
    
    def register_tool(self, tool: ToolDefinition):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.type})")
    
    def _process_data_tool(self, data: str, operation: str) -> Dict[str, Any]:
        """Implementation of data processing tool"""
        try:
            if operation == "clean":
                # Simulate data cleaning
                cleaned_data = data.strip().replace("\n\n", "\n")
                return {"success": True, "result": cleaned_data, "operation": "clean"}
            elif operation == "parse":
                # Simulate data parsing
                lines = data.split("\n")
                return {"success": True, "result": {"lines": len(lines), "words": len(data.split())}, "operation": "parse"}
            elif operation == "analyze":
                # Simulate data analysis
                analysis = {
                    "length": len(data),
                    "word_count": len(data.split()),
                    "line_count": len(data.split("\n")),
                    "has_numbers": any(char.isdigit() for char in data)
                }
                return {"success": True, "result": analysis, "operation": "analyze"}
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _file_operation_tool(self, file_path: str, operation: str, content: str = None) -> Dict[str, Any]:
        """Implementation of file operation tool"""
        try:
            if operation == "read":
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                return {"success": True, "result": file_content, "operation": "read"}
            elif operation == "write":
                if content is None:
                    return {"success": False, "error": "Content is required for write operation"}
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"success": True, "result": f"Written {len(content)} characters to {file_path}", "operation": "write"}
            elif operation == "analyze":
                import os
                if not os.path.exists(file_path):
                    return {"success": False, "error": "File does not exist"}
                stat = os.stat(file_path)
                analysis = {
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": os.path.splitext(file_path)[1]
                }
                return {"success": True, "result": analysis, "operation": "analyze"}
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _web_search_tool(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Implementation of web search tool (mock implementation)"""
        # This is a mock implementation. In a real scenario, you would integrate with a search API
        mock_results = [
            {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/result{i+1}", "snippet": f"This is a mock search result {i+1} for the query '{query}'"}
            for i in range(min(max_results, 3))
        ]
        return {"success": True, "result": mock_results, "query": query}
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None) -> MCPMessage:
        """Add a message to conversation history"""
        message = MCPMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        
        # Trim conversation history if too long
        if len(self.conversation_history) > self.config.max_conversation_length:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_length:]
        
        logger.debug(f"Added message: {role} - {content[:100]}...")
        return message
    
    def _prepare_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Prepare conversation history for LLM API"""
        messages = []
        
        # Add system message
        system_prompt = f"""
You are {self.config.name}, an advanced AI assistant powered by the DataMax framework.

You have access to the following tools:
{json.dumps([{"name": tool.name, "description": tool.description, "type": tool.type} for tool in self.tools.values()], indent=2)}

You can call these tools to help users with various tasks including data processing, file operations, and web searches.
Always be helpful, accurate, and provide detailed explanations of your actions.

Session ID: {self.session_id}
Timestamp: {datetime.now().isoformat()}
"""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history with proper formatting
        for msg in self.conversation_history:
            if msg.role != MessageRole.SYSTEM:  # Skip system messages from history
                if msg.role == MessageRole.TOOL:
                    # Tool messages need special formatting
                    messages.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.metadata.get("tool_call_id") if msg.metadata else None
                    })
                elif msg.role == MessageRole.ASSISTANT and msg.metadata and "tool_calls" in msg.metadata:
                    # Assistant messages with tool calls need special formatting
                    tool_calls = msg.metadata["tool_calls"]
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": tool_calls
                    })
                else:
                    # Regular messages
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
        
        return messages
    
    def _prepare_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Prepare tools for LLM function calling"""
        if not self.config.enable_tool_calling:
            return []
        
        tools = []
        for tool in self.tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        
        return tools
    
    def _save_request_body(self, request_kwargs: Dict[str, Any], request_type: str = "initial"):
        """Save complete request body to file for debugging"""
        try:
            import os
            from datetime import datetime
            
            # Create logs directory if it doesn't exist
            logs_dir = "mcp_request_logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            # Generate filename with timestamp and type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{logs_dir}/request_{request_type}_{timestamp}.json"
            
            # Prepare request data for saving
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "agent_name": self.config.name,
                "request_type": request_type,
                "request_body": request_kwargs
            }
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Request body ({request_type}) saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save request body: {e}")
    
    async def process_message(self, user_message: str) -> MCPMessage:
        """Process a user message and generate response"""
        try:
            # Add user message to history
            self.add_message(MessageRole.USER, user_message)
            
            # Prepare messages and tools for LLM
            messages = self._prepare_messages_for_llm()
            tools = self._prepare_tools_for_llm()
            
            # Call LLM
            response_kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            if tools:
                response_kwargs["tools"] = tools
                response_kwargs["tool_choice"] = "auto"
            
            # Save complete request body to file
            self._save_request_body(response_kwargs)
            
            response = self.llm_client.chat.completions.create(**response_kwargs)
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls
            if assistant_message.tool_calls:
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
                tool_call_message = self.add_message(
                    MessageRole.ASSISTANT,
                    assistant_message.content or "I'll help you with that using the available tools.",
                    metadata={"tool_calls": tool_calls_data}
                )
                
                # Execute tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if tool_name in self.tools:
                        tool_impl = self.tools[tool_name].implementation
                        if tool_impl:
                            result = tool_impl(**tool_args)
                            tool_results.append({
                                "tool_call_id": tool_call.id,
                                "result": result
                            })
                            
                            # Add tool result message
                            self.add_message(
                                MessageRole.TOOL,
                                json.dumps(result),
                                metadata={"tool_call_id": tool_call.id, "tool_name": tool_name}
                            )
                
                # Get final response after tool execution
                final_messages = self._prepare_messages_for_llm()
                final_request_kwargs = {
                    "model": self.config.model_name,
                    "messages": final_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                
                # Save final request body to file
                self._save_request_body(final_request_kwargs, "final_response")
                
                final_response = self.llm_client.chat.completions.create(**final_request_kwargs)
                
                final_content = final_response.choices[0].message.content
                return self.add_message(MessageRole.ASSISTANT, final_content)
            
            else:
                # No tool calls, just add the assistant response
                return self.add_message(MessageRole.ASSISTANT, assistant_message.content)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            return self.add_message(MessageRole.ASSISTANT, error_response)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            "session_id": self.session_id,
            "agent_name": self.config.name,
            "message_count": len(self.conversation_history),
            "tools_available": list(self.tools.keys()),
            "last_message_time": self.conversation_history[-1].timestamp.isoformat() if self.conversation_history else None
        }
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation history"""
        if format.lower() == "json":
            return json.dumps([msg.to_dict() for msg in self.conversation_history], indent=2, ensure_ascii=False)
        elif format.lower() == "markdown":
            md_content = f"# Conversation History - {self.config.name}\n\n"
            md_content += f"**Session ID:** {self.session_id}\n\n"
            
            for msg in self.conversation_history:
                md_content += f"## {msg.role.value.title()} ({msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
                md_content += f"{msg.content}\n\n"
                if msg.metadata:
                    md_content += f"*Metadata: {json.dumps(msg.metadata)}*\n\n"
            
            return md_content
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage and demo functions
def create_demo_agent() -> MCPAgent:
    """Create a demo MCP agent with sample configuration"""
    config = MCPAgentConfig(
        name="DataMax-Demo-Agent",
        api_key="your-api-key-here",  # Replace with actual API key
        base_url="https://api.openai.com/v1",  # Or your preferred LLM provider
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        enable_tool_calling=True,
        enable_memory=True
    )
    
    return MCPAgent(config)


async def demo_conversation():
    """Demonstrate MCP agent capabilities"""
    print("🚀 DataMax MCP Agent Demo")
    print("=" * 50)
    
    # Create agent
    agent = create_demo_agent()
    
    # Demo conversations
    demo_messages = [
        "Hello! Can you help me analyze some data?",
        "I have a text file with some data. Can you help me process it?",
        "Can you search for information about 'machine learning best practices'?",
        "What tools do you have available?"
    ]
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n👤 User {i}: {message}")
        response = await agent.process_message(message)
        print(f"🤖 Agent: {response.content}")
        print("-" * 30)
    
    # Show conversation summary
    print("\n📊 Conversation Summary:")
    summary = agent.get_conversation_summary()
    print(json.dumps(summary, indent=2))
    
    # Export conversation
    print("\n📄 Exporting conversation...")
    conversation_json = agent.export_conversation("json")
    with open("demo_conversation.json", "w", encoding="utf-8") as f:
        f.write(conversation_json)
    print("Conversation exported to demo_conversation.json")


if __name__ == "__main__":
    # Run the demo
    print("DataMax MCP Agent Generator - Example Implementation")
    print("This is a comprehensive example of MCP + LLM integration.")
    print("\nTo run the demo, make sure you have:")
    print("1. OpenAI API key (or compatible LLM provider)")
    print("2. Required dependencies installed")
    print("\nThen run: python -m asyncio datamax.generator.mcp_agent_generator.demo_conversation()")
    
    # You can also run the demo directly:
    # asyncio.run(demo_conversation())