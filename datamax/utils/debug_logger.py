"""
Elegant debug logging system for DataMax
Provides structured, context-aware debug logging with minimal code intrusion
"""
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional
from loguru import logger


class DebugContext:
    """
    Context manager for structured debug logging
    Provides clean separation between debug and business logic
    """
    
    def __init__(self, enabled: bool = False, context_name: str = ""):
        self.enabled = enabled
        self.context_name = context_name
        self._indent_level = 0
    
    def log(self, message: str, **kwargs):
        """Log a debug message with context"""
        if not self.enabled:
            return
        
        indent = "  " * self._indent_level
        if kwargs:
            formatted_kwargs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.debug(f"{indent}[{self.context_name}] {message} | {formatted_kwargs}")
        else:
            logger.debug(f"{indent}[{self.context_name}] {message}")
    
    def log_params(self, **params):
        """Log multiple parameters in a structured way"""
        if not self.enabled:
            return
        
        self.log("Parameters:")
        self._indent_level += 1
        for key, value in params.items():
            # Handle sensitive data
            if 'key' in key.lower() and value:
                value = '***'
            # Handle length for collections
            elif hasattr(value, '__len__') and not isinstance(value, str):
                value = f"<{type(value).__name__} len={len(value)}>"
            # Truncate long strings
            elif isinstance(value, str) and len(value) > 100:
                value = f"{value[:100]}..."
            
            self.log(f"{key}: {value}")
        self._indent_level -= 1
    
    def log_data_structure(self, data: Any, name: str = "data"):
        """Log information about a data structure"""
        if not self.enabled:
            return
        
        data_type = type(data).__name__
        
        if hasattr(data, '__len__'):
            length = len(data)
            self.log(f"{name}: {data_type} with length {length}")
            
            # Log first few items for lists
            if isinstance(data, list) and length > 0:
                preview_count = min(3, length)
                self.log(f"First {preview_count} items preview:")
                self._indent_level += 1
                for i, item in enumerate(data[:preview_count]):
                    item_type = type(item).__name__
                    self.log(f"[{i}]: {item_type}")
                self._indent_level -= 1
        else:
            self.log(f"{name}: {data_type}")
    
    @contextmanager
    def section(self, section_name: str):
        """Create a nested debug section"""
        if self.enabled:
            self.log(f"=== {section_name} START ===")
            self._indent_level += 1
        
        try:
            yield self
        finally:
            if self.enabled:
                self._indent_level -= 1
                self.log(f"=== {section_name} END ===")
    
    def preview_qa_pairs(self, data: Any, max_preview: int = 10):
        """Preview QA pairs in a structured format"""
        if not self.enabled:
            return
        
        qa_pairs = self._extract_qa_pairs(data)
        
        if not qa_pairs:
            self.log("No QA pairs to preview")
            return
        
        preview_count = min(max_preview, len(qa_pairs))
        self.log(f"Previewing {preview_count} of {len(qa_pairs)} QA pairs")
        
        print("\n" + "=" * 60)
        print(f"QA PAIRS PREVIEW ({preview_count}/{len(qa_pairs)})")
        print("=" * 60)
        
        for i, qa in enumerate(qa_pairs[:preview_count], 1):
            print(f"\n--- QA Pair {i} ---")
            print(f"Question: {qa.get('instruction', qa.get('question', 'N/A'))}")
            print(f"Answer: {qa.get('output', 'N/A')}")
            print(f"Label: {qa.get('label', 'N/A')}")
        
        print("\n" + "=" * 60 + "\n")
    
    @staticmethod
    def _extract_qa_pairs(data: Any) -> list:
        """Extract QA pairs from various data structures"""
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return data
        elif isinstance(data, dict):
            if "qa_pairs" in data:
                return data["qa_pairs"]
            elif "data" in data:
                return data["data"]
            else:
                return [data]
        return []


def debug_method(context_name: str = ""):
    """
    Decorator for methods that need debug logging
    Automatically creates and manages debug context
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract debug flag from kwargs
            debug = kwargs.get('debug', False)
            
            # Create debug context
            debug_ctx = DebugContext(
                enabled=debug,
                context_name=context_name or func.__name__
            )
            
            # Inject debug context into kwargs
            kwargs['_debug_ctx'] = debug_ctx
            
            # Log method entry
            debug_ctx.log(f"Method called: {func.__name__}")
            
            try:
                result = func(self, *args, **kwargs)
                debug_ctx.log(f"Method completed successfully")
                return result
            except Exception as e:
                debug_ctx.log(f"Method failed with error: {type(e).__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator


class DebugLogger:
    """
    Static utility class for one-off debug logging
    Use when you don't need full context management
    """
    
    @staticmethod
    def log_if(condition: bool, message: str, **kwargs):
        """Log only if condition is true"""
        if condition:
            if kwargs:
                formatted = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                logger.debug(f"{message} | {formatted}")
            else:
                logger.debug(message)
    
    @staticmethod
    def log_params_if(condition: bool, **params):
        """Log parameters only if condition is true"""
        if condition:
            ctx = DebugContext(enabled=True, context_name="params")
            ctx.log_params(**params)
