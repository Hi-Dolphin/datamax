# Debug System Documentation

## Overview

The DataMax debug system provides an elegant, architectural approach to debugging that separates debug concerns from business logic. It offers structured, context-aware logging with minimal code intrusion.

## Architecture

The debug system is built on three main components:

### 1. DebugContext Class

A context manager that provides structured debug logging with hierarchical sections and automatic indentation.

**Key Features:**
- Context-aware logging with automatic indentation
- Structured parameter logging with sensitive data protection
- Data structure inspection
- Nested debug sections using context managers
- QA pairs preview functionality

### 2. debug_method Decorator

A decorator that automatically creates and manages debug context for methods.

**Key Features:**
- Automatic debug context creation
- Method entry/exit logging
- Exception tracking
- Zero boilerplate in method implementation

### 3. DebugLogger Utility Class

Static utility methods for one-off debug logging without full context management.

## Usage Examples

### Basic Usage with DebugContext

```python
from datamax.utils.debug_logger import DebugContext

def process_data(data, debug=False):
    # Create debug context
    dbg = DebugContext(enabled=debug, context_name="process_data")
    
    # Log parameters
    dbg.log_params(
        data_length=len(data),
        data_type=type(data).__name__
    )
    
    # Use sections for logical grouping
    with dbg.section("Data Processing"):
        dbg.log("Starting data processing")
        # ... processing logic ...
        dbg.log("Processing complete")
    
    return result
```

### Using Nested Sections

```python
def complex_operation(debug=False):
    dbg = DebugContext(enabled=debug, context_name="complex_operation")
    
    with dbg.section("Phase 1: Initialization"):
        dbg.log("Initializing resources")
        # ... initialization code ...
    
    with dbg.section("Phase 2: Processing"):
        dbg.log("Processing data")
        with dbg.section("Sub-task: Validation"):
            dbg.log("Validating input")
            # ... validation code ...
    
    with dbg.section("Phase 3: Cleanup"):
        dbg.log("Cleaning up resources")
        # ... cleanup code ...
```

### Logging Data Structures

```python
def analyze_data(data, debug=False):
    dbg = DebugContext(enabled=debug, context_name="analyze_data")
    
    # Automatically logs type, length, and preview
    dbg.log_data_structure(data, "input_data")
    
    # Process data...
    result = process(data)
    
    dbg.log_data_structure(result, "output_data")
    return result
```

### Preview QA Pairs

```python
def generate_qa_pairs(content, debug=False):
    dbg = DebugContext(enabled=debug, context_name="generate_qa_pairs")
    
    # Generate QA pairs...
    qa_pairs = generate(content)
    
    # Preview first 10 QA pairs with formatted output
    dbg.preview_qa_pairs(qa_pairs, max_preview=10)
    
    return qa_pairs
```

### Using the debug_method Decorator

```python
from datamax.utils.debug_logger import debug_method

class DataProcessor:
    @debug_method(context_name="process")
    def process(self, data, debug=False, **kwargs):
        # Debug context is automatically created and injected as _debug_ctx
        dbg = kwargs.get('_debug_ctx')
        
        dbg.log("Processing started")
        # ... processing logic ...
        dbg.log("Processing completed")
        
        return result
```

### One-off Debug Logging

```python
from datamax.utils.debug_logger import DebugLogger

def quick_function(data, debug=False):
    # Simple conditional logging
    DebugLogger.log_if(debug, "Processing data", count=len(data))
    
    # Process data...
    
    DebugLogger.log_if(debug, "Processing complete")
```

## Best Practices

### 1. Use Descriptive Context Names

```python
# Good
dbg = DebugContext(enabled=debug, context_name="get_pre_label")

# Bad
dbg = DebugContext(enabled=debug, context_name="func")
```

### 2. Group Related Operations in Sections

```python
with dbg.section("Content Preparation"):
    dbg.log("Fetching content")
    content = fetch_content()
    dbg.log_data_structure(content, "fetched_content")

with dbg.section("Content Processing"):
    dbg.log("Processing content")
    result = process_content(content)
```

### 3. Log Parameters at Method Entry

```python
def process_data(data, chunk_size=500, debug=False):
    dbg = DebugContext(enabled=debug, context_name="process_data")
    
    # Log all relevant parameters upfront
    dbg.log_params(
        data_length=len(data),
        chunk_size=chunk_size,
        data_type=type(data).__name__
    )
    
    # ... rest of the method ...
```

### 4. Use log_data_structure for Complex Data

```python
# Instead of manually logging details
dbg.log(f"Data type: {type(data)}, length: {len(data)}")

# Use the built-in method
dbg.log_data_structure(data, "data")
```

### 5. Protect Sensitive Information

The debug system automatically masks sensitive data:

```python
dbg.log_params(
    api_key=api_key,  # Automatically masked as '***'
    username=username,
    password=password  # Automatically masked as '***'
)
```

## Features

### Automatic Sensitive Data Protection

Parameters containing 'key', 'password', 'token', 'secret' are automatically masked:

```python
dbg.log_params(
    api_key="sk-1234567890",  # Logged as: api_key: ***
    base_url="https://api.example.com"  # Logged normally
)
```

### Smart Data Structure Logging

Automatically handles different data types:

```python
# Lists: Shows type, length, and preview of first 3 items
dbg.log_data_structure([1, 2, 3, 4, 5], "numbers")
# Output: numbers: list with length 5
#         First 3 items preview:
#           [0]: int
#           [1]: int
#           [2]: int

# Dicts: Shows type and length
dbg.log_data_structure({"key": "value"}, "config")
# Output: config: dict with length 1

# Strings: Shows type and length
dbg.log_data_structure("Hello World", "message")
# Output: message: str with length 11
```

### Hierarchical Logging with Indentation

```python
dbg.log("Level 0")
with dbg.section("Section 1"):
    dbg.log("Level 1")
    with dbg.section("Section 2"):
        dbg.log("Level 2")
```

Output:
```
[get_pre_label] Level 0
[get_pre_label] === Section 1 START ===
  [get_pre_label] Level 1
  [get_pre_label] === Section 2 START ===
    [get_pre_label] Level 2
  [get_pre_label] === Section 2 END ===
[get_pre_label] === Section 1 END ===
```

### QA Pairs Preview

Formatted preview of QA pairs with clear visual separation:

```python
dbg.preview_qa_pairs(qa_data, max_preview=10)
```

Output:
```
============================================================
QA PAIRS PREVIEW (10/100)
============================================================

--- QA Pair 1 ---
Question: What is the capital of France?
Answer: Paris
Label: Geography

--- QA Pair 2 ---
Question: What is 2 + 2?
Answer: 4
Label: Mathematics

============================================================
```

## Performance Considerations

The debug system is designed for zero overhead when disabled:

```python
dbg = DebugContext(enabled=False, context_name="operation")

# These calls return immediately without any processing
dbg.log("This won't be logged")
dbg.log_params(param1=value1, param2=value2)
dbg.log_data_structure(data, "data")

with dbg.section("Section"):
    # Section context manager is a no-op when disabled
    pass
```

## Migration Guide

### Before (Old Debug Style)

```python
def get_pre_label(self, debug=False, **kwargs):
    if debug:
        logger.debug(f"get_pre_label called with parameters:")
        logger.debug(f"  content: {content is not None}")
        logger.debug(f"  api_key: {'***' if api_key else None}")
        logger.debug(f"  base_url: {base_url}")
    
    if content is not None:
        text = content
        if debug:
            logger.debug(f"Using external content, length: {len(text)}")
    else:
        if debug:
            logger.debug("Fetching content")
        processed = self.get_data()
        if debug:
            logger.debug(f"Got data: {type(processed)}")
```

### After (New Debug Style)

```python
def get_pre_label(self, debug=False, **kwargs):
    dbg = DebugContext(enabled=debug, context_name="get_pre_label")
    
    dbg.log_params(
        content_provided=content is not None,
        api_key='***' if api_key else None,
        base_url=base_url
    )
    
    with dbg.section("Content Preparation"):
        if content is not None:
            text = content
            dbg.log(f"Using external content, length: {len(text)}")
        else:
            dbg.log("Fetching content")
            processed = self.get_data()
            dbg.log_data_structure(processed, "processed_data")
```

## Benefits

1. **Separation of Concerns**: Debug logic is cleanly separated from business logic
2. **Readability**: Code is more readable without scattered `if debug:` checks
3. **Maintainability**: Debug code is easier to maintain and extend
4. **Consistency**: Standardized debug output format across the codebase
5. **Zero Overhead**: No performance impact when debug is disabled
6. **Type Safety**: Better IDE support and type checking
7. **Structured Output**: Hierarchical, indented output for better readability

## API Reference

### DebugContext

#### Constructor
```python
DebugContext(enabled: bool = False, context_name: str = "")
```

#### Methods

- `log(message: str, **kwargs)`: Log a debug message
- `log_params(**params)`: Log multiple parameters in structured format
- `log_data_structure(data: Any, name: str = "data")`: Log information about a data structure
- `section(section_name: str)`: Context manager for nested debug sections
- `preview_qa_pairs(data: Any, max_preview: int = 10)`: Preview QA pairs

### debug_method Decorator

```python
@debug_method(context_name: str = "")
def method(self, *args, debug=False, **kwargs):
    # Access debug context via kwargs['_debug_ctx']
    pass
```

### DebugLogger

#### Static Methods

- `log_if(condition: bool, message: str, **kwargs)`: Conditional logging
- `log_params_if(condition: bool, **params)`: Conditional parameter logging

## Conclusion

The new debug system provides a clean, maintainable, and efficient way to add debugging capabilities to your code. By using context managers and structured logging, it keeps your code clean while providing powerful debugging features when needed.
