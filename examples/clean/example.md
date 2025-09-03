# DataMax Clean Module Examples

## Overview

The Clean module provides comprehensive text cleaning and processing capabilities. It can remove HTML tags, abnormal characters, filter content by quality metrics, and desensitize sensitive information.

## Prerequisites

No API keys required for basic cleaning functionality.

## CLI Command Examples

### Basic Text Cleaning

Clean a text file using the default full cleaning pipeline:

```bash
datamax clean examples/clean/sample_text.txt
```

This applies complete cleaning (abnormal + filter + privacy) and saves the result as `sample_text_cleaned.txt`.

### Specific Cleaning Modes

#### Abnormal Character Cleaning Only
```bash
datamax clean examples/clean/sample_text.txt --mode abnormal
```

#### Content Quality Filtering
```bash
datamax clean examples/clean/sample_text.txt --mode filter --filter-threshold 0.5
```

#### Privacy Desensitization
```bash
datamax clean examples/clean/sample_text.txt --mode privacy
```

### Advanced Options

#### Custom Output File
```bash
datamax clean examples/clean/sample_text.txt --output cleaned_output.txt
```

#### Read from Standard Input
```bash
echo "Text with <html> tags and sensitive@email.com data" | datamax clean --stdin
```

#### Write to Standard Output
```bash
datamax clean examples/clean/sample_text.txt --stdout
```

## Python Code Examples

### Basic Usage with CleanerCLI

```python
from datamax.cli.cleaner_cli import CleanerCLI

# Initialize the cleaner
cleaner = CleanerCLI(verbose=True)

# Read and clean a file
with open('examples/clean/sample_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Perform full cleaning pipeline
result = cleaner.clean_full(text)
if 'text' in result:
    print("Cleaned text:")
    print(result['text'])
```

### Specific Cleaning Operations

```python
from datamax.cli.cleaner_cli import CleanerCLI

cleaner = CleanerCLI()

# Abnormal character cleaning
abnormal_result = cleaner.clean_abnormal(text)
print("Abnormal cleaned text:", abnormal_result.get('text', ''))

# Content filtering with custom parameters
filter_result = cleaner.clean_filter(
    text,
    filter_threshold=0.6,
    min_chars=30,
    max_chars=1000
)
print("Filtered text:", filter_result.get('text', ''))

# Privacy desensitization
privacy_result = cleaner.clean_privacy(text)
print("Privacy cleaned text:", privacy_result.get('text', ''))
```

### File-based Cleaning

```python
from datamax.cli.cleaner_cli import CleanerCLI

cleaner = CleanerCLI(verbose=True)

# Clean file and save result
output_path = cleaner.clean_file(
    input_file='examples/clean/sample_text.txt',
    output_file='examples/clean/cleaned_result.txt',
    mode='full'
)
print(f"Cleaned file saved to: {output_path}")

# Clean from stdin (simulated)
import sys
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

cleaner.clean_stdin(mode='full')

# Restore stdout
sys.stdout = old_stdout
cleaned_text = captured_output.getvalue()
print("Stdin cleaned text:", cleaned_text)
```

### Advanced Configuration

```python
from datamax.cli.cleaner_cli import CleanerCLI

cleaner = CleanerCLI()

# Full cleaning with custom parameters
result = cleaner.clean_full(
    text,
    filter_threshold=0.7,
    min_chars=50,
    max_chars=5000,
    numeric_threshold=0.5
)

print("Advanced cleaned text:")
print(result.get('text', ''))
```

## Expected Output

The cleaning process will:
1. Remove HTML tags and abnormal characters
2. Filter out low-quality content based on repetition and length
3. Desensitize sensitive information like emails, phone numbers, and IP addresses
4. Preserve clean, readable text content

## Best Practices

1. Use `full` mode for most general cleaning tasks
2. Adjust `filter_threshold` based on your content type (lower values are more strict)
3. Use `privacy` mode when dealing with user-generated content
4. Test cleaning parameters on a small sample before processing large datasets
5. Always backup original files before batch cleaning operations
