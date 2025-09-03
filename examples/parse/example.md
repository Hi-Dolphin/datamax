# DataMax Parse Module Examples

## Overview

The Parse module provides comprehensive document parsing capabilities supporting multiple file formats including PDF, DOCX, images, and more. It can extract text content, perform OCR on images, and integrate with vision models for advanced content analysis.

## Prerequisites

For advanced features:
- OpenAI API key for MLLM functionality (set as `OPENAI_API_KEY`)
- Basic parsing doesn't require API keys

## CLI Command Examples

### Basic Document Parsing

Parse a single document with automatic format detection:

```bash
datamax parser parse examples/parse/sample_document.txt
```

This will parse the document and save the result as `sample_document_parsed.md`.

### Parse with Specific Output Format

#### Markdown Output
```bash
datamax parser parse examples/parse/sample_document.txt --format markdown
```

#### JSON Output
```bash
datamax parser parse examples/parse/sample_document.txt --format json
```

#### Text Output
```bash
datamax parser parse examples/parse/sample_document.txt --format text
```

### Advanced PDF Parsing

#### Use MinerU for Enhanced PDF Parsing
```bash
datamax parser parse document.pdf --use-mineru
```

#### Use Qwen-VL OCR for PDF
```bash
datamax parser parse document.pdf --use-qwen-vl-ocr
```

### Image Processing with Vision Models

#### Basic Image Parsing with MLLM
```bash
export OPENAI_API_KEY="your-api-key-here"
datamax parser parse image.jpg --use-mllm
```

#### Custom MLLM Prompt
```bash
datamax parser parse image.jpg --use-mllm \
  --mllm-prompt "Describe this image in detail, including any text visible"
```

#### Custom Model and API Settings
```bash
datamax parser parse image.jpg --use-mllm \
  --api-key your-openai-key \
  --model gpt-4-vision-preview \
  --base-url https://api.openai.com/v1
```

### Batch Processing

Process multiple files in a directory:

```bash
datamax parser batch examples/parse ./parsed_output
```

#### Recursive Batch Processing
```bash
datamax parser batch examples/parse ./parsed_output --recursive
```

#### Filter by File Pattern
```bash
datamax parser batch examples/parse ./parsed_output --pattern "*.txt"
```

#### Advanced Batch with MLLM
```bash
datamax parser batch examples/parse ./parsed_output \
  --use-mllm \
  --recursive \
  --max-workers 4
```

### Office Document Processing

#### Word Document
```bash
datamax parser parse document.docx --to-markdown
```

#### PowerPoint Presentation
```bash
datamax parser parse presentation.pptx
```

#### Excel Spreadsheet
```bash
datamax parser parse spreadsheet.xlsx
```

## Python Code Examples

### Basic File Parsing

```python
from datamax.cli.parser_cli import ParserCLI

# Initialize the parser
parser = ParserCLI(verbose=True)

# Parse a single file
result = parser.parse_file(
    input_file='examples/parse/sample_document.txt',
    output_file='examples/parse/parsed_result.md',
    format='markdown',
    domain='Technology'
)

print("Parsing completed successfully!")
print(f"Result saved to: examples/parse/parsed_result.md")
```

### Parse with Different Output Formats

```python
from datamax.cli.parser_cli import ParserCLI

parser = ParserCLI(verbose=True)

# Parse to different formats
formats = ['markdown', 'json', 'text']

for fmt in formats:
    output_file = f'examples/parse/sample_document_parsed.{fmt}'
    result = parser.parse_file(
        input_file='examples/parse/sample_document.txt',
        output_file=output_file,
        format=fmt
    )
    print(f"Parsed to {fmt}: {output_file}")
```

### Advanced PDF Processing

```python
from datamax.cli.parser_cli import ParserCLI

parser = ParserCLI(verbose=True)

# Parse PDF with advanced options
result = parser.parse_file(
    input_file='document.pdf',
    output_file='document_parsed.md',
    format='markdown',
    use_mineru=True,  # Enhanced PDF parsing
    use_qwen_vl_ocr=True,  # Better OCR
    domain='Technology'
)

print("Advanced PDF parsing completed!")
```

### Image Processing with MLLM

```python
import os
from datamax.cli.parser_cli import ParserCLI

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

parser = ParserCLI(verbose=True)

# Parse image with vision model
result = parser.parse_file(
    input_file='image.jpg',
    output_file='image_description.md',
    format='markdown',
    use_mllm=True,
    mllm_system_prompt='Describe this image in detail, including any visible text and objects.',
    model_name='gpt-4-vision-preview'
)

print("Image parsing with MLLM completed!")
```

### Batch Processing

```python
from datamax.cli.parser_cli import ParserCLI
import os

parser = ParserCLI(verbose=True)

# Batch process all files in a directory
results = parser.parse_batch(
    input_dir='examples/parse',
    output_dir='examples/parse/batch_output',
    format='markdown',
    recursive=True,
    max_workers=4,
    continue_on_error=True,
    use_mllm=True  # Enable MLLM for images
)

# Summarize results
successful = len([r for r in results if r.get('success', False)])
failed = len(results) - successful

print("Batch processing completed!")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
```

### Custom MLLM Configuration

```python
import os
from datamax.cli.parser_cli import ParserCLI

# Set API configuration
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'

parser = ParserCLI(verbose=True)

# Parse with custom MLLM settings
result = parser.parse_file(
    input_file='diagram.png',
    output_file='diagram_analysis.md',
    use_mllm=True,
    mllm_system_prompt='Analyze this technical diagram and explain its components and relationships.',
    api_key='your-custom-api-key',  # Override environment variable
    base_url='https://custom-api-endpoint.com/v1',
    model_name='gpt-4-vision-preview'
)

print("Custom MLLM parsing completed!")
```

### Processing Different File Types

```python
from datamax.cli.parser_cli import ParserCLI

parser = ParserCLI(verbose=True)

# Define different file types to process
file_types = [
    'document.pdf',
    'presentation.pptx',
    'spreadsheet.xlsx',
    'image.jpg',
    'text_file.txt'
]

for file_path in file_types:
    if os.path.exists(file_path):
        try:
            result = parser.parse_file(
                input_file=file_path,
                output_file=f'parsed_{os.path.basename(file_path)}.md',
                format='markdown'
            )
            print(f"Successfully parsed: {file_path}")
        except Exception as e:
            print(f"Failed to parse {file_path}: {str(e)}")
    else:
        print(f"File not found: {file_path}")
```

### Advanced Batch Processing with Filtering

```python
from datamax.cli.parser_cli import ParserCLI
import glob

parser = ParserCLI(verbose=True)

# Process only PDF files in nested directories
results = parser.parse_batch(
    input_dir='examples/parse',
    output_dir='examples/parse/pdf_output',
    format='markdown',
    pattern='**/*.pdf',  # Recursive pattern
    recursive=True,
    max_workers=8,
    use_mineru=True,  # Enhanced PDF parsing
    use_qwen_vl_ocr=True  # Better OCR
)

print("PDF batch processing completed!")
print(f"Processed {len(results)} PDF files")
```

## Expected Output

### Markdown Output Format
```markdown
# DataMax Parser Example Document

## Overview

This is a sample document for testing DataMax parser functionality. The parser supports various file formats including PDF, DOCX, images, and more.

## Features Demonstrated

### Text Parsing
- Plain text extraction from documents
- Format preservation
- Character encoding handling

### Image Processing
- OCR (Optical Character Recognition)
- Vision model integration
- Multi-modal content analysis
```

### JSON Output Format
```json
{
  "content": "# DataMax Parser Example Document\n\n## Overview\n\nThis is a sample document...",
  "metadata": {
    "file_type": "text",
    "domain": "Technology",
    "word_count": 245,
    "character_count": 1456,
    "encoding": "utf-8"
  }
}
```

### Text Output Format
```
DataMax Parser Example Document

Overview

This is a sample document for testing DataMax parser functionality...

Features Demonstrated

Text Parsing
- Plain text extraction from documents
- Format preservation
- Character encoding handling
```

### Image MLLM Analysis Result
```markdown
# Image Analysis: diagram.png

## Visual Description
This appears to be a technical diagram showing a neural network architecture with multiple layers.

## Key Components Identified
1. **Input Layer**: Accepts initial data input
2. **Hidden Layers**: Three intermediate processing layers with neurons
3. **Output Layer**: Final classification/prediction layer
4. **Connections**: Arrows showing data flow between layers

## Technical Details
- Network has 4 layers total (1 input, 2 hidden, 1 output)
- Each layer contains multiple neurons (represented as circles)
- Arrows indicate weighted connections between neurons
- This appears to be a feed-forward neural network design
```

## Best Practices

1. **Format Selection**: Use Markdown for structured documents, JSON for programmatic processing, Text for simple extraction
2. **API Keys**: Store OpenAI API keys as environment variables, never hardcode them
3. **MLLM Usage**: Only enable MLLM for images/diagrams that require visual analysis
4. **Batch Processing**: Use multi-worker processing for large file collections
5. **Error Handling**: Enable `continue_on_error=True` for batch processing to handle mixed file types
6. **Domain Context**: Specify appropriate domain context for better parsing results
7. **Output Locations**: Organize output directories by file type or processing date
8. **Resource Management**: Monitor API usage when processing large numbers of images
9. **Quality Validation**: Review parsed output for accuracy, especially with complex layouts
10. **Incremental Processing**: Use file patterns to process only new or specific file types

## Supported File Formats

| Format | Supported | Notes |
|--------|-----------|-------|
| PDF | ✅ | MinerU recommended for complex layouts |
| DOCX | ✅ | Full formatting preservation |
| XLSX/XLS | ✅ | Table structure maintained |
| PPTX/PPT | ✅ | Slide content extracted |
| TXT | ✅ | Character encoding handling |
| MD | ✅ | Markdown structure preserved |
| HTML | ✅ | Clean text extraction |
| Images | ✅ | OCR and MLLM support |
| EPUB | ✅ | E-book content extraction |
| Code files | ✅ | Syntax highlighting preserved |
