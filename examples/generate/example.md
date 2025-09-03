# DataMax Generate Module Examples

## Overview

The Generate module provides QA pair generation capabilities for text documents. It supports both regular QA generation and multimodal QA generation for documents containing images.

## Prerequisites

For multimodal QA generation:
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- For basic QA generation, no API keys are required

## CLI Command Examples

### Basic QA Generation

Generate QA pairs from a markdown document:

```bash
datamax generator qa examples/generate/sample_document.md
```

This will generate 5 QA pairs per text chunk and save the result as `sample_document_qa.json`.

### With Custom Parameters

#### Custom Question Number and Chunk Size
```bash
datamax generator qa examples/generate/sample_document.md \
  --question-number 3 \
  --chunk-size 300 \
  --chunk-overlap 50
```

#### Custom Output Location
```bash
datamax generator qa examples/generate/sample_document.md \
  --output my_qa_pairs.json
```

#### Custom Model and API Settings
```bash
datamax generator qa examples/generate/sample_document.md \
  --api-key your-openai-api-key \
  --base-url https://api.openai.com/v1 \
  --model gpt-4
```

### Multimodal QA Generation

Generate QA pairs from documents with images (requires OpenAI API key):

```bash
export OPENAI_API_KEY="your-api-key-here"
datamax generator multimodal examples/generate/sample_document.md
```

#### Custom Model for Multimodal
```bash
datamax generator multimodal examples/generate/sample_document.md \
  --model gpt-4-vision-preview \
  --question-number 2
```

### Advanced Options

#### Batch Processing with Multiple Workers
```bash
datamax generator qa examples/generate/sample_document.md \
  --max-workers 8
```

#### List Available Generators
```bash
datamax generator list
```

## Python Code Examples

### Basic QA Generation

```python
from datamax.cli.generator_cli import GeneratorCLI

# Initialize the generator
generator = GeneratorCLI(verbose=True)

# Generate QA pairs from a document
result = generator.generate_qa(
    input_file='examples/generate/sample_document.md',
    output_file='examples/generate/qa_output.json',
    question_number=5,
    chunk_size=500,
    chunk_overlap=100
)

print("QA generation completed!")
print(f"Generated {len(result['qa_pairs'])} QA pairs")

# Access the generated QA pairs
for i, qa in enumerate(result['qa_pairs'][:3]):
    print(f"Q{i+1}: {qa['question']}")
    print(f"A{i+1}: {qa['answer']}")
    print("---")
```

### Custom Configuration

```python
from datamax.cli.generator_cli import GeneratorCLI

generator = GeneratorCLI(verbose=True)

# Generate with custom API settings
result = generator.generate_qa(
    input_file='examples/generate/sample_document.md',
    output_file='examples/generate/custom_qa.json',
    api_key='your-openai-api-key',
    base_url='https://api.openai.com/v1',
    model='gpt-4',
    question_number=3,
    chunk_size=300,
    chunk_overlap=50,
    max_workers=4
)

print("Custom QA generation result:")
print(f"Total QA pairs: {len(result['qa_pairs'])}")
```

### Multimodal QA Generation

```python
import os
from datamax.cli.generator_cli import GeneratorCLI

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

generator = GeneratorCLI(verbose=True)

# Generate multimodal QA pairs (requires document with images)
result = generator.generate_multimodal_qa(
    input_file='examples/generate/sample_document.md',
    output_file='examples/generate/multimodal_qa.json',
    model='gpt-4-vision-preview',
    question_number=2,
    chunk_size=2000,
    chunk_overlap=300,
    max_workers=2
)

print("Multimodal QA generation completed!")
print(f"Generated {len(result)} multimodal QA pairs")
```

### Advanced Usage with Direct Classes

```python
from datamax.generator.qa_generator import QAGenerator
from datamax.generator.multimodal_qa_generator import MultimodalQAGenerator

# Basic QA generation
qa_generator = QAGenerator(
    model_name='gpt-4',
    api_key='your-api-key',
    base_url='https://api.openai.com/v1'
)

# Read document content
with open('examples/generate/sample_document.md', 'r', encoding='utf-8') as f:
    document_content = f.read()

# Generate QA pairs
qa_pairs = qa_generator.generate_qa_pairs(
    text=document_content,
    question_number=5,
    chunk_size=500,
    chunk_overlap=100
)

print(f"Generated {len(qa_pairs)} QA pairs using direct class")
```

### Batch Processing

```python
from datamax.cli.generator_cli import GeneratorCLI
import glob
from pathlib import Path

generator = GeneratorCLI(verbose=True)

# Find all markdown files in a directory
input_files = glob.glob('examples/generate/*.md')

for input_file in input_files:
    # Generate output filename
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_qa.json"

    try:
        result = generator.generate_qa(
            input_file=input_file,
            output_file=str(output_file),
            question_number=5,
            chunk_size=500
        )

        print(f"Processed {input_file}: {len(result['qa_pairs'])} QA pairs")

    except Exception as e:
        print(f"Failed to process {input_file}: {str(e)}")

print("Batch processing completed!")
```

### Using Environment Variables

```python
import os
from datamax.cli.generator_cli import GeneratorCLI

# Set environment variables for API configuration
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'

generator = GeneratorCLI(verbose=True)

# Generate QA without explicitly passing API parameters
result = generator.generate_qa(
    input_file='examples/generate/sample_document.md',
    output_file='examples/generate/env_qa.json',
    model='gpt-4',
    question_number=5
)

print("QA generation using environment variables:")
print(f"QA pairs generated: {len(result['qa_pairs'])}")
```

## Expected Output

### QA Generation Result
```json
{
  "qa_pairs": [
    {
      "question": "What is machine learning?",
      "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
      "chunk_id": "chunk_1",
      "confidence": 0.95
    },
    {
      "question": "What are the main types of machine learning?",
      "answer": "The main types are supervised learning, unsupervised learning, and reinforcement learning.",
      "chunk_id": "chunk_1",
      "confidence": 0.92
    }
  ],
  "metadata": {
    "total_chunks": 5,
    "total_questions": 25,
    "model_used": "gpt-4",
    "processing_time": 45.2
  }
}
```

### Multimodal QA Result
```json
[
  {
    "question": "What does the diagram show?",
    "answer": "The diagram illustrates the machine learning workflow from data collection to model deployment.",
    "image_context": "Figure 1: ML Pipeline Diagram",
    "chunk_id": "chunk_3"
  },
  {
    "question": "How many layers does the neural network have?",
    "answer": "The neural network diagram shows 4 hidden layers plus input and output layers, for a total of 6 layers.",
    "image_context": "Figure 2: Neural Network Architecture",
    "chunk_id": "chunk_4"
  }
]
```

## Best Practices

1. **Chunk Size**: Adjust chunk size based on document complexity (smaller chunks for technical content)
2. **Question Number**: Use 3-5 questions per chunk for most documents
3. **Overlap**: Use 10-20% overlap between chunks to maintain context
4. **API Keys**: Store API keys as environment variables for security
5. **Batch Processing**: Use multiple workers for large document collections
6. **Model Selection**: Use GPT-4 for best quality, GPT-3.5-turbo for speed/cost optimization
7. **Multimodal**: Only use multimodal generation when documents contain relevant images
8. **Quality Review**: Always review generated QA pairs for accuracy before use
