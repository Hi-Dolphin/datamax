# DataMax Examples

This directory contains comprehensive examples demonstrating how to use DataMax's complete data processing pipeline, including crawler, parser, cleaner, and generator components.

## 📁 Files Overview

### 1. `complete_pipeline_example.py`
**Complete end-to-end pipeline demonstration**

This example shows how to use all DataMax components together in a complete workflow:
- **Crawler**: Fetch academic papers from ArXiv
- **Parser**: Extract and structure content
- **Cleaner**: Clean and filter data
- **Generator**: Generate QA pairs and domain trees

```bash
python complete_pipeline_example.py
```

**Features:**
- Automated 5-step processing pipeline
- Comprehensive error handling
- Progress logging and reporting
- Configurable parameters
- Output summary and statistics

### 2. `quick_start_example.py`
**Individual component demonstrations**

This example demonstrates each DataMax component separately with simple, focused examples:

```bash
python quick_start_example.py
```

**What you'll learn:**
- How to configure and use the ArXiv crawler
- How to parse crawler data to markdown
- How to clean and filter text content
- How to generate QA pairs (with API key)
- How to customize configuration

### 3. `multimodal_example.py`
**Multimodal processing demonstration**

Specialized example showing how to process documents with images and generate multimodal QA pairs:
- 🖼️ **Image processing**: Extract and associate images from documents
- 🔗 **Image-text correlation**: Establish connections between images and text content
- 🤖 **Multimodal QA**: Generate question-answer pairs based on both images and text
- 📊 **Visualization support**: Handle various image formats

### 4. `crawler_example.py`
**Web crawler demonstration**

Specialized example showing how to use DataMax for web content crawling and processing:
- 🕷️ **Smart crawling**: Support multiple web scraping strategies
- 🔍 **Content extraction**: Automatically extract and structure web content
- 🧹 **Quality control**: Content cleaning and quality filtering
- 📈 **Batch processing**: Support batch URL processing

### 5. `config_example.yaml`
**Configuration template**

A comprehensive configuration file showing all available options for customizing DataMax behavior:

```yaml
# Example usage in your code
from datamax.crawler import CrawlerConfig
import yaml

with open('config_example.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
crawler_config = CrawlerConfig(config['crawlers'])
```

### 6. `test_examples.py`
**示例测试套件**

自动化测试脚本，验证所有示例的功能：
- 🧪 **自动测试**: 自动运行所有示例并验证输出
- 📊 **详细报告**: 生成完整的测试报告
- 🔍 **错误诊断**: 提供详细的错误信息和建议
- ⚡ **快速验证**: 快速验证环境配置是否正确

## 🚀 Getting Started

### Prerequisites

1. **Install DataMax**:
   ```bash
   pip install datamax
   ```

2. **Set up API credentials** (for QA generation):
   ```bash
   export DASHSCOPE_API_KEY="your-api-key-here"
   export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   ```

### Quick Start

1. **Run the quick start example** to understand individual components:
   ```bash
   cd examples
   python quick_start_example.py
   ```

2. **Run the complete pipeline** for end-to-end processing:
   ```bash
   python complete_pipeline_example.py
   ```

3. **Run the multimodal processing example** for image and text processing:
   ```bash
   python multimodal_example.py
   ```

4. **Run the web crawler example** for web content processing:
   ```bash
   python crawler_example.py
   ```

5. **Run the test suite** to validate all examples:
   ```bash
   # 运行测试套件（验证所有示例）
   python test_examples.py
   
   # 运行特定示例的测试
   python test_examples.py --example quick_start
   
   # 详细测试输出
   python test_examples.py --verbose
   ```

### Using Custom Configuration

```bash
# Use configuration file
python complete_pipeline_example.py --config config_example.yaml
```

## 📊 Example Outputs

After running the examples, you'll find the following output files:

### Complete Pipeline Example Output
```
examples/
├── pipeline_output/          # Complete pipeline outputs
│   ├── raw_data.json        # Crawled data from ArXiv
│   ├── parsed_data.md       # Parsed markdown content
│   ├── cleaned_data.md      # Cleaned and filtered content
│   ├── qa_pairs.json        # Generated QA pairs
│   ├── multimodal_qa_pairs.json  # Multimodal QA (if images present)
│   ├── pipeline_report.json # Execution summary
│   └── datamax.log         # Processing logs
└── quick_start_output/       # Quick start outputs
    ├── arxiv_results.json   # Raw crawler results
    ├── parsed_content.md    # Parsed content
    ├── cleaned_content.md   # Cleaned content
    └── qa_pairs.json        # Generated QA pairs
```

### Multimodal Example Output
```
multimodal_output/
├── sample_content.md      # Sample markdown content
├── images/               # Sample image files
├── parsed_document.md    # Parsed document
├── cleaned_content.md    # Cleaned content
├── image_associations.json # Image-text associations
├── multimodal_qa_pairs.json # Multimodal QA pairs
└── multimodal_report.json   # Processing report
```

### Crawler Example Output
```
crawler_output/
├── raw_content/          # Raw crawled content
├── parsed_content/       # Parsed content
├── cleaned_content/      # Cleaned content
├── qa_pairs/            # Generated QA pairs
├── crawl_summary.json   # Crawl summary
├── cleaning_summary.json # Cleaning summary
├── qa_summary.json      # QA generation summary
└── crawler_report.json  # Complete report
```

### Typical Output Examples

**Standard QA Pair Example**:
```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is a branch of artificial intelligence that enables computers to learn and improve without being explicitly programmed.",
  "confidence": 0.95,
  "source": "parsed_document.md",
  "metadata": {
    "domain": "technology",
    "difficulty": "beginner"
  }
}
```

**Multimodal QA Pair Example**:
```json
{
  "question": "What are the characteristics of the architecture shown in the image?",
  "answer": "The architecture diagram shows the multi-head attention mechanism of the Transformer model, including encoder and decoder structures.",
  "image_path": "./images/transformer_architecture.png",
  "image_description": "Transformer architecture overview diagram",
  "context": "This paper introduces a new Transformer architecture optimization method..."
}
```

## 🔧 Customization

### Modifying Search Parameters

```python
# In complete_pipeline_example.py or quick_start_example.py
CONFIG = {
    "search_query": "your custom query",  # Change search terms
    "max_results": 10,                    # Number of papers to fetch
    "question_number": 8,                 # QA pairs per chunk
    # ... other parameters
}
```

### Using Custom Configuration

```python
import yaml
from datamax.crawler import CrawlerConfig

# Load custom config
with open('config_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use in your pipeline
crawler_config = CrawlerConfig(config['crawlers'])
crawler = ArxivCrawler(config=crawler_config)
```

### Adding Custom Cleaning Rules

```python
from datamax.cleaner import AbnormalCleaner

# Custom cleaning workflow
cleaner = AbnormalCleaner(content)
cleaner.remove_html_tags()
cleaner.convert_newlines()
cleaner.simplify_chinese()
# Add your custom cleaning steps
result = cleaner.to_clean()
```

## 🎯 Use Cases

### Academic Research
```python
# Search for specific research topics
CONFIG["search_query"] = "transformer attention mechanism"
CONFIG["max_results"] = 20
```

### Content Processing
```python
# Process local documents instead of crawling
from datamax.generator import load_and_split_text

chunks = load_and_split_text(
    file_path="your_document.pdf",
    chunk_size=1000,
    chunk_overlap=200
)
```

### Multilingual Processing
```python
# Configure for Chinese content
from datamax.cleaner import AbnormalCleaner

cleaner = AbnormalCleaner(chinese_content)
cleaner.simplify_chinese()  # Convert traditional to simplified
result = cleaner.to_clean()
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Not Set**:
   ```
   ⚠️ Please set your DASHSCOPE_API_KEY environment variable
   ```
   **Solution**: Set your API key as shown in prerequisites

2. **Network Connection Issues**:
   ```
   ❌ Crawling failed: Connection timeout
   ```
   **Solution**: Check internet connection and increase timeout in config

3. **Insufficient Content**:
   ```
   ⚠️ Content failed character count filter
   ```
   **Solution**: Adjust `min_chars` parameter in cleaner configuration

### Debug Mode

Enable detailed logging:
```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

## 📚 Additional Resources

- **DataMax Documentation**: [Main README](../README.md)
- **API Reference**: Check individual module documentation
- **Configuration Guide**: See `config_example.yaml` for all options
- **Advanced Usage**: Explore the source code in `datamax/` directory

## 🤝 Contributing

To add new examples:
1. Create a new Python file with clear documentation
2. Follow the existing code style and structure
3. Include error handling and logging
4. Update this README with your example

## 📄 License

These examples are part of the DataMax project and follow the same license terms.