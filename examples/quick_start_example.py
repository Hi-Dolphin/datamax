#!/usr/bin/env python3
"""
DataMax Quick Start Example

This example demonstrates the basic usage of each DataMax component:
- Crawler: Fetch data from ArXiv
- Parser: Parse and extract content
- Cleaner: Clean and filter data
- Generator: Generate QA pairs

Usage:
    python quick_start_example.py
"""

import asyncio
import json
from pathlib import Path
from loguru import logger

# DataMax imports
from datamax.crawler import ArxivCrawler, CrawlerConfig
from datamax.parser import CrawlerParser
from datamax.cleaner import AbnormalCleaner, TextFilter
from datamax.generator import DomainTree, full_qa_labeling_process


def example_1_crawler():
    """Example 1: Using the Crawler"""
    print("\n" + "="*50)
    print("Example 1: ArXiv Crawler")
    print("="*50)
    
    async def crawl_papers():
        # Configure the crawler
        config = {
            'arxiv': {
                'max_results': 3,
                'timeout': 30
            },
            'storage': {
                'type': 'local',
                'base_path': './quick_start_output'
            }
        }
        
        # Create crawler instance
        crawler = ArxivCrawler(config=config)
        
        # Search for papers
        logger.info("Searching for machine learning papers...")
        results = await crawler.crawl_async(
            "machine learning",
            max_results=3
        )
        
        # Save results
        output_path = Path("./quick_start_output/arxiv_results.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Found {len(results.get('data', []))} papers, saved to {output_path}")
        return str(output_path)
    
    return asyncio.run(crawl_papers())


def example_2_parser(data_path: str):
    """Example 2: Using the Parser"""
    print("\n" + "="*50)
    print("Example 2: Data Parser")
    print("="*50)
    
    # Create parser instance with file path
    logger.info(f"Loading data from {data_path}...")
    parser = CrawlerParser(file_path=data_path)
    
    # Parse to markdown
    logger.info("Parsing data to markdown format...")
    result = parser.parse()
    markdown_content = result.content
    
    # Save parsed content
    output_path = Path("./quick_start_output/parsed_content.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info(f"Parsed content saved to {output_path}")
    return str(output_path)


def example_3_cleaner(content_path: str):
    """Example 3: Using the Cleaner"""
    print("\n" + "="*50)
    print("Example 3: Data Cleaner")
    print("="*50)
    
    # Read content
    with open(content_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.info("Original content length: {} characters".format(len(content)))
    
    # Step 1: Abnormal cleaning
    logger.info("Applying abnormal character cleaning...")
    cleaner = AbnormalCleaner(content)
    cleaned_result = cleaner.to_clean()
    cleaned_content = cleaned_result['text']
    
    logger.info("After cleaning length: {} characters".format(len(cleaned_content)))
    
    # Step 2: Text filtering
    logger.info("Applying text filters...")
    text_filter = TextFilter(cleaned_content)
    
    # Check various filters
    char_filter_passed = text_filter.filter_by_char_count(min_chars=50, max_chars=10000)
    repetition_filter_passed = text_filter.filter_by_word_repetition(threshold=0.7)
    numeric_filter_passed = text_filter.filter_by_numeric_content(threshold=0.6)
    
    logger.info(f"Character count filter: {'PASSED' if char_filter_passed else 'FAILED'}")
    logger.info(f"Word repetition filter: {'PASSED' if repetition_filter_passed else 'FAILED'}")
    logger.info(f"Numeric content filter: {'PASSED' if numeric_filter_passed else 'FAILED'}")
    
    # Save cleaned content
    output_path = Path("./quick_start_output/cleaned_content.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    logger.info(f"Cleaned content saved to {output_path}")
    return str(output_path)


def example_4_generator(content_path: str):
    """Example 4: Using the Generator"""
    print("\n" + "="*50)
    print("Example 4: QA Generator")
    print("="*50)
    
    # Read cleaned content
    with open(content_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if API key is available
    import os
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        logger.warning("DASHSCOPE_API_KEY not found. Skipping QA generation.")
        logger.info("To use QA generation, set your API key:")
        logger.info("export DASHSCOPE_API_KEY='your-actual-api-key'")
        return None
    
    # Example 4.1: Create Domain Tree
    logger.info("Creating domain tree...")
    domain_tree = DomainTree()
    logger.info(f"Domain tree created with structure: {domain_tree.__class__.__name__}")
    
    # Example 4.2: Generate QA pairs
    logger.info("Generating QA pairs...")
    try:
        qa_results = full_qa_labeling_process(
            content=content,
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model_name="qwen-plus",
            chunk_size=800,
            chunk_overlap=100,
            question_number=3,  # Generate fewer questions for quick demo
            max_workers=2,
            use_tree_label=True,
            interactive_tree=False
        )
        
        # Save QA results
        output_path = Path("./quick_start_output/qa_pairs.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(qa_results)} QA pairs, saved to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"QA generation failed: {str(e)}")
        return None


def example_5_integration():
    """Example 5: Integration with Custom Configuration"""
    print("\n" + "="*50)
    print("Example 5: Custom Configuration")
    print("="*50)
    
    # Example of loading custom configuration
    custom_config = {
        "crawler": {
            "search_query": "natural language processing",
            "max_results": 2
        },
        "parser": {
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "cleaner": {
            "min_chars": 100,
            "max_chars": 5000
        },
        "generator": {
            "question_number": 2,
            "max_workers": 1
        }
    }
    
    logger.info("Custom configuration loaded:")
    for section, settings in custom_config.items():
        logger.info(f"  {section}: {settings}")
    
    logger.info("Configuration can be used to customize pipeline behavior")
    return custom_config


def main():
    """Main function to run all examples"""
    print("DataMax Quick Start Examples")
    print("This demo shows basic usage of each DataMax component.\n")
    
    try:
        # Example 1: Crawler
        data_path = example_1_crawler()
        
        # Example 2: Parser
        content_path = example_2_parser(data_path)
        
        # Example 3: Cleaner
        cleaned_path = example_3_cleaner(content_path)
        
        # Example 4: Generator
        qa_path = example_4_generator(cleaned_path)
        
        # Example 5: Configuration
        config = example_5_integration()
        
        # Summary
        print("\n" + "="*60)
        print("QUICK START SUMMARY")
        print("="*60)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print(f"  - Raw data: ./quick_start_output/arxiv_results.json")
        print(f"  - Parsed content: ./quick_start_output/parsed_content.md")
        print(f"  - Cleaned content: ./quick_start_output/cleaned_content.md")
        if qa_path:
            print(f"  - QA pairs: ./quick_start_output/qa_pairs.json")
        else:
            print(f"  - QA pairs: (skipped - no API key)")
        
        print("\nNext steps:")
        print("  1. Set DASHSCOPE_API_KEY to enable QA generation")
        print("  2. Try the complete_pipeline_example.py for full workflow")
        print("  3. Customize config_example.yaml for your needs")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Quick start failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)