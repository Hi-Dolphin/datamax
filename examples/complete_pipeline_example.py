#!/usr/bin/env python3
"""
DataMax Complete Pipeline Example

This example demonstrates a complete data processing pipeline using DataMax:
1. Crawler: Fetch data from ArXiv
2. Parser: Parse and extract content
3. Cleaner: Clean and filter the data
4. Generator: Generate QA pairs and domain trees

Usage:
    python complete_pipeline_example.py
"""

import asyncio
import json
import os
from pathlib import Path
from loguru import logger

# DataMax imports
from datamax.crawler import ArxivCrawler
from datamax.parser import CrawlerParser
from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization
from datamax.generator import (
    DomainTree,
    full_qa_labeling_process,
    generate_multimodal_qa_pairs,
    parse_markdown_and_associate_images
)

# Configuration
CONFIG = {
    "output_dir": "./pipeline_output",
    "search_query": "maritime ai",
    "max_results": 3,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "question_number": 5,
    "api_key": os.getenv("DASHSCOPE_API_KEY", "your-api-key"),
    "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "model_name": "qwen-plus"
}


class DataMaxPipeline:
    """Complete DataMax processing pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        crawler_config = {
            'max_results': config["max_results"],
            'timeout': 60,
            'base_path': str(self.output_dir / 'raw_data')
        }
        
        self.crawler = ArxivCrawler(config=crawler_config)
        # Parser will be initialized when needed with specific file path
        
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
    
    async def step1_crawl_data(self) -> str:
        """Step 1: Crawl data from ArXiv."""
        logger.info("🕷️ Step 1: Starting data crawling...")
        
        try:
            # Search for papers
            results = await self.crawler.crawl_async(
                self.config["search_query"]
            )
            
            # Save raw results
            raw_data_path = self.output_dir / "raw_data.json"
            with open(raw_data_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Crawled {len(results)} papers, saved to {raw_data_path}")
            return str(raw_data_path)
            
        except Exception as e:
            logger.error(f"❌ Crawling failed: {str(e)}")
            raise
    
    def step2_parse_data(self, raw_data_path: str) -> str:
        """Step 2: Parse the crawled data."""
        logger.info("📄 Step 2: Starting data parsing...")
        
        try:
            # Create parser instance with the raw data file path
            parser = CrawlerParser(raw_data_path)
            # Parse data (load_data is called automatically)
            parsed_output = parser.parse()
            parsed_content = parsed_output.content
            
            # Save parsed content
            parsed_data_path = self.output_dir / "parsed_data.md"
            with open(parsed_data_path, 'w', encoding='utf-8') as f:
                f.write(parsed_content)
            
            logger.info(f"✅ Data parsed and saved to {parsed_data_path}")
            return str(parsed_data_path)
            
        except Exception as e:
            logger.error(f"❌ Parsing failed: {str(e)}")
            raise
    
    def step3_clean_data(self, parsed_data_path: str) -> str:
        """Step 3: Clean and filter the parsed data."""
        logger.info("🧹 Step 3: Starting data cleaning...")
        
        try:
            # Read parsed data
            with open(parsed_data_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Step 3.1: Abnormal cleaning
            logger.info("  🔧 Applying abnormal character cleaning...")
            abnormal_cleaner = AbnormalCleaner(content)
            cleaned_content = abnormal_cleaner.to_clean()
            
            # Step 3.2: Text filtering
            logger.info("  🔍 Applying text filtering...")
            text_filter = TextFilter(cleaned_content['text'])
            
            # Check if content passes filters
            if not text_filter.filter_by_char_count(min_chars=100, max_chars=50000):
                logger.warning("  ⚠️ Content failed character count filter")
            
            if not text_filter.filter_by_word_repetition(threshold=0.7):
                logger.warning("  ⚠️ Content failed word repetition filter")
            
            if not text_filter.filter_by_numeric_content(threshold=0.6):
                logger.warning("  ⚠️ Content failed numeric content filter")
            
            # Step 3.3: Privacy desensitization
            logger.info("  🔒 Applying privacy desensitization...")
            privacy_cleaner = PrivacyDesensitization(cleaned_content['text'])
            final_content = privacy_cleaner.to_private()
            
            # Save cleaned data
            cleaned_data_path = self.output_dir / "cleaned_data.md"
            with open(cleaned_data_path, 'w', encoding='utf-8') as f:
                f.write(final_content['text'])
            
            logger.info(f"✅ Data cleaned and saved to {cleaned_data_path}")
            return str(cleaned_data_path)
            
        except Exception as e:
            logger.error(f"❌ Cleaning failed: {str(e)}")
            raise
    
    def step4_generate_qa_pairs(self, cleaned_data_path: str) -> dict:
        """Step 4: Generate QA pairs and domain tree."""
        logger.info("🤖 Step 4: Starting QA generation...")
        
        try:
            # Read cleaned data
            with open(cleaned_data_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Step 4.1: Generate domain tree
            logger.info("  🌳 Generating domain tree...")
            domain_tree = DomainTree()
            
            # Step 4.2: Generate QA pairs using the full process
            logger.info("  ❓ Generating QA pairs...")
            qa_results = full_qa_labeling_process(
                content=content,
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
                model_name=self.config["model_name"],
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
                question_number=self.config["question_number"],
                max_workers=3,
                use_tree_label=True,
                interactive_tree=False  # Set to False for automated processing
            )
            
            # Extract domain tree from qa_results if available
            if 'domain_tree' in qa_results and qa_results['domain_tree']:
                domain_tree = qa_results['domain_tree']
                
                # Save domain tree structure and content
                domain_tree_output_path = self.output_dir / "domain_tree.json"
                with open(domain_tree_output_path, 'w', encoding='utf-8') as f:
                    json.dump(domain_tree.to_json(), f, indent=2, ensure_ascii=False)
                logger.info(f"  💾 Domain tree saved to: {domain_tree_output_path}")
            
            # Save QA results (extract qa_pairs from the new structure)
            qa_pairs = qa_results.get('qa_pairs', qa_results)  # Fallback for backward compatibility
            qa_output_path = self.output_dir / "qa_pairs.json"
            with open(qa_output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ QA pairs generated and saved to {qa_output_path}")
            
            return {
                "qa_pairs_path": str(qa_output_path),
                "qa_results": qa_results,
                "qa_pairs_count": len(qa_pairs),
                "domain_tree_path": str(domain_tree_output_path) if 'domain_tree' in qa_results and qa_results['domain_tree'] else None
            }
            
        except Exception as e:
            logger.error(f"❌ QA generation failed: {str(e)}")
            raise
    
    def step5_generate_multimodal_qa(self, cleaned_data_path: str) -> str:
        """Step 5: Generate multimodal QA pairs (if images are present)."""
        logger.info("🖼️ Step 5: Starting multimodal QA generation...")
        
        try:
            # Check if the content has associated images
            image_associations = parse_markdown_and_associate_images(
                cleaned_data_path,
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            
            if not image_associations:
                logger.info("  ℹ️ No images found, skipping multimodal QA generation")
                return None
            
            # Generate multimodal QA pairs
            logger.info(f"  🎨 Found {len(image_associations)} image associations")
            multimodal_qa_results = generate_multimodal_qa_pairs(
                image_associations,
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
                model_name=self.config["model_name"]
            )
            
            # Save multimodal QA results
            multimodal_qa_path = self.output_dir / "multimodal_qa_pairs.json"
            with open(multimodal_qa_path, 'w', encoding='utf-8') as f:
                json.dump(multimodal_qa_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Multimodal QA pairs generated and saved to {multimodal_qa_path}")
            return str(multimodal_qa_path)
            
        except Exception as e:
            logger.error(f"❌ Multimodal QA generation failed: {str(e)}")
            # Don't raise here as this is optional
            return None
    
    def generate_summary_report(self, results: dict) -> str:
        """Generate a summary report of the pipeline execution."""
        logger.info("📊 Generating pipeline summary report...")
        
        # Create a serializable copy of results
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'to_json'):  # Handle DomainTree objects
                serializable_results[key] = value.to_json()
            elif isinstance(value, dict) and 'domain_tree' in value:
                # Handle nested structures containing domain_tree
                serializable_value = value.copy()
                if value['domain_tree'] and hasattr(value['domain_tree'], 'to_json'):
                    serializable_value['domain_tree'] = value['domain_tree'].to_json()
                else:
                    serializable_value['domain_tree'] = None
                serializable_results[key] = serializable_value
            else:
                serializable_results[key] = value
        
        # Get QA pairs count from the new structure
        qa_results = results.get("qa_results", {})
        if isinstance(qa_results, dict) and 'qa_pairs' in qa_results:
            qa_pairs_count = len(qa_results['qa_pairs'])
        else:
            qa_pairs_count = len(qa_results) if qa_results else 0
        
        report = {
            "pipeline_config": self.config,
            "execution_results": serializable_results,
            "output_files": {
                "raw_data": results.get("raw_data_path"),
                "parsed_data": results.get("parsed_data_path"),
                "cleaned_data": results.get("cleaned_data_path"),
                "qa_pairs": results.get("qa_pairs_path"),
                "domain_tree": results.get("domain_tree_path"),
                "multimodal_qa": results.get("multimodal_qa_path")
            },
            "statistics": {
                "total_qa_pairs": qa_pairs_count,
                "processing_steps": 5
            }
        }
        
        # Save report
        report_path = self.output_dir / "pipeline_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Pipeline report saved to {report_path}")
        return str(report_path)
    
    async def run_complete_pipeline(self) -> dict:
        """Run the complete data processing pipeline."""
        logger.info("🚀 Starting complete DataMax pipeline...")
        
        results = {}
        
        try:
            # Step 1: Crawl data
            results["raw_data_path"] = await self.step1_crawl_data()
            
            # Step 2: Parse data
            results["parsed_data_path"] = self.step2_parse_data(results["raw_data_path"])
            
            # Step 3: Clean data
            results["cleaned_data_path"] = self.step3_clean_data(results["parsed_data_path"])
            
            # Step 4: Generate QA pairs
            qa_results = self.step4_generate_qa_pairs(results["cleaned_data_path"])
            results.update(qa_results)
            
            # Step 5: Generate multimodal QA (optional)
            multimodal_qa_path = self.step5_generate_multimodal_qa(results["cleaned_data_path"])
            if multimodal_qa_path:
                results["multimodal_qa_path"] = multimodal_qa_path
            
            # Generate summary report
            results["report_path"] = self.generate_summary_report(results)
            
            logger.info("🎉 Complete pipeline execution finished successfully!")
            logger.info(f"📁 All outputs saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Pipeline execution failed: {str(e)}")
            raise


async def main():
    """Main function to run the complete pipeline example."""
    
    # Check API key
    if CONFIG["api_key"] == "your-api-key-here":
        logger.warning("⚠️ Please set your DASHSCOPE_API_KEY environment variable")
        logger.info("You can still run the crawler, parser, and cleaner steps")
    
    # Initialize and run pipeline
    pipeline = DataMaxPipeline(CONFIG)
    
    try:
        results = await pipeline.run_complete_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"📊 Total QA pairs generated: {len(results.get('qa_results', []))}")
        print(f"📁 Output directory: {pipeline.output_dir}")
        print("\n📋 Generated files:")
        for key, path in results.items():
            if key.endswith('_path') and path:
                print(f"  • {key.replace('_path', '').replace('_', ' ').title()}: {Path(path).name}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the complete pipeline
    exit_code = asyncio.run(main())
    exit(exit_code)