#!/usr/bin/env python3
"""
DataMax Crawler Example

This example demonstrates how to use DataMax's crawler functionality
to scrape web content and process it through the complete pipeline.

Features:
- Web scraping with various strategies
- Content extraction and parsing
- Data cleaning and filtering
- QA generation from scraped content

Usage:
    python crawler_example.py
"""

import os
import json
import asyncio
from pathlib import Path
from loguru import logger
from urllib.parse import urljoin, urlparse

# DataMax imports
from datamax.parser import DataMax
from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization
from datamax.generator import DomainTree, full_qa_labeling_process

# Configuration
CONFIG = {
    "output_dir": "./crawler_output",
    "target_urls": [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://docs.python.org/3/tutorial/introduction.html"
    ],
    "max_pages": 3,
    "delay_between_requests": 2,  # seconds
    "user_agent": "DataMax-Crawler/1.0 (+https://github.com/your-repo/datamax)",
    "timeout": 30,
    "max_content_length": 100000,  # characters
    "enable_privacy_protection": True,
    "qa_generation": {
        "enabled": True,
        "max_qa_pairs": 10,
        "domain": "technology"
    }
}


class WebCrawler:
    """Web crawler for DataMax content processing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw_content").mkdir(exist_ok=True)
        (self.output_dir / "parsed_content").mkdir(exist_ok=True)
        (self.output_dir / "cleaned_content").mkdir(exist_ok=True)
        (self.output_dir / "qa_pairs").mkdir(exist_ok=True)
        
        logger.info(f"Web crawler initialized with output directory: {self.output_dir}")
    
    def get_safe_filename(self, url: str) -> str:
        """Generate a safe filename from URL."""
        parsed = urlparse(url)
        filename = f"{parsed.netloc}_{parsed.path.replace('/', '_')}"
        # Remove invalid characters
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        return filename[:100]  # Limit length
    
    def step1_crawl_web_content(self, urls: list = None) -> list:
        """Step 1: Crawl web content from target URLs."""
        logger.info("🕷️ Step 1: Crawling web content...")
        
        if urls is None:
            urls = self.config["target_urls"]
        
        crawled_data = []
        
        for i, url in enumerate(urls[:self.config["max_pages"]]):
            logger.info(f"Crawling {i+1}/{len(urls[:self.config['max_pages']])}: {url}")
            
            try:
                # Use DataMax to crawl the URL
                dm = DataMax(url=url, to_markdown=True)
                
                # Get the crawled data
                data = dm.get_data()
                
                if isinstance(data, list) and data:
                    content = data[0].get('content', '')
                    metadata = data[0].get('metadata', {})
                else:
                    content = data.get('content', '') if isinstance(data, dict) else str(data)
                    metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
                
                if not content:
                    logger.warning(f"No content extracted from {url}")
                    continue
                
                # Prepare crawled item
                crawled_item = {
                    "url": url,
                    "content": content,
                    "metadata": metadata,
                    "content_length": len(content),
                    "crawl_timestamp": dm.get_timestamp() if hasattr(dm, 'get_timestamp') else None
                }
                
                # Save raw content
                filename = self.get_safe_filename(url)
                raw_path = self.output_dir / "raw_content" / f"{filename}.md"
                with open(raw_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                crawled_item["raw_file"] = str(raw_path)
                crawled_data.append(crawled_item)
                
                logger.info(f"✅ Crawled {len(content)} characters from {url}")
                
                # Delay between requests
                if i < len(urls) - 1:
                    import time
                    time.sleep(self.config["delay_between_requests"])
                
            except Exception as e:
                logger.error(f"❌ Failed to crawl {url}: {str(e)}")
                continue
        
        # Save crawl summary
        summary_path = self.output_dir / "crawl_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(crawled_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Crawled {len(crawled_data)} pages successfully")
        logger.info(f"Crawl summary saved to {summary_path}")
        
        return crawled_data
    
    def step2_parse_and_structure(self, crawled_data: list) -> list:
        """Step 2: Parse and structure the crawled content."""
        logger.info("📄 Step 2: Parsing and structuring content...")
        
        parsed_data = []
        
        for item in crawled_data:
            logger.info(f"Parsing content from {item['url']}...")
            
            try:
                content = item['content']
                
                # Basic content structuring
                structured_content = {
                    "url": item['url'],
                    "title": self._extract_title(content),
                    "sections": self._extract_sections(content),
                    "content": content,
                    "metadata": item.get('metadata', {}),
                    "word_count": len(content.split()),
                    "char_count": len(content)
                }
                
                # Save parsed content
                filename = self.get_safe_filename(item['url'])
                parsed_path = self.output_dir / "parsed_content" / f"{filename}_parsed.json"
                with open(parsed_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_content, f, indent=2, ensure_ascii=False)
                
                structured_content["parsed_file"] = str(parsed_path)
                parsed_data.append(structured_content)
                
                logger.info(f"✅ Parsed content: {structured_content['word_count']} words, {len(structured_content['sections'])} sections")
                
            except Exception as e:
                logger.error(f"❌ Failed to parse content from {item['url']}: {str(e)}")
                continue
        
        logger.info(f"✅ Parsed {len(parsed_data)} documents")
        return parsed_data
    
    def _extract_title(self, content: str) -> str:
        """Extract title from markdown content."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"
    
    def _extract_sections(self, content: str) -> list:
        """Extract sections from markdown content."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": '\n'.join(current_content).strip(),
                        "level": current_section.count('#')
                    })
                
                # Start new section
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections.append({
                "title": current_section,
                "content": '\n'.join(current_content).strip(),
                "level": current_section.count('#')
            })
        
        return sections
    
    def step3_clean_and_filter(self, parsed_data: list) -> list:
        """Step 3: Clean and filter the parsed content."""
        logger.info("🧹 Step 3: Cleaning and filtering content...")
        
        cleaned_data = []
        
        for item in parsed_data:
            logger.info(f"Cleaning content from {item['url']}...")
            
            try:
                content = item['content']
                
                # Apply data cleaning
                cleaner = AbnormalCleaner(content)
                cleaner.convert_newlines()
                cleaner.single_space()
                cleaner.tabs_to_spaces()
                cleaner.simplify_chinese()
                cleaner.remove_invisible_chars()
                
                cleaned_result = cleaner.clean()
                cleaned_content = cleaned_result['text']
                
                # Apply text filtering
                text_filter = TextFilter(cleaned_content)
                
                # Check content quality
                quality_checks = {
                    "char_count_ok": text_filter.filter_by_char_count(min_chars=500, max_chars=self.config["max_content_length"]),
                    "word_count_ok": text_filter.filter_by_word_count(min_words=100, max_words=10000),
                    "line_count_ok": text_filter.filter_by_line_count(min_lines=10, max_lines=1000),
                    "language_ok": text_filter.filter_by_language(['en', 'zh']),
                    "no_spam": text_filter.filter_spam_content()
                }
                
                # Apply privacy protection if enabled
                if self.config["enable_privacy_protection"]:
                    privacy_filter = PrivacyDesensitization(cleaned_content)
                    privacy_filter.remove_email()
                    privacy_filter.remove_phone()
                    privacy_filter.remove_id_card()
                    privacy_filter.remove_credit_card()
                    cleaned_content = privacy_filter.get_text()
                
                # Prepare cleaned item
                cleaned_item = {
                    "url": item['url'],
                    "title": item['title'],
                    "content": cleaned_content,
                    "original_length": len(item['content']),
                    "cleaned_length": len(cleaned_content),
                    "quality_checks": quality_checks,
                    "passed_filters": all(quality_checks.values()),
                    "sections": item['sections'],
                    "metadata": item['metadata']
                }
                
                # Save cleaned content
                filename = self.get_safe_filename(item['url'])
                cleaned_path = self.output_dir / "cleaned_content" / f"{filename}_cleaned.md"
                with open(cleaned_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                cleaned_item["cleaned_file"] = str(cleaned_path)
                cleaned_data.append(cleaned_item)
                
                status = "✅ PASSED" if cleaned_item["passed_filters"] else "⚠️ FILTERED"
                logger.info(f"{status} Content cleaned: {cleaned_item['original_length']} → {cleaned_item['cleaned_length']} chars")
                
            except Exception as e:
                logger.error(f"❌ Failed to clean content from {item['url']}: {str(e)}")
                continue
        
        # Save cleaning summary
        cleaning_summary = {
            "total_processed": len(parsed_data),
            "successfully_cleaned": len(cleaned_data),
            "passed_filters": len([item for item in cleaned_data if item["passed_filters"]]),
            "quality_statistics": self._calculate_quality_stats(cleaned_data)
        }
        
        summary_path = self.output_dir / "cleaning_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(cleaning_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Cleaned {len(cleaned_data)} documents, {cleaning_summary['passed_filters']} passed all filters")
        return cleaned_data
    
    def _calculate_quality_stats(self, cleaned_data: list) -> dict:
        """Calculate quality statistics for cleaned data."""
        if not cleaned_data:
            return {}
        
        stats = {
            "avg_length": sum(item["cleaned_length"] for item in cleaned_data) / len(cleaned_data),
            "min_length": min(item["cleaned_length"] for item in cleaned_data),
            "max_length": max(item["cleaned_length"] for item in cleaned_data),
            "total_content": sum(item["cleaned_length"] for item in cleaned_data)
        }
        
        # Quality check statistics
        quality_checks = {}
        if cleaned_data:
            first_item_checks = cleaned_data[0].get("quality_checks", {})
            for check_name in first_item_checks.keys():
                passed_count = sum(1 for item in cleaned_data if item.get("quality_checks", {}).get(check_name, False))
                quality_checks[check_name] = {
                    "passed": passed_count,
                    "total": len(cleaned_data),
                    "percentage": (passed_count / len(cleaned_data)) * 100
                }
        
        stats["quality_checks"] = quality_checks
        return stats
    
    def step4_generate_qa_pairs(self, cleaned_data: list) -> list:
        """Step 4: Generate QA pairs from cleaned content."""
        logger.info("🤖 Step 4: Generating QA pairs...")
        
        if not self.config["qa_generation"]["enabled"]:
            logger.info("QA generation disabled in config")
            return []
        
        qa_results = []
        
        # Filter content that passed quality checks
        valid_content = [item for item in cleaned_data if item["passed_filters"]]
        
        if not valid_content:
            logger.warning("No valid content available for QA generation")
            return []
        
        for item in valid_content:
            logger.info(f"Generating QA pairs for {item['url']}...")
            
            try:
                # Create domain tree for the content
                domain_tree = DomainTree()
                domain_tree.build_tree_from_text(
                    text=item['content'],
                    domain=self.config["qa_generation"]["domain"]
                )
                
                # Generate QA pairs using the full labeling process
                qa_pairs = full_qa_labeling_process(
                    content=item['content'],
                    domain=self.config["qa_generation"]["domain"],
                    max_qa_pairs=self.config["qa_generation"]["max_qa_pairs"]
                )
                
                if qa_pairs:
                    qa_result = {
                        "url": item['url'],
                        "title": item['title'],
                        "domain": self.config["qa_generation"]["domain"],
                        "qa_pairs": qa_pairs,
                        "qa_count": len(qa_pairs),
                        "content_length": item['cleaned_length']
                    }
                    
                    # Save QA pairs
                    filename = self.get_safe_filename(item['url'])
                    qa_path = self.output_dir / "qa_pairs" / f"{filename}_qa.json"
                    with open(qa_path, 'w', encoding='utf-8') as f:
                        json.dump(qa_result, f, indent=2, ensure_ascii=False)
                    
                    qa_result["qa_file"] = str(qa_path)
                    qa_results.append(qa_result)
                    
                    logger.info(f"✅ Generated {len(qa_pairs)} QA pairs")
                else:
                    logger.warning(f"No QA pairs generated for {item['url']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate QA pairs for {item['url']}: {str(e)}")
                continue
        
        # Save QA generation summary
        qa_summary = {
            "total_documents": len(valid_content),
            "successful_generations": len(qa_results),
            "total_qa_pairs": sum(result["qa_count"] for result in qa_results),
            "average_qa_per_document": sum(result["qa_count"] for result in qa_results) / len(qa_results) if qa_results else 0
        }
        
        summary_path = self.output_dir / "qa_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(qa_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {qa_summary['total_qa_pairs']} QA pairs from {len(qa_results)} documents")
        return qa_results
    
    def generate_final_report(self, crawled_data: list, parsed_data: list, cleaned_data: list, qa_results: list) -> str:
        """Generate a comprehensive final report."""
        logger.info("📊 Generating final crawler report...")
        
        report = {
            "crawler_config": self.config,
            "processing_summary": {
                "crawled_pages": len(crawled_data),
                "parsed_documents": len(parsed_data),
                "cleaned_documents": len(cleaned_data),
                "quality_passed": len([item for item in cleaned_data if item["passed_filters"]]),
                "qa_generated": len(qa_results),
                "total_qa_pairs": sum(result["qa_count"] for result in qa_results)
            },
            "content_statistics": {
                "total_content_crawled": sum(item["content_length"] for item in crawled_data),
                "total_content_cleaned": sum(item["cleaned_length"] for item in cleaned_data),
                "average_content_length": sum(item["cleaned_length"] for item in cleaned_data) / len(cleaned_data) if cleaned_data else 0
            },
            "urls_processed": [item["url"] for item in crawled_data],
            "quality_metrics": self._calculate_quality_stats(cleaned_data),
            "output_files": {
                "raw_content_dir": str(self.output_dir / "raw_content"),
                "parsed_content_dir": str(self.output_dir / "parsed_content"),
                "cleaned_content_dir": str(self.output_dir / "cleaned_content"),
                "qa_pairs_dir": str(self.output_dir / "qa_pairs")
            }
        }
        
        # Save final report
        report_path = self.output_dir / "crawler_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Final report saved to {report_path}")
        return str(report_path)
    
    def run_crawler_pipeline(self, urls: list = None) -> dict:
        """Run the complete crawler pipeline."""
        logger.info("🚀 Starting web crawler pipeline...")
        
        try:
            # Step 1: Crawl web content
            crawled_data = self.step1_crawl_web_content(urls)
            
            if not crawled_data:
                logger.error("No content was successfully crawled")
                return {}
            
            # Step 2: Parse and structure
            parsed_data = self.step2_parse_and_structure(crawled_data)
            
            # Step 3: Clean and filter
            cleaned_data = self.step3_clean_and_filter(parsed_data)
            
            # Step 4: Generate QA pairs
            qa_results = self.step4_generate_qa_pairs(cleaned_data)
            
            # Generate final report
            report_path = self.generate_final_report(crawled_data, parsed_data, cleaned_data, qa_results)
            
            results = {
                "crawled_data": crawled_data,
                "parsed_data": parsed_data,
                "cleaned_data": cleaned_data,
                "qa_results": qa_results,
                "report_path": report_path
            }
            
            logger.info("🎉 Web crawler pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"💥 Crawler pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the crawler example."""
    
    print("🕷️ DataMax Web Crawler Example")
    print("This demo shows how to crawl web content and process it through the complete DataMax pipeline.\n")
    
    # Initialize crawler
    crawler = WebCrawler(CONFIG)
    
    try:
        # Run crawler pipeline
        results = crawler.run_crawler_pipeline()
        
        if not results:
            logger.error("Crawler pipeline failed to produce results")
            return 1
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 WEB CRAWLER SUMMARY")
        print("="*60)
        print(f"🕷️ Pages crawled: {len(results['crawled_data'])}")
        print(f"📄 Documents parsed: {len(results['parsed_data'])}")
        print(f"🧹 Documents cleaned: {len(results['cleaned_data'])}")
        print(f"✅ Quality passed: {len([item for item in results['cleaned_data'] if item['passed_filters']])}")
        print(f"❓ QA pairs generated: {sum(result['qa_count'] for result in results['qa_results'])}")
        print(f"📁 Output directory: {crawler.output_dir}")
        
        print("\n📋 Processed URLs:")
        for item in results['crawled_data']:
            status = "✅" if any(c['url'] == item['url'] and c['passed_filters'] for c in results['cleaned_data']) else "⚠️"
            print(f"  {status} {item['url']} ({item['content_length']} chars)")
        
        print("\n📁 Output directories:")
        print(f"  • Raw content: {crawler.output_dir / 'raw_content'}")
        print(f"  • Parsed content: {crawler.output_dir / 'parsed_content'}")
        print(f"  • Cleaned content: {crawler.output_dir / 'cleaned_content'}")
        print(f"  • QA pairs: {crawler.output_dir / 'qa_pairs'}")
        
        print("\n🔗 Next steps:")
        print("  1. Review the crawler report in crawler_report.json")
        print("  2. Check the cleaned content for quality")
        print("  3. Examine the generated QA pairs")
        print("  4. Modify CONFIG to crawl your own URLs")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Crawler example failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)