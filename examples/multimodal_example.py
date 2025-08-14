#!/usr/bin/env python3
"""
DataMax Multimodal Processing Example

This example demonstrates how to process documents with images and generate
multimodal QA pairs using DataMax's multimodal capabilities.

Features:
- Process documents with embedded images
- Extract and associate images with text content
- Generate multimodal QA pairs
- Handle various image formats

Usage:
    python multimodal_example.py
"""

import os
import json
import asyncio
from pathlib import Path
from loguru import logger

# DataMax imports
from datamax.parser import DataMax
from datamax.cleaner import AbnormalCleaner, TextFilter
from datamax.generator import (
    parse_markdown_and_associate_images,
    generate_multimodal_qa_with_dashscope,
    get_instruction_prompt
)

# Configuration
CONFIG = {
    "output_dir": "./multimodal_output",
    "api_key": os.getenv("DASHSCOPE_API_KEY", "your-api-key-here"),
    "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "model_name": "qwen-vl-plus",  # Use vision-language model
    "sample_document": "sample_document_with_images.pdf"  # You can replace with your document
}


class MultimodalProcessor:
    """Multimodal document processor for DataMax."""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Multimodal processor initialized with output directory: {self.output_dir}")
    
    def create_sample_markdown_with_images(self) -> str:
        """Create a sample markdown file with image references for demonstration."""
        logger.info("📝 Creating sample markdown with image references...")
        
        sample_content = """
# Machine Learning Research Paper

## Abstract
This paper presents a novel approach to transformer architecture optimization.

![Architecture Diagram](./images/transformer_architecture.png)
*Figure 1: Transformer architecture overview*

## Introduction
Transformers have revolutionized natural language processing. The attention mechanism
allows models to focus on relevant parts of the input sequence.

## Methodology

### Attention Mechanism
The multi-head attention mechanism can be visualized as follows:

![Attention Visualization](./images/attention_heatmap.jpg)
*Figure 2: Attention weights visualization*

The mathematical formulation is:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Experimental Setup
We conducted experiments on multiple datasets:

![Results Chart](./images/results_chart.png)
*Figure 3: Performance comparison across datasets*

## Results
Our approach shows significant improvements:
- BLEU score: 34.2 (+2.1)
- ROUGE-L: 28.7 (+1.8)
- Training time: 15% reduction

![Performance Graph](./images/performance_graph.svg)
*Figure 4: Training loss over epochs*

## Conclusion
The proposed method demonstrates superior performance while maintaining efficiency.

### Future Work
- Explore larger model sizes
- Test on multilingual datasets
- Investigate few-shot learning capabilities

![Future Roadmap](./images/roadmap.png)
*Figure 5: Research roadmap for next phase*
"""
        
        # Save sample content
        sample_path = self.output_dir / "sample_content.md"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        logger.info(f"✅ Sample markdown created: {sample_path}")
        return str(sample_path)
    
    def create_sample_images_directory(self):
        """Create sample image files for demonstration."""
        logger.info("🖼️ Creating sample images directory...")
        
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Create placeholder image files (SVG format for text-based demonstration)
        image_files = [
            "transformer_architecture.png",
            "attention_heatmap.jpg", 
            "results_chart.png",
            "performance_graph.svg",
            "roadmap.png"
        ]
        
        # Create simple SVG placeholders
        svg_template = '''
<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#f0f0f0" stroke="#ccc" stroke-width="2"/>
  <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-family="Arial, sans-serif" font-size="16">
    {title}
  </text>
  <text x="50%" y="70%" text-anchor="middle" dy=".3em" font-family="Arial, sans-serif" font-size="12" fill="#666">
    Sample Image for Demo
  </text>
</svg>
'''
        
        titles = [
            "Transformer Architecture",
            "Attention Heatmap",
            "Results Chart", 
            "Performance Graph",
            "Research Roadmap"
        ]
        
        for i, image_file in enumerate(image_files):
            image_path = images_dir / image_file
            with open(image_path, 'w', encoding='utf-8') as f:
                f.write(svg_template.format(title=titles[i]))
        
        logger.info(f"✅ Created {len(image_files)} sample images in {images_dir}")
        return str(images_dir)
    
    def step1_parse_document_with_images(self, document_path: str = None) -> str:
        """Step 1: Parse document and extract content with image associations."""
        logger.info("📄 Step 1: Parsing document with image extraction...")
        
        if document_path and Path(document_path).exists():
            # Parse real document
            logger.info(f"Parsing document: {document_path}")
            
            try:
                # Use DataMax to parse document with image extraction
                dm = DataMax(file_path=document_path, to_markdown=True)
                parsed_data = dm.get_data()
                
                if isinstance(parsed_data, list):
                    content = parsed_data[0].get('content', '')
                else:
                    content = parsed_data.get('content', '')
                
                # Save parsed content
                parsed_path = self.output_dir / "parsed_document.md"
                with open(parsed_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"✅ Document parsed and saved to {parsed_path}")
                return str(parsed_path)
                
            except Exception as e:
                logger.warning(f"Failed to parse document {document_path}: {str(e)}")
                logger.info("Falling back to sample content...")
        
        # Create sample content for demonstration
        logger.info("Using sample content for demonstration...")
        sample_path = self.create_sample_markdown_with_images()
        self.create_sample_images_directory()
        
        return sample_path
    
    def step2_clean_content(self, content_path: str) -> str:
        """Step 2: Clean the parsed content."""
        logger.info("🧹 Step 2: Cleaning multimodal content...")
        
        try:
            # Read content
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply cleaning (preserve image references)
            cleaner = AbnormalCleaner(content)
            # Don't remove HTML tags as they might contain image references
            cleaner.convert_newlines()
            cleaner.single_space()
            cleaner.tabs_to_spaces()
            cleaner.simplify_chinese()
            cleaner.remove_invisible_chars()
            
            cleaned_result = cleaner.no_html_clean()  # Use no_html_clean to preserve image tags
            cleaned_content = cleaned_result['text']
            
            # Apply text filtering
            text_filter = TextFilter(cleaned_content)
            if not text_filter.filter_by_char_count(min_chars=100, max_chars=50000):
                logger.warning("Content failed character count filter")
            
            # Save cleaned content
            cleaned_path = self.output_dir / "cleaned_content.md"
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            logger.info(f"✅ Content cleaned and saved to {cleaned_path}")
            return str(cleaned_path)
            
        except Exception as e:
            logger.error(f"❌ Cleaning failed: {str(e)}")
            raise
    
    def step3_extract_image_associations(self, content_path: str) -> list:
        """Step 3: Extract image-text associations."""
        logger.info("🔗 Step 3: Extracting image-text associations...")
        
        try:
            # Parse markdown and associate images
            image_associations = parse_markdown_and_associate_images(content_path)
            
            if not image_associations:
                logger.warning("No image associations found in the content")
                return []
            
            # Save associations for inspection
            associations_path = self.output_dir / "image_associations.json"
            with open(associations_path, 'w', encoding='utf-8') as f:
                json.dump(image_associations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Found {len(image_associations)} image associations")
            logger.info(f"Associations saved to {associations_path}")
            
            # Log association details
            for i, assoc in enumerate(image_associations[:3]):  # Show first 3
                logger.info(f"  Association {i+1}:")
                logger.info(f"    Image: {assoc.get('image_path', 'N/A')}")
                logger.info(f"    Context length: {len(assoc.get('context', ''))} chars")
            
            return image_associations
            
        except Exception as e:
            logger.error(f"❌ Image association extraction failed: {str(e)}")
            raise
    
    def step4_generate_multimodal_qa(self, image_associations: list) -> str:
        """Step 4: Generate multimodal QA pairs."""
        logger.info("🤖 Step 4: Generating multimodal QA pairs...")
        
        # Check API key
        if self.config["api_key"] == "your-api-key-here":
            logger.warning("⚠️ DASHSCOPE_API_KEY not set. Creating mock QA pairs...")
            return self._create_mock_multimodal_qa(image_associations)
        
        try:
            # Generate multimodal QA pairs
            multimodal_qa_results = []
            
            for i, association in enumerate(image_associations):
                logger.info(f"Processing association {i+1}/{len(image_associations)}...")
                
                try:
                    # Generate QA for this image-text pair
                    qa_result = generate_multimodal_qa_with_dashscope(
                        image_path=association.get('image_path'),
                        context=association.get('context', ''),
                        api_key=self.config["api_key"],
                        base_url=self.config["base_url"],
                        model_name=self.config["model_name"]
                    )
                    
                    if qa_result:
                        multimodal_qa_results.append({
                            "association_id": i,
                            "image_path": association.get('image_path'),
                            "qa_pairs": qa_result
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate QA for association {i}: {str(e)}")
                    continue
            
            # Save results
            output_path = self.output_dir / "multimodal_qa_pairs.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(multimodal_qa_results, f, indent=2, ensure_ascii=False)
            
            total_qa_pairs = sum(len(result.get('qa_pairs', [])) for result in multimodal_qa_results)
            logger.info(f"✅ Generated {total_qa_pairs} multimodal QA pairs from {len(multimodal_qa_results)} associations")
            logger.info(f"Results saved to {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Multimodal QA generation failed: {str(e)}")
            raise
    
    def _create_mock_multimodal_qa(self, image_associations: list) -> str:
        """Create mock multimodal QA pairs for demonstration."""
        logger.info("Creating mock multimodal QA pairs for demonstration...")
        
        mock_qa_results = []
        
        for i, association in enumerate(image_associations[:3]):  # Limit to first 3
            mock_qa = {
                "association_id": i,
                "image_path": association.get('image_path'),
                "qa_pairs": [
                    {
                        "question": f"What does the image in {Path(association.get('image_path', '')).name} show?",
                        "answer": f"This image shows content related to the context: {association.get('context', '')[:100]}...",
                        "image_description": f"Mock description for {Path(association.get('image_path', '')).name}"
                    },
                    {
                        "question": f"How does this image relate to the surrounding text?",
                        "answer": "The image provides visual support for the concepts discussed in the text.",
                        "image_description": "Visual representation of the textual content"
                    }
                ]
            }
            mock_qa_results.append(mock_qa)
        
        # Save mock results
        output_path = self.output_dir / "multimodal_qa_pairs.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mock_qa_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created mock multimodal QA pairs: {output_path}")
        return str(output_path)
    
    def generate_summary_report(self, results: dict) -> str:
        """Generate a summary report of the multimodal processing."""
        logger.info("📊 Generating multimodal processing report...")
        
        report = {
            "processing_type": "multimodal",
            "config": self.config,
            "results": results,
            "statistics": {
                "total_images_processed": len(results.get("image_associations", [])),
                "total_qa_pairs": results.get("total_qa_pairs", 0),
                "processing_steps": 4
            },
            "output_files": {
                "parsed_content": results.get("parsed_path"),
                "cleaned_content": results.get("cleaned_path"),
                "image_associations": results.get("associations_path"),
                "multimodal_qa": results.get("qa_path")
            }
        }
        
        # Save report
        report_path = self.output_dir / "multimodal_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Multimodal report saved to {report_path}")
        return str(report_path)
    
    def run_multimodal_pipeline(self, document_path: str = None) -> dict:
        """Run the complete multimodal processing pipeline."""
        logger.info("🚀 Starting multimodal processing pipeline...")
        
        results = {}
        
        try:
            # Step 1: Parse document with images
            results["parsed_path"] = self.step1_parse_document_with_images(document_path)
            
            # Step 2: Clean content
            results["cleaned_path"] = self.step2_clean_content(results["parsed_path"])
            
            # Step 3: Extract image associations
            image_associations = self.step3_extract_image_associations(results["cleaned_path"])
            results["image_associations"] = image_associations
            
            if image_associations:
                # Step 4: Generate multimodal QA
                results["qa_path"] = self.step4_generate_multimodal_qa(image_associations)
                
                # Count total QA pairs
                if Path(results["qa_path"]).exists():
                    with open(results["qa_path"], 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                    results["total_qa_pairs"] = sum(len(item.get('qa_pairs', [])) for item in qa_data)
                else:
                    results["total_qa_pairs"] = 0
            else:
                logger.warning("No image associations found, skipping QA generation")
                results["total_qa_pairs"] = 0
            
            # Generate summary report
            results["report_path"] = self.generate_summary_report(results)
            
            logger.info("🎉 Multimodal processing pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"💥 Multimodal pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the multimodal processing example."""
    
    print("🖼️ DataMax Multimodal Processing Example")
    print("This demo shows how to process documents with images and generate multimodal QA pairs.\n")
    
    # Check API key
    if CONFIG["api_key"] == "your-api-key-here":
        logger.warning("⚠️ DASHSCOPE_API_KEY not found. Will create mock QA pairs for demonstration.")
        logger.info("To use real multimodal QA generation, set your API key:")
        logger.info("export DASHSCOPE_API_KEY='your-actual-api-key'")
    
    # Initialize processor
    processor = MultimodalProcessor(CONFIG)
    
    try:
        # Check if user provided a document
        document_path = CONFIG.get("sample_document")
        if document_path and not Path(document_path).exists():
            logger.info(f"Sample document {document_path} not found. Using demo content.")
            document_path = None
        
        # Run pipeline
        results = processor.run_multimodal_pipeline(document_path)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 MULTIMODAL PROCESSING SUMMARY")
        print("="*60)
        print(f"📊 Total images processed: {len(results.get('image_associations', []))}")
        print(f"❓ Total QA pairs generated: {results.get('total_qa_pairs', 0)}")
        print(f"📁 Output directory: {processor.output_dir}")
        print("\n📋 Generated files:")
        for key, path in results.items():
            if key.endswith('_path') and path:
                print(f"  • {key.replace('_path', '').replace('_', ' ').title()}: {Path(path).name}")
        print("\n🔗 Next steps:")
        print("  1. Review the image associations in image_associations.json")
        print("  2. Check the generated QA pairs in multimodal_qa_pairs.json")
        print("  3. Try with your own documents containing images")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Multimodal processing failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)