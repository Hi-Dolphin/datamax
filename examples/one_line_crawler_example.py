#!/usr/bin/env python3
"""
One-line Crawler Example

This example demonstrates how to use the one-line crawler interface
to crawl data from various sources with minimal code.

Usage:
    python one_line_crawler_example.py
"""

import sys
import os
import json
import asyncio

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datamax
from datamax.crawler.storage_adapter import create_storage_adapter


async def save_with_storage_adapter(storage_adapter, data, identifier):
    """Helper function to save data using storage adapter."""
    return await storage_adapter.save(data, identifier)


def main():
    """Demonstrate one-line crawling functionality."""
    print("=== DataMax One-Line Crawler Example ===\n")

    # Create storage adapter using environment variable configuration
    # Default configuration will use local storage in ./output directory
    storage_config = {
        'default_format': os.environ.get('STORAGE_DEFAULT_FORMAT', 'json'),
        'output_dir': os.environ.get('STORAGE_OUTPUT_DIR', './output'),
        'cloud_storage': {
            'enabled': os.environ.get('STORAGE_CLOUD_ENABLED', 'false').lower() == 'true',
            'provider': os.environ.get('STORAGE_CLOUD_PROVIDER', 's3')
        }
    }
    
    try:
        storage_adapter = create_storage_adapter(storage_config)
        print(f"Storage adapter created. Output directory: {storage_config['output_dir']}")
    except Exception as e:
        print(f"Failed to create storage adapter: {e}")
        print("Falling back to JSON file saving...")
        storage_adapter = None

    # Auto mode - use all engines with count parameter
    print("1. Auto mode - using all available engines with count parameter...")
    try:
        result = datamax.crawl("maritime weather routing", engine="auto", count=5)
        print(f"   Success! Found {result.get('total_results', 0)} papers")
        print(f"   Requested count: 5")
        
        # Save using storage adapter or fallback to JSON
        if storage_adapter:
            try:
                # Save using the storage adapter
                file_path = asyncio.run(save_with_storage_adapter(storage_adapter, result, "maritime_weather_routing"))
                print(f"   Results saved using storage adapter to: {file_path}")
            except Exception as e:
                print(f"   Failed to save using storage adapter: {e}")
                # Fallback to JSON file
                output_file = "maritime_weather_routing_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"   Results saved to {output_file} (fallback)")
        else:
            # Fallback to JSON file
            output_file = "maritime_weather_routing_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   Results saved to {output_file}")
        
        # Show first result title
        papers = result.get('data', [])
        if papers:
            print(f"   First paper: {papers[0].get('title', 'No title')}")
        
    except Exception as e:
        print(f"   Failed: {e}\n")
    
    print("=== Example completed ===")


if __name__ == "__main__":
    main()