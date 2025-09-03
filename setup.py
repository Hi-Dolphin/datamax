#!/usr/bin/env python3
"""
DataMax - Advanced Data Crawling and Processing Framework
Setup configuration for package installation
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're using Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("DataMax requires Python 3.8 or higher")

# Get the long description from README
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py
def get_version():
    """Extract version from datamax/__init__.py"""
    version_file = here / "datamax" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.2.0"  # fallback version

# Core dependencies
core_requirements = [
    "oss2>=2.19.1,<3.0.0",
    "aliyun-python-sdk-core>=2.16.0,<3.0.0",
    "aliyun-python-sdk-kms>=2.16.5,<3.0.0",
    "crcmod>=1.7,<2.0.0",
    "langdetect>=1.0.9,<2.0.0",
    "loguru>=0.7.3,<1.0.0",
    "python-docx>=1.1.2,<2.0.0",
    "python-dotenv>=1.1.0,<2.0.0",
    "pymupdf>=1.24.14,<2.0.0",
    "pypdf>=5.5.0,<6.0.0",
    "openpyxl>=3.1.5,<4.0.0",
    "pandas>=2.2.3,<3.0.0",
    "numpy>=2.2.6,<3.0.0",
    "requests>=2.32.3,<3.0.0",
    "tqdm>=4.67.1,<5.0.0",
    "pydantic>=2.10.6,<3.0.0",
    "pydantic-settings>=2.9.1,<3.0.0",
    "python-magic>=0.4.27,<1.0.0",
    "PyYAML>=6.0.2,<7.0.0",
    "Pillow>=11.2.1,<12.0.0",
    "packaging>=24.2,<25.0",
    "beautifulsoup4>=4.13.4,<5.0.0",
    "minio>=7.2.15,<8.0.0",
    "openai>=1.82.0,<2.0.0",
    "jionlp>=1.5.23,<2.0.0",
    "chardet>=5.2.0,<6.0.0",
    "python-pptx>=1.0.2,<2.0.0",
    "tiktoken>=0.9.0,<1.0.0",
    "markitdown>=0.1.1,<1.0.0",
    "xlrd>=2.0.1,<3.0.0",
    "tabulate>=0.9.0,<1.0.0",
    "unstructured>=0.17.2,<1.0.0",
    "markdown>=3.8,<4.0.0",
    "langchain>=0.3.0,<1.0.0",
    "langchain-community>=0.3.0,<1.0.0",
    "langchain-text-splitters>=0.3.0,<1.0.0",
    "ebooklib==0.19",
    "setuptools",
    "aiohttp>=3.8.0",
    "click>=8.0.0",
    "lxml>=4.9.0",
    "python-dateutil>=2.8.0",
    "typing-extensions>=4.0.0",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-timeout>=2.1.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-timeout>=2.1.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=2.20.0",
]

# Test dependencies
test_requirements = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-timeout>=2.1.0",
    "aioresponses>=0.7.0",
]

# Documentation dependencies
docs_requirements = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "myst-parser>=0.18.0",
    "sphinx-autodoc-typehints>=1.19.0",
]

# All dependencies for complete development setup
all_requirements = list(set(
    core_requirements + 
    dev_requirements + 
    test_requirements + 
    docs_requirements
))

setup(
    # Basic package information
    name="pydatamax",
    version=get_version(),
    description="Advanced Data Crawling and Processing Framework - A library for parsing and converting various file formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author and contact information
    author="ccy",
    author_email="cy.kron@foxmail.com",
    maintainer="DataMax Team",
    maintainer_email="cy.kron@foxmail.com",
    
    # URLs
    url="https://github.com/Hi-Dolphin/datamax",
    project_urls={
        "Homepage": "https://github.com/Hi-Dolphin/datamax",
        "Documentation": "https://github.com/Hi-Dolphin/datamax/docs",
        "Repository": "https://github.com/Hi-Dolphin/datamax",
        "Bug Reports": "https://github.com/Hi-Dolphin/datamax/issues",
        "Source": "https://github.com/Hi-Dolphin/datamax",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_dir={"": "."},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "pydatamax": [
            "py.typed",  # PEP 561 marker file
            "*.json",
            "*.yaml",
            "*.yml",
        ],
    },
    
    # Dependencies
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
        "test": test_requirements,
        "docs": docs_requirements,
        "all": all_requirements,
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "pydatamax=pydatamax.cli.main:main",
        ],
    },
    
    # Classification
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        
        # Topic
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: General",
        "Topic :: Utilities",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        
        # Operating system
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        
        # Framework
        "Framework :: AsyncIO",
        
        # Natural language
        "Natural Language :: English",
        "Natural Language :: Chinese (Simplified)",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "crawler", "scraping", "data-processing", "arxiv", "web-scraping",
        "data-extraction", "parsing", "async", "cli", "framework",
        "academic-papers", "research", "automation", "data-collection",
        "file-conversion", "document-processing"
    ],
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Test configuration
    test_suite="tests",
    tests_require=test_requirements,
)
