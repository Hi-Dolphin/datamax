# Contributing to DataMax

We welcome contributions to the DataMax project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be collaborative**: Work together to improve the project
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone has different skill levels and backgrounds

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of web scraping and data processing
- Familiarity with async/await programming

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/datamax.git
   cd datamax
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/Hi-Dolphin/datamax.git
   ```

## Development Setup

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   # Or using the requirements files:
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
# Run tests
python -m pytest

# Check code style
make lint

# Run type checking
mypy datamax/
```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check issues labeled `help wanted` for areas needing assistance
- Comment on the issue to let others know you're working on it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow the coding standards (see below)
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new crawler for academic papers"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `style:` for formatting changes
- `chore:` for maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

### Code Formatting

We use automated formatting tools:

```bash
# Format code with Black
black datamax/

# Sort imports with isort
isort datamax/

# Check with flake8
flake8 datamax/
```

### Type Checking

```bash
# Run mypy for type checking
mypy datamax/
```

### Example Code Style

```python
from typing import Optional, List, Dict, Any
import asyncio
from dataclasses import dataclass

@dataclass
class CrawlerConfig:
    """Configuration for web crawler.
    
    Args:
        max_concurrent: Maximum number of concurrent requests
        delay: Delay between requests in seconds
        timeout: Request timeout in seconds
    """
    max_concurrent: int = 10
    delay: float = 1.0
    timeout: float = 30.0

async def fetch_data(
    url: str, 
    config: CrawlerConfig,
    headers: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """Fetch data from URL with given configuration.
    
    Args:
        url: The URL to fetch data from
        config: Crawler configuration
        headers: Optional HTTP headers
        
    Returns:
        Parsed data or None if failed
        
    Raises:
        CrawlerError: If the request fails
    """
    # Implementation here
    pass
```

## Testing Guidelines

### Test Structure

- Unit tests: Test individual functions and classes
- Integration tests: Test component interactions
- End-to-end tests: Test complete workflows

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock, patch
from datamax.crawlers import ArxivCrawler

class TestArxivCrawler:
    """Test cases for ArxivCrawler."""
    
    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        return ArxivCrawler()
    
    @pytest.mark.asyncio
    async def test_fetch_paper_by_id(self, crawler):
        """Test fetching paper by ArXiv ID."""
        paper_id = "2301.00001"
        result = await crawler.fetch_paper(paper_id)
        
        assert result is not None
        assert result.title
        assert result.authors
        assert result.abstract
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_paper_network_error(self, mock_get, crawler):
        """Test handling of network errors."""
        mock_get.side_effect = aiohttp.ClientError("Network error")
        
        result = await crawler.fetch_paper("invalid-id")
        assert result is None
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_crawlers.py

# Run with coverage
pytest --cov=datamax

# Run only unit tests
pytest -m "not integration and not network"

# Run integration tests
pytest -m integration
```

### Test Markers

- `@pytest.mark.unit`: Unit tests (default)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.network`: Tests requiring network access
- `@pytest.mark.slow`: Slow-running tests

## Documentation

### Docstring Style

Use Google-style docstrings:

```python
def process_data(data: List[Dict], format_type: str = "json") -> str:
    """Process raw data into specified format.
    
    Args:
        data: List of data dictionaries to process
        format_type: Output format ("json", "csv", "xml")
        
    Returns:
        Formatted data as string
        
    Raises:
        ValueError: If format_type is not supported
        ProcessingError: If data processing fails
        
    Example:
        >>> data = [{"title": "Test", "content": "Content"}]
        >>> result = process_data(data, "json")
        >>> print(result)
        '[{"title": "Test", "content": "Content"}]'
    """
```

### README Updates

When adding new features:

1. Update the feature list in README.md
2. Add usage examples
3. Update API documentation
4. Add to CHANGELOG.md

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what changes you made and why
3. **Testing**: Describe how you tested your changes
4. **Documentation**: Note any documentation updates
5. **Breaking Changes**: Highlight any breaking changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. Automated checks must pass (CI/CD)
2. Code review by maintainers
3. Address feedback and update PR
4. Final approval and merge

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment**: Python version, OS, DataMax version
2. **Steps to reproduce**: Clear, step-by-step instructions
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error messages**: Full error messages and stack traces
6. **Code samples**: Minimal code to reproduce the issue

### Bug Report Template

```markdown
**Environment:**
- Python version: 3.10.5
- DataMax version: 0.2.0
- OS: Windows 11

**Steps to reproduce:**
1. Create crawler with config X
2. Call method Y with parameters Z
3. Observe error

**Expected behavior:**
Should return parsed data

**Actual behavior:**
Raises ValueError

**Error message:**
```
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    result = crawler.parse(data)
ValueError: Invalid data format
```

**Code sample:**
```python
from datamax import WebCrawler
crawler = WebCrawler()
result = crawler.parse("invalid data")
```
```

## Feature Requests

### Proposing New Features

1. **Use case**: Describe the problem you're trying to solve
2. **Proposed solution**: How should the feature work?
3. **Alternatives**: What alternatives have you considered?
4. **Implementation**: Any ideas on implementation?

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context, screenshots, or examples.
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions and reviews

### Getting Help

1. Check existing documentation
2. Search existing issues
3. Ask in GitHub Discussions
4. Create a new issue if needed

### Recognition

Contributors are recognized in:

- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes
- GitHub contributor graphs

## Development Workflow

### Daily Development

```bash
# Start development
git checkout main
git pull upstream main
git checkout -b feature/new-feature

# Make changes
# ... edit files ...

# Test changes
make test
make lint

# Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Create pull request on GitHub
```

### Keeping Fork Updated

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

### Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Publish to PyPI

## Advanced Contributing

### Adding New Crawlers

1. Create new crawler class inheriting from `BaseCrawler`
2. Implement required methods
3. Add configuration options
4. Write comprehensive tests
5. Update documentation
6. Add CLI integration

### Adding New Parsers

1. Create parser class inheriting from `BaseParser`
2. Implement parsing logic
3. Add format validation
4. Write tests with sample data
5. Update documentation

### Performance Optimization

1. Profile code with `cProfile`
2. Use async/await for I/O operations
3. Implement caching where appropriate
4. Add benchmarks for critical paths
5. Monitor memory usage

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing issues and discussions
3. Ask in GitHub Discussions
4. Contact maintainers if needed

Thank you for contributing to DataMax! 🚀