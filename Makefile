# Makefile for DataMax project
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-unit test-integration test-network test-coverage
.PHONY: lint format type-check security docs clean build publish
.PHONY: setup-dev setup-hooks run-crawler run-parser demo

# Default target
help:
	@echo "DataMax Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install package in production mode"
	@echo "  install-dev      Install package in development mode with all dependencies"
	@echo "  setup-dev        Complete development environment setup"
	@echo "  setup-hooks      Install pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-network     Run network tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-benchmark   Run performance benchmarks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run all linters (flake8, bandit)"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo "  quality          Run all quality checks"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Build Commands:"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  publish          Publish package to PyPI"
	@echo "  publish-test     Publish package to Test PyPI"
	@echo ""
	@echo "Demo Commands:"
	@echo "  demo             Run demo examples"
	@echo "  run-crawler      Run crawler demo"
	@echo "  run-parser       Run parser demo"
	@echo ""
	@echo "Utility Commands:"
	@echo "  status           Show project status"
	@echo "  deps-update      Update dependencies"
	@echo "  deps-check       Check for dependency vulnerabilities"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,docs]"
	pip install -r dev-requirements.txt

setup-dev: install-dev setup-hooks
	@echo "Development environment setup complete!"

setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Testing commands
test:
	python run_tests.py

test-unit:
	pytest tests/ -m "not integration and not network" -v

test-integration:
	pytest tests/ -m integration -v

test-network:
	pytest tests/ -m network -v

test-coverage:
	pytest tests/ --cov=datamax --cov=pydatamax --cov-report=html --cov-report=term-missing

test-benchmark:
	pytest tests/ -m benchmark --benchmark-only

test-all:
	tox

# Code quality commands
lint:
	flake8 datamax pydatamax tests
	@echo "Linting completed!"

format:
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive datamax pydatamax tests
	isort datamax pydatamax tests
	black datamax pydatamax tests
	@echo "Code formatting completed!"

type-check:
	mypy datamax pydatamax --ignore-missing-imports
	@echo "Type checking completed!"

security:
	bandit -r datamax pydatamax
	safety check
	@echo "Security checks completed!"

quality: lint type-check security
	@echo "All quality checks completed!"

# Documentation commands
docs:
	cd docs && make html
	@echo "Documentation built! Open docs/_build/html/index.html"

docs-serve:
	cd docs/_build/html && python -m http.server 8000

docs-clean:
	cd docs && make clean

# Build commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	@echo "Cleanup completed!"

build: clean
	python -m build
	@echo "Package built successfully!"

publish: build
	twine check dist/*
	twine upload dist/*
	@echo "Package published to PyPI!"

publish-test: build
	twine check dist/*
	twine upload --repository testpypi dist/*
	@echo "Package published to Test PyPI!"

# Demo commands
demo: run-crawler run-parser

run-crawler:
	@echo "Running ArXiv crawler demo..."
	datamax crawler arxiv 2301.07041 --output ./demo_data
	@echo "Running web crawler demo..."
	datamax crawler web https://example.com --output ./demo_data

run-parser:
	@echo "Running parser demo..."
	datamax parse demo_data/arxiv_2301.07041.json --output demo_paper.md
	@echo "Demo completed! Check demo_paper.md"

# Utility commands
status:
	@echo "DataMax Project Status"
	@echo "====================="
	@echo "Python version: $$(python --version)"
	@echo "Package version: $$(python -c 'import datamax; print(datamax.__version__)')"
	@echo "Installation path: $$(python -c 'import datamax; print(datamax.__file__)')"
	@echo "Available crawlers:"
	datamax crawler list
	@echo "System status:"
	datamax status

deps-update:
	pip-compile requirements.in
	pip-compile dev-requirements.in
	pip-compile test-requirements.in
	@echo "Dependencies updated!"

deps-check:
	safety check
	pip-audit
	@echo "Dependency security check completed!"

# Development workflow shortcuts
dev-check: format lint type-check test-unit
	@echo "Development checks completed!"

ci-check: lint type-check security test-coverage
	@echo "CI checks completed!"

release-check: clean quality test build
	@echo "Release checks completed!"

# Docker commands (if Docker is available)
docker-build:
	docker build -t datamax:latest .

docker-test:
	docker run --rm datamax:latest python -m pytest tests/

docker-run:
	docker run --rm -it datamax:latest bash

# Performance profiling
profile:
	python -m cProfile -o profile.stats scripts/profile_crawler.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile:
	mprof run scripts/profile_memory.py
	mprof plot

# Git workflow helpers
git-setup:
	git config --local core.hooksPath .githooks
	git config --local commit.template .gitmessage

push-check: dev-check
	git status
	@echo "Ready to push!"

# Environment info
env-info:
	@echo "Environment Information"
	@echo "======================"
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Virtual Environment: $$VIRTUAL_ENV"
	@echo "Working Directory: $$(pwd)"
	@echo "Git Branch: $$(git branch --show-current)"
	@echo "Git Status:"
	git status --porcelain