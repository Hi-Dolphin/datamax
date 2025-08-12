#!/usr/bin/env python3
"""Test runner script for DataMax project.

This script provides convenient ways to run different types of tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ pytest not found. Please install pytest: pip install pytest")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="DataMax Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run integration tests
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --verbose          # Run with verbose output
  python run_tests.py --fast             # Skip slow tests
  python run_tests.py --module crawler   # Run specific module tests
        """
    )
    
    # Test type options
    parser.add_argument(
        '--unit', action='store_true',
        help='Run unit tests only'
    )
    parser.add_argument(
        '--integration', action='store_true',
        help='Run integration tests'
    )
    parser.add_argument(
        '--network', action='store_true',
        help='Run network-dependent tests'
    )
    parser.add_argument(
        '--slow', action='store_true',
        help='Include slow tests'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Skip slow tests (default behavior)'
    )
    
    # Output options
    parser.add_argument(
        '--coverage', action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Quiet output'
    )
    
    # Test selection
    parser.add_argument(
        '--module', choices=['crawler', 'parser', 'cli'],
        help='Run tests for specific module'
    )
    parser.add_argument(
        '--file', type=str,
        help='Run specific test file'
    )
    parser.add_argument(
        '--test', type=str,
        help='Run specific test function/class'
    )
    
    # Other options
    parser.add_argument(
        '--install-deps', action='store_true',
        help='Install test dependencies before running'
    )
    parser.add_argument(
        '--html-report', action='store_true',
        help='Generate HTML coverage report'
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path('pytest.ini').exists():
        print("❌ Error: pytest.ini not found. Please run from the project root directory.")
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        deps_cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.[test]']
        if not run_command(deps_cmd, "Installing test dependencies"):
            sys.exit(1)
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test path
    if args.module:
        cmd.append(f'tests/test_{args.module}.py')
    elif args.file:
        cmd.append(args.file)
    else:
        cmd.append('tests/')
    
    # Add specific test if provided
    if args.test:
        if '::' not in args.test:
            cmd[-1] += f'::{args.test}'
        else:
            cmd.append(args.test)
    
    # Add markers
    markers = []
    if args.unit:
        markers.append('unit')
    if args.integration:
        markers.append('integration')
    if args.network:
        markers.append('network')
    if not args.slow and not args.integration:
        markers.append('not slow')
    
    if markers:
        cmd.extend(['-m', ' and '.join(markers)])
    
    # Add output options
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')
    
    # Add coverage options
    if args.coverage:
        cmd.extend([
            '--cov=datamax',
            '--cov-report=term-missing'
        ])
        if args.html_report:
            cmd.append('--cov-report=html:htmlcov')
    
    # Add integration test options
    if args.integration:
        cmd.append('--run-integration')
    if args.network:
        cmd.append('--run-network')
    if args.slow:
        cmd.append('--run-slow')
    
    # Run the tests
    success = run_command(cmd, "DataMax Tests")
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        if args.coverage and args.html_report:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("💥 Some tests failed. Check the output above for details.")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()