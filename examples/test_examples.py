#!/usr/bin/env python3
"""
DataMax Examples Test Suite

This script tests all the examples in the DataMax examples directory
to ensure they work correctly and produce expected outputs.

Usage:
    python test_examples.py
    python test_examples.py --example quick_start
    python test_examples.py --verbose
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from loguru import logger
import tempfile
import shutil

# Add the parent directory to the path so we can import datamax
sys.path.insert(0, str(Path(__file__).parent.parent))


class ExampleTester:
    """Test runner for DataMax examples."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.examples_dir = Path(__file__).parent
        self.test_results = {}
        
        # Configure logger
        if verbose:
            logger.remove()
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.remove()
            logger.add(sys.stdout, level="INFO")
    
    def setup_test_environment(self) -> Path:
        """Set up a temporary test environment."""
        test_dir = Path(tempfile.mkdtemp(prefix="datamax_test_"))
        logger.info(f"Created test environment: {test_dir}")
        return test_dir
    
    def cleanup_test_environment(self, test_dir: Path):
        """Clean up the test environment."""
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up test environment: {test_dir}")
    
    def test_quick_start_example(self) -> dict:
        """Test the quick start example."""
        logger.info("Testing quick_start_example.py...")
        
        test_dir = self.setup_test_environment()
        result = {
            "name": "quick_start_example",
            "status": "unknown",
            "output_dir": None,
            "files_created": [],
            "errors": []
        }
        
        try:
            # Change to test directory
            original_cwd = os.getcwd()
            os.chdir(test_dir)
            
            # Run the example
            cmd = [sys.executable, str(self.examples_dir / "quick_start_example.py")]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if process.returncode == 0:
                result["status"] = "success"
                logger.info("Quick start example completed successfully")
                
                # Check for output files
                output_files = list(test_dir.rglob("*"))
                result["files_created"] = [str(f.relative_to(test_dir)) for f in output_files if f.is_file()]
                
                if self.verbose:
                    logger.debug(f"Output: {process.stdout}")
                    logger.debug(f"Files created: {result['files_created']}")
            else:
                result["status"] = "failed"
                result["errors"].append(f"Exit code: {process.returncode}")
                result["errors"].append(f"Stderr: {process.stderr}")
                logger.error(f"Quick start example failed: {process.stderr}")
            
            os.chdir(original_cwd)
            
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["errors"].append("Test timed out after 5 minutes")
            logger.error("Quick start example timed out")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Quick start example error: {str(e)}")
        finally:
            self.cleanup_test_environment(test_dir)
        
        return result
    
    def test_multimodal_example(self) -> dict:
        """Test the multimodal example."""
        logger.info("Testing multimodal_example.py...")
        
        test_dir = self.setup_test_environment()
        result = {
            "name": "multimodal_example",
            "status": "unknown",
            "output_dir": None,
            "files_created": [],
            "errors": []
        }
        
        try:
            # Change to test directory
            original_cwd = os.getcwd()
            os.chdir(test_dir)
            
            # Run the example
            cmd = [sys.executable, str(self.examples_dir / "multimodal_example.py")]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if process.returncode == 0:
                result["status"] = "success"
                logger.info("Multimodal example completed successfully")
                
                # Check for specific output files
                expected_files = [
                    "multimodal_output/sample_content.md",
                    "multimodal_output/image_associations.json",
                    "multimodal_output/multimodal_qa_pairs.json",
                    "multimodal_output/multimodal_report.json"
                ]
                
                output_files = list(test_dir.rglob("*"))
                result["files_created"] = [str(f.relative_to(test_dir)) for f in output_files if f.is_file()]
                
                # Verify expected files exist
                missing_files = []
                for expected_file in expected_files:
                    if not (test_dir / expected_file).exists():
                        missing_files.append(expected_file)
                
                if missing_files:
                    result["errors"].append(f"Missing expected files: {missing_files}")
                    logger.warning(f"Missing expected files: {missing_files}")
                
                if self.verbose:
                    logger.debug(f"Output: {process.stdout}")
                    logger.debug(f"Files created: {result['files_created']}")
            else:
                result["status"] = "failed"
                result["errors"].append(f"Exit code: {process.returncode}")
                result["errors"].append(f"Stderr: {process.stderr}")
                logger.error(f"Multimodal example failed: {process.stderr}")
            
            os.chdir(original_cwd)
            
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["errors"].append("Test timed out after 5 minutes")
            logger.error("Multimodal example timed out")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Multimodal example error: {str(e)}")
        finally:
            self.cleanup_test_environment(test_dir)
        
        return result
    
    def test_crawler_example(self) -> dict:
        """Test the crawler example."""
        logger.info("Testing crawler_example.py...")
        
        test_dir = self.setup_test_environment()
        result = {
            "name": "crawler_example",
            "status": "unknown",
            "output_dir": None,
            "files_created": [],
            "errors": []
        }
        
        try:
            # Change to test directory
            original_cwd = os.getcwd()
            os.chdir(test_dir)
            
            # Run the example (this might fail due to network issues, which is expected)
            cmd = [sys.executable, str(self.examples_dir / "crawler_example.py")]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for network operations
            )
            
            # For crawler, we're more lenient as it depends on network connectivity
            if process.returncode == 0:
                result["status"] = "success"
                logger.info("Crawler example completed successfully")
                
                # Check for output files
                output_files = list(test_dir.rglob("*"))
                result["files_created"] = [str(f.relative_to(test_dir)) for f in output_files if f.is_file()]
                
                if self.verbose:
                    logger.debug(f"Output: {process.stdout}")
                    logger.debug(f"Files created: {result['files_created']}")
            else:
                # Check if it's a network-related failure
                stderr_lower = process.stderr.lower()
                if any(keyword in stderr_lower for keyword in ['network', 'connection', 'timeout', 'dns', 'http']):
                    result["status"] = "skipped"
                    result["errors"].append("Skipped due to network connectivity issues")
                    logger.warning("Crawler example skipped due to network issues")
                else:
                    result["status"] = "failed"
                    result["errors"].append(f"Exit code: {process.returncode}")
                    result["errors"].append(f"Stderr: {process.stderr}")
                    logger.error(f"Crawler example failed: {process.stderr}")
            
            os.chdir(original_cwd)
            
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["errors"].append("Test timed out after 10 minutes")
            logger.error("Crawler example timed out")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Crawler example error: {str(e)}")
        finally:
            self.cleanup_test_environment(test_dir)
        
        return result
    
    def test_complete_pipeline_example(self) -> dict:
        """Test the complete pipeline example."""
        logger.info("Testing complete_pipeline_example.py...")
        
        test_dir = self.setup_test_environment()
        result = {
            "name": "complete_pipeline_example",
            "status": "unknown",
            "output_dir": None,
            "files_created": [],
            "errors": []
        }
        
        try:
            # Change to test directory
            original_cwd = os.getcwd()
            os.chdir(test_dir)
            
            # Run the example
            cmd = [sys.executable, str(self.examples_dir / "complete_pipeline_example.py")]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if process.returncode == 0:
                result["status"] = "success"
                logger.info("Complete pipeline example completed successfully")
                
                # Check for output files
                output_files = list(test_dir.rglob("*"))
                result["files_created"] = [str(f.relative_to(test_dir)) for f in output_files if f.is_file()]
                
                if self.verbose:
                    logger.debug(f"Output: {process.stdout}")
                    logger.debug(f"Files created: {result['files_created']}")
            else:
                result["status"] = "failed"
                result["errors"].append(f"Exit code: {process.returncode}")
                result["errors"].append(f"Stderr: {process.stderr}")
                logger.error(f"Complete pipeline example failed: {process.stderr}")
            
            os.chdir(original_cwd)
            
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["errors"].append("Test timed out after 10 minutes")
            logger.error("Complete pipeline example timed out")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Complete pipeline example error: {str(e)}")
        finally:
            self.cleanup_test_environment(test_dir)
        
        return result
    
    def test_imports(self) -> dict:
        """Test that all required imports work correctly."""
        logger.info("Testing imports...")
        
        result = {
            "name": "imports",
            "status": "unknown",
            "errors": []
        }
        
        try:
            # Test core imports
            from datamax.parser import DataMax
            from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization
            from datamax.generator import DomainTree, full_qa_labeling_process
            
            # Test backward compatibility imports
            from datamax.utils import AbnormalCleaner as UtilsCleaner
            from datamax.utils import DomainTree as UtilsDomainTree
            
            result["status"] = "success"
            logger.info("All imports successful")
            
        except ImportError as e:
            result["status"] = "failed"
            result["errors"].append(f"Import error: {str(e)}")
            logger.error(f"Import failed: {str(e)}")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Import test error: {str(e)}")
        
        return result
    
    def run_all_tests(self, specific_example: str = None) -> dict:
        """Run all tests or a specific example test."""
        logger.info("Starting DataMax examples test suite...")
        
        tests = {
            "imports": self.test_imports,
            "quick_start": self.test_quick_start_example,
            "multimodal": self.test_multimodal_example,
            "crawler": self.test_crawler_example,
            "complete_pipeline": self.test_complete_pipeline_example
        }
        
        if specific_example:
            if specific_example not in tests:
                logger.error(f"Unknown example: {specific_example}")
                logger.info(f"Available examples: {list(tests.keys())}")
                return {}
            
            logger.info(f"Running specific test: {specific_example}")
            tests = {specific_example: tests[specific_example]}
        
        results = {}
        
        for test_name, test_func in tests.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {str(e)}")
                results[test_name] = {
                    "name": test_name,
                    "status": "crashed",
                    "errors": [str(e)]
                }
        
        return results
    
    def generate_test_report(self, results: dict) -> str:
        """Generate a comprehensive test report."""
        logger.info("Generating test report...")
        
        report = {
            "test_summary": {
                "total_tests": len(results),
                "successful": len([r for r in results.values() if r["status"] == "success"]),
                "failed": len([r for r in results.values() if r["status"] == "failed"]),
                "skipped": len([r for r in results.values() if r["status"] == "skipped"]),
                "errors": len([r for r in results.values() if r["status"] == "error"]),
                "timeouts": len([r for r in results.values() if r["status"] == "timeout"]),
                "crashed": len([r for r in results.values() if r["status"] == "crashed"])
            },
            "test_results": results,
            "recommendations": []
        }
        
        # Add recommendations based on results
        if report["test_summary"]["failed"] > 0:
            report["recommendations"].append("Some tests failed. Check the error messages and ensure all dependencies are installed.")
        
        if report["test_summary"]["skipped"] > 0:
            report["recommendations"].append("Some tests were skipped (likely due to network issues). This is normal for crawler tests.")
        
        if report["test_summary"]["successful"] == report["test_summary"]["total_tests"]:
            report["recommendations"].append("All tests passed! The DataMax examples are working correctly.")
        
        # Save report
        report_path = Path("test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test report saved to {report_path}")
        return str(report_path)
    
    def print_summary(self, results: dict):
        """Print a summary of test results."""
        print("\n" + "="*60)
        print("DATAMAX EXAMPLES TEST SUMMARY")
        print("="*60)
        
        total = len(results)
        successful = len([r for r in results.values() if r["status"] == "success"])
        failed = len([r for r in results.values() if r["status"] == "failed"])
        skipped = len([r for r in results.values() if r["status"] == "skipped"])
        errors = len([r for r in results.values() if r["status"] in ["error", "timeout", "crashed"]])
        
        print(f"Total tests: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {errors}")
        
        print("\nTest Details:")
        for test_name, result in results.items():
            status_emoji = {
                "success": "[PASS]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
                "error": "[ERROR]",
                "timeout": "[TIMEOUT]",
                "crashed": "[CRASH]"
            }.get(result["status"], "[UNKNOWN]")
            
            print(f"  {status_emoji} {test_name}: {result['status'].upper()}")
            
            if result.get("errors") and self.verbose:
                for error in result["errors"]:
                    print(f"    - {error}")
        
        print("\nNext steps:")
        if successful == total:
            print("  All tests passed! Your DataMax examples are ready to use.")
        else:
            print("  1. Check the test report for detailed error information")
            print("  2. Ensure all dependencies are installed: pip install -r requirements.txt")
            print("  3. Check network connectivity for crawler tests")
            print("  4. Run tests individually for debugging: python test_examples.py --example <name>")
        
        print("="*60)


def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description="Test DataMax examples")
    parser.add_argument("--example", help="Run a specific example test", 
                       choices=["imports", "quick_start", "multimodal", "crawler", "complete_pipeline"])
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ExampleTester(verbose=args.verbose)
    
    try:
        # Run tests
        results = tester.run_all_tests(specific_example=args.example)
        
        if not results:
            return 1
        
        # Generate report
        report_path = tester.generate_test_report(results)
        
        # Print summary
        tester.print_summary(results)
        
        # Return appropriate exit code
        failed_tests = [r for r in results.values() if r["status"] in ["failed", "error", "crashed"]]
        return 1 if failed_tests else 0
        
    except Exception as e:
        logger.error(f"Test suite crashed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)