#!/usr/bin/env python3
"""
DataMax Health Check Script

This script performs comprehensive health checks for the DataMax application.
It can be used in Docker containers, monitoring systems, or standalone deployments.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import psutil
except ImportError:
    psutil = None

# Health check configuration
HEALTH_CHECK_CONFIG = {
    'timeout': 30,
    'max_memory_percent': 80,
    'max_cpu_percent': 90,
    'min_disk_space_gb': 1,
    'required_ports': [8000],  # Default DataMax port
    'required_processes': ['datamax'],
    'required_files': [
        'datamax/__init__.py',
        'requirements.txt'
    ],
    'required_directories': [
        'data',
        'logs'
    ]
}


class HealthChecker:
    """Comprehensive health checker for DataMax application."""
    
    def __init__(self, config: Optional[Dict] = None, verbose: bool = False):
        self.config = config or HEALTH_CHECK_CONFIG
        self.verbose = verbose
        self.results = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_result(self, check_name: str, status: str, message: str, 
                   details: Optional[Dict] = None) -> None:
        """Log a health check result."""
        result = {
            'check': check_name,
            'status': status,
            'message': message,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.results.append(result)
        
        if status == 'PASS':
            self.logger.info(f"✓ {check_name}: {message}")
        elif status == 'WARN':
            self.logger.warning(f"⚠ {check_name}: {message}")
        else:  # FAIL
            self.logger.error(f"✗ {check_name}: {message}")
        
        if self.verbose and details:
            self.logger.debug(f"Details: {json.dumps(details, indent=2)}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        try:
            version = sys.version_info
            version_str = f"{version.major}.{version.minor}.{version.micro}"
            
            if version.major == 3 and version.minor >= 10:
                self.log_result(
                    'python_version', 'PASS',
                    f"Python {version_str} is compatible",
                    {'version': version_str, 'required': '>=3.10'}
                )
                return True
            else:
                self.log_result(
                    'python_version', 'FAIL',
                    f"Python {version_str} is not compatible (requires >=3.10)",
                    {'version': version_str, 'required': '>=3.10'}
                )
                return False
        except Exception as e:
            self.log_result(
                'python_version', 'FAIL',
                f"Failed to check Python version: {e}"
            )
            return False
    
    def check_datamax_import(self) -> bool:
        """Check if DataMax can be imported."""
        try:
            import datamax
            version = getattr(datamax, '__version__', 'unknown')
            self.log_result(
                'datamax_import', 'PASS',
                f"DataMax v{version} imported successfully",
                {'version': version}
            )
            return True
        except ImportError as e:
            self.log_result(
                'datamax_import', 'FAIL',
                f"Failed to import DataMax: {e}"
            )
            return False
        except Exception as e:
            self.log_result(
                'datamax_import', 'FAIL',
                f"Unexpected error importing DataMax: {e}"
            )
            return False
    
    def check_required_files(self) -> bool:
        """Check if required files exist."""
        missing_files = []
        existing_files = []
        
        for file_path in self.config.get('required_files', []):
            path = Path(file_path)
            if path.exists():
                existing_files.append(str(path))
            else:
                missing_files.append(str(path))
        
        if missing_files:
            self.log_result(
                'required_files', 'FAIL',
                f"Missing required files: {', '.join(missing_files)}",
                {'missing': missing_files, 'existing': existing_files}
            )
            return False
        else:
            self.log_result(
                'required_files', 'PASS',
                f"All {len(existing_files)} required files exist",
                {'files': existing_files}
            )
            return True
    
    def check_required_directories(self) -> bool:
        """Check if required directories exist."""
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in self.config.get('required_directories', []):
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                existing_dirs.append(str(path))
            else:
                missing_dirs.append(str(path))
        
        if missing_dirs:
            self.log_result(
                'required_directories', 'WARN',
                f"Missing directories (will be created): {', '.join(missing_dirs)}",
                {'missing': missing_dirs, 'existing': existing_dirs}
            )
            # Try to create missing directories
            for dir_path in missing_dirs:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    self.logger.error(f"Failed to create directory {dir_path}: {e}")
            return True
        else:
            self.log_result(
                'required_directories', 'PASS',
                f"All {len(existing_dirs)} required directories exist",
                {'directories': existing_dirs}
            )
            return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        if not psutil:
            self.log_result(
                'disk_space', 'WARN',
                "psutil not available, skipping disk space check"
            )
            return True
        
        try:
            usage = psutil.disk_usage('.')
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            used_percent = (usage.used / usage.total) * 100
            
            min_space = self.config.get('min_disk_space_gb', 1)
            
            if free_gb >= min_space:
                self.log_result(
                    'disk_space', 'PASS',
                    f"Sufficient disk space: {free_gb:.1f}GB free",
                    {
                        'free_gb': round(free_gb, 1),
                        'total_gb': round(total_gb, 1),
                        'used_percent': round(used_percent, 1),
                        'required_gb': min_space
                    }
                )
                return True
            else:
                self.log_result(
                    'disk_space', 'FAIL',
                    f"Insufficient disk space: {free_gb:.1f}GB free (requires {min_space}GB)",
                    {
                        'free_gb': round(free_gb, 1),
                        'total_gb': round(total_gb, 1),
                        'used_percent': round(used_percent, 1),
                        'required_gb': min_space
                    }
                )
                return False
        except Exception as e:
            self.log_result(
                'disk_space', 'FAIL',
                f"Failed to check disk space: {e}"
            )
            return False
    
    def check_memory_usage(self) -> bool:
        """Check memory usage."""
        if not psutil:
            self.log_result(
                'memory_usage', 'WARN',
                "psutil not available, skipping memory check"
            )
            return True
        
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            max_percent = self.config.get('max_memory_percent', 80)
            
            if used_percent <= max_percent:
                self.log_result(
                    'memory_usage', 'PASS',
                    f"Memory usage OK: {used_percent:.1f}% used",
                    {
                        'used_percent': round(used_percent, 1),
                        'available_gb': round(memory.available / (1024**3), 1),
                        'total_gb': round(memory.total / (1024**3), 1),
                        'max_percent': max_percent
                    }
                )
                return True
            else:
                self.log_result(
                    'memory_usage', 'WARN',
                    f"High memory usage: {used_percent:.1f}% used (threshold: {max_percent}%)",
                    {
                        'used_percent': round(used_percent, 1),
                        'available_gb': round(memory.available / (1024**3), 1),
                        'total_gb': round(memory.total / (1024**3), 1),
                        'max_percent': max_percent
                    }
                )
                return True  # Warning, not failure
        except Exception as e:
            self.log_result(
                'memory_usage', 'FAIL',
                f"Failed to check memory usage: {e}"
            )
            return False
    
    def check_cpu_usage(self) -> bool:
        """Check CPU usage."""
        if not psutil:
            self.log_result(
                'cpu_usage', 'WARN',
                "psutil not available, skipping CPU check"
            )
            return True
        
        try:
            # Get CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=1)
            max_percent = self.config.get('max_cpu_percent', 90)
            
            if cpu_percent <= max_percent:
                self.log_result(
                    'cpu_usage', 'PASS',
                    f"CPU usage OK: {cpu_percent:.1f}% used",
                    {
                        'used_percent': round(cpu_percent, 1),
                        'max_percent': max_percent,
                        'cpu_count': psutil.cpu_count()
                    }
                )
                return True
            else:
                self.log_result(
                    'cpu_usage', 'WARN',
                    f"High CPU usage: {cpu_percent:.1f}% used (threshold: {max_percent}%)",
                    {
                        'used_percent': round(cpu_percent, 1),
                        'max_percent': max_percent,
                        'cpu_count': psutil.cpu_count()
                    }
                )
                return True  # Warning, not failure
        except Exception as e:
            self.log_result(
                'cpu_usage', 'FAIL',
                f"Failed to check CPU usage: {e}"
            )
            return False
    
    def check_http_endpoint(self, url: str = None, timeout: int = None) -> bool:
        """Check if HTTP endpoint is responding."""
        if not requests:
            self.log_result(
                'http_endpoint', 'WARN',
                "requests library not available, skipping HTTP check"
            )
            return True
        
        url = url or "http://localhost:8000/health"
        timeout = timeout or self.config.get('timeout', 30)
        
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                self.log_result(
                    'http_endpoint', 'PASS',
                    f"HTTP endpoint responding: {url}",
                    {
                        'url': url,
                        'status_code': response.status_code,
                        'response_time': response.elapsed.total_seconds()
                    }
                )
                return True
            else:
                self.log_result(
                    'http_endpoint', 'FAIL',
                    f"HTTP endpoint returned {response.status_code}: {url}",
                    {
                        'url': url,
                        'status_code': response.status_code,
                        'response_time': response.elapsed.total_seconds()
                    }
                )
                return False
        except requests.exceptions.ConnectionError:
            self.log_result(
                'http_endpoint', 'FAIL',
                f"Cannot connect to HTTP endpoint: {url}",
                {'url': url, 'error': 'connection_error'}
            )
            return False
        except requests.exceptions.Timeout:
            self.log_result(
                'http_endpoint', 'FAIL',
                f"HTTP endpoint timeout: {url}",
                {'url': url, 'timeout': timeout}
            )
            return False
        except Exception as e:
            self.log_result(
                'http_endpoint', 'FAIL',
                f"HTTP endpoint check failed: {e}",
                {'url': url, 'error': str(e)}
            )
            return False
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        required_vars = self.config.get('required_env_vars', [])
        missing_vars = []
        existing_vars = []
        
        for var_name in required_vars:
            if os.getenv(var_name):
                existing_vars.append(var_name)
            else:
                missing_vars.append(var_name)
        
        if missing_vars:
            self.log_result(
                'environment_variables', 'WARN',
                f"Missing environment variables: {', '.join(missing_vars)}",
                {'missing': missing_vars, 'existing': existing_vars}
            )
            return True  # Warning, not failure
        else:
            self.log_result(
                'environment_variables', 'PASS',
                f"All {len(existing_vars)} required environment variables set",
                {'variables': existing_vars}
            )
            return True
    
    def run_all_checks(self, include_http: bool = False, http_url: str = None) -> Tuple[bool, Dict]:
        """Run all health checks."""
        self.logger.info("Starting DataMax health checks...")
        start_time = time.time()
        
        checks = [
            self.check_python_version,
            self.check_datamax_import,
            self.check_required_files,
            self.check_required_directories,
            self.check_disk_space,
            self.check_memory_usage,
            self.check_cpu_usage,
            self.check_environment_variables,
        ]
        
        if include_http:
            checks.append(lambda: self.check_http_endpoint(http_url))
        
        passed = 0
        failed = 0
        warnings = 0
        
        for check in checks:
            try:
                result = check()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"Check failed with exception: {e}")
                failed += 1
        
        # Count warnings
        warnings = sum(1 for r in self.results if r['status'] == 'WARN')
        
        duration = time.time() - start_time
        overall_status = failed == 0
        
        summary = {
            'overall_status': 'HEALTHY' if overall_status else 'UNHEALTHY',
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'total_checks': len(checks),
            'duration_seconds': round(duration, 2),
            'timestamp': time.time(),
            'results': self.results
        }
        
        if overall_status:
            self.logger.info(f"✓ All health checks passed ({passed}/{len(checks)}) in {duration:.2f}s")
        else:
            self.logger.error(f"✗ Health checks failed ({failed}/{len(checks)} failed) in {duration:.2f}s")
        
        return overall_status, summary


def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(
        description="DataMax Health Check Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Basic health check
  %(prog)s --verbose                # Verbose output
  %(prog)s --http                   # Include HTTP endpoint check
  %(prog)s --http --url http://localhost:8080/health
  %(prog)s --json                   # JSON output
  %(prog)s --config config.json    # Custom configuration

Exit codes:
  0: All checks passed
  1: One or more checks failed
  2: Script error
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--http',
        action='store_true',
        help='Include HTTP endpoint health check'
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8000/health',
        help='HTTP endpoint URL to check (default: %(default)s)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='HTTP request timeout in seconds (default: %(default)s)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file (JSON)'
    )
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    config = HEALTH_CHECK_CONFIG.copy()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
                config.update(custom_config)
        except Exception as e:
            print(f"Error loading config file: {e}", file=sys.stderr)
            return 2
    
    # Update timeout from command line
    config['timeout'] = args.timeout
    
    try:
        # Run health checks
        checker = HealthChecker(config=config, verbose=args.verbose)
        success, summary = checker.run_all_checks(
            include_http=args.http,
            http_url=args.url if args.http else None
        )
        
        # Output results
        if args.json:
            print(json.dumps(summary, indent=2))
        
        # Return appropriate exit code
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nHealth check interrupted by user", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Health check failed with error: {e}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())