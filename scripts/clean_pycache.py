#!/usr/bin/env python3
"""
Script to clean up __pycache__ directories and .pyc/.pyo files.
Usage: python scripts/clean_pycache.py [path]
"""

import os
import shutil
import sys
import argparse


def clean_pycache(path):
    """
    Recursively remove all __pycache__ directories and .pyc/.pyo files
    from the given path.
    """
    removed_dirs = 0
    removed_files = 0

    for root, dirs, files in os.walk(path, topdown=False):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed directory: {pycache_path}")
                removed_dirs += 1
            except OSError as e:
                print(f"Error removing directory {pycache_path}: {e}")

        # Remove .pyc and .pyo files
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                    removed_files += 1
                except OSError as e:
                    print(f"Error removing file {file_path}: {e}")

    print("Cleanup completed!")
    print(f"Removed {removed_dirs} __pycache__ directories and {removed_files} .pyc/.pyo files.")


def main():
    parser = argparse.ArgumentParser(description='Clean __pycache__ directories and .pyc/.pyo files')
    parser.add_argument('path', nargs='?', default='.', help='Path to clean (default: current directory)')

    args = parser.parse_args()
    clean_path = args.path

    if not os.path.exists(clean_path):
        print(f"Error: Path '{clean_path}' does not exist.")
        sys.exit(1)

    clean_pycache(clean_path)


if __name__ == "__main__":
    main()
