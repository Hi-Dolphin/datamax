"""
Clean text using DataMax cleaner pipeline.
"""

from datamax import DataMax


def main():
    # Use a small text file from examples or replace with your own path
    input_path = "examples/parse/sample_document.txt"

    cleaned = DataMax(file_path=input_path).clean_data(["abnormal", "filter", "private"])
    print(cleaned.get("content", "")[:200])


if __name__ == "__main__":
    main()
