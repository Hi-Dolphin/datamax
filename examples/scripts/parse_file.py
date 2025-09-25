"""
Parse a single file with DataMax and print summary.
"""
from datamax import DataMax


def main():
    # Change to your local file path
    input_path = "examples/parse/sample_document.txt"

    dm = DataMax(file_path=input_path, to_markdown=False)
    result = dm.get_data()

    content = result.get("content", "") if isinstance(result, dict) else ""
    print(f"Extension: {result.get('extension')}")
    print(f"Content length: {len(content)}")
    print(f"Lifecycle events: {len(result.get('lifecycle', []))}")


if __name__ == "__main__":
    main()

