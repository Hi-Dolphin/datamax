#!/usr/bin/env python3
"""
Extract Alpaca-format JSONL dataset from a directory of JSONL files.

Reads every JSON line in the provided directory, pulls the `instruction`,
`input`, and `output` fields, and consolidates them into a single JSONL
file suitable for Alpaca-style training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield JSONL files under `root` in sorted order."""
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    for path in sorted(p for p in root.glob(pattern) if p.is_file()):
        yield path


def extract_entries(jsonl_path: Path) -> Iterator[dict[str, str]]:
    """Extract Alpaca fields from a JSONL file."""
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse {jsonl_path}:{line_num}: {exc}"
                ) from exc

            instruction = record.get("instruction", "")
            input_text = record.get("input", "")
            output = record.get("output", "")

            if not instruction and not output:
                # Skip entries that do not contain any meaningful data.
                continue

            yield {
                "instruction": str(instruction),
                "input": str(input_text),
                "output": str(output),
            }


def dump_jsonl(entries: Iterable[dict[str, str]], output_path: Path) -> int:
    """Write entries to a JSONL file."""
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Alpaca fields from JSONL datasets into one file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing source JSONL files.",
        default="/mnt/f/datamax/train/data/Step1/航运行业知识补充-高怡雯/IMO"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("alpaca_dataset.jsonl"),
        help="Path to write the consolidated Alpaca JSONL file "
        "(default: ./alpaca_dataset.jsonl).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for JSONL files under the input directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = args.input_dir
    if not root.exists() or not root.is_dir():
        parser.error(f"Input directory does not exist or is not a directory: {root}")

    jsonl_files = list(iter_jsonl_files(root, args.recursive))
    if not jsonl_files:
        parser.error(f"No JSONL files found under {root}")

    consolidated: list[dict[str, str]] = []
    for path in jsonl_files:
        consolidated.extend(extract_entries(path))

    if not consolidated:
        parser.error("No Alpaca entries were extracted.")

    output_path: Path = args.output
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = dump_jsonl(consolidated, output_path)

    print(
        f"Extracted {total} Alpaca entries from {len(jsonl_files)} files into {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# python examples/scripts/extract_alpaca_jsonl.py train/data -o outputs/alpaca2.jsonl --recursive