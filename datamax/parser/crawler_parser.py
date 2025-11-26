"""Crawler Data Parser

Provides parser implementation for crawler data formats.
Supports parsing data from ArXiv crawler and web crawler.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from datamax.utils.lifecycle_types import LifeType

from .base import BaseLife, MarkdownOutputVo


class CrawlerParser(BaseLife):
    """Parser for crawler data formats.

    Handles data from various crawler sources including ArXiv papers
    and general web pages.
    """

    def __init__(self, file_path: str, domain: str = "Technology", **kwargs):
        """Initialize crawler parser.

        Args:
            file_path: Path to the crawler data file
            domain: Domain category for the data
            **kwargs: Additional arguments
        """
        super().__init__(domain=domain, **kwargs)
        self.file_path = file_path
        self.parsed_data = None
        self.raw_data = None

    def parse(self) -> MarkdownOutputVo:
        """Parse crawler data file.

        Returns:
            MarkdownOutputVo containing parsed content

        Raises:
            Exception: If parsing fails
        """
        try:
            # Load raw data
            self.raw_data = self._load_data()

            # Parse based on data type
            if self._is_arxiv_data():
                markdown_content = self._parse_arxiv_data()
            elif self._is_web_data():
                markdown_content = self._parse_web_data()
            elif self._is_search_results():
                markdown_content = self._parse_search_results()
            else:
                markdown_content = self._parse_generic_data()

            # Create output object
            output = MarkdownOutputVo(extension="crawler", content=markdown_content)

            self.parsed_data = output

            return output

        except Exception as e:
            raise Exception(f"Failed to parse crawler data: {str(e)}") from e

    def _load_data(self) -> Dict[str, Any]:
        """Load data from file.

        Returns:
            Loaded data dictionary

        Raises:
            Exception: If file loading fails
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise Exception(
                f"Failed to load data from {self.file_path}: {str(e)}"
            ) from e

    def _is_arxiv_data(self) -> bool:
        """Check if data is from ArXiv crawler.

        Returns:
            True if data is from ArXiv
        """
        if not self.raw_data:
            return False

        # Check for ArXiv-specific fields
        data_section = self.raw_data.get("data", {})
        if isinstance(data_section, dict):
            return (
                "arxiv_id" in data_section
                or data_section.get("source") == "arxiv"
                or self.raw_data.get("type") == "single_paper"
            )
        elif isinstance(data_section, list) and data_section:
            return data_section[0].get("source") == "arxiv"

        return False

    def _is_web_data(self) -> bool:
        """Check if data is from web crawler.

        Returns:
            True if data is from web crawler
        """
        if not self.raw_data:
            return False

        return (
            self.raw_data.get("type") == "web_page"
            or self.raw_data.get("source") == "web"
        )

    def _is_search_results(self) -> bool:
        """Check if data contains search results.

        Returns:
            True if data contains search results
        """
        if not self.raw_data:
            return False

        return self.raw_data.get("type") == "search_results"

    def _parse_arxiv_data(self) -> str:
        """Parse ArXiv paper data to markdown.

        Returns:
            Markdown formatted content
        """
        data = self.raw_data.get("data", {})

        if self.raw_data.get("type") == "single_paper":
            return self._format_arxiv_paper(data)
        else:
            # Handle list of papers
            papers = data if isinstance(data, list) else [data]
            content_parts = []

            for i, paper in enumerate(papers, 1):
                content_parts.append(f"## Paper {i}\n")
                content_parts.append(self._format_arxiv_paper(paper))
                content_parts.append("\n---\n")

            return "\n".join(content_parts)

    def _format_arxiv_paper(self, paper: Dict[str, Any]) -> str:
        """Format single ArXiv paper to markdown.

        Args:
            paper: Paper data dictionary

        Returns:
            Markdown formatted paper content
        """
        parts = []

        # Title
        title = paper.get("title", "Unknown Title")
        parts.append(f"# {title}\n")

        # Metadata
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id:
            parts.append(f"**ArXiv ID:** {arxiv_id}\n")

        # Authors
        authors = paper.get("authors", [])
        if authors:
            authors_str = ", ".join(authors)
            parts.append(f"**Authors:** {authors_str}\n")

        # Categories
        categories = paper.get("categories", [])
        if categories:
            categories_str = ", ".join(categories)
            parts.append(f"**Categories:** {categories_str}\n")

        # Dates
        published_date = paper.get("published_date")
        if published_date:
            parts.append(f"**Published:** {published_date}\n")

        updated_date = paper.get("updated_date")
        if updated_date and updated_date != published_date:
            parts.append(f"**Updated:** {updated_date}\n")

        # DOI
        doi = paper.get("doi")
        if doi:
            parts.append(f"**DOI:** {doi}\n")

        # PDF URL
        pdf_url = paper.get("pdf_url")
        if pdf_url:
            parts.append(f"**PDF:** [{pdf_url}]({pdf_url})\n")

        # Abstract/Summary
        summary = paper.get("summary", "")
        if summary:
            parts.append("## Abstract\n")
            parts.append(f"{summary}\n")

        # Crawl metadata
        crawled_at = paper.get("crawled_at")
        if crawled_at:
            parts.append(f"\n*Crawled at: {crawled_at}*\n")

        return "\n".join(parts)

    def _parse_web_data(self) -> str:
        """Parse web page data to markdown.

        Returns:
            Markdown formatted content
        """
        parts = []

        parts.append(self._format_web_title())
        parts.append(self._format_web_url())
        parts.append(self._format_web_basic_metadata())
        parts.append(self._format_web_content())
        parts.append(self._format_web_links())
        parts.append(self._format_web_crawl_metadata())

        # Remove None
        parts = [p for p in parts if p]

        return "\n".join(parts)

    def _format_web_title(self) -> str:
        metadata = self.raw_data.get("metadata", {})
        title = metadata.get("title") or metadata.get("og_title") or "Web Page"
        return f"# {title}\n"

    def _format_web_url(self) -> str | None:
        url = self.raw_data.get("url") or self.raw_data.get("original_url")
        return f"**URL:** [{url}]({url})\n" if url else None

    def _format_web_basic_metadata(self) -> str | None:
        metadata = self.raw_data.get("metadata", {})
        parts = []

        desc = metadata.get("description") or metadata.get("og_description")
        if desc:
            parts.append(f"**Description:** {desc}\n")

        author = metadata.get("author")
        if author:
            parts.append(f"**Author:** {author}\n")

        lang = metadata.get("language")
        if lang:
            parts.append(f"**Language:** {lang}\n")

        keywords = metadata.get("keywords", [])
        if keywords:
            parts.append(f"**Keywords:** {', '.join(keywords)}\n")

        return "".join(parts) if parts else None

    def _format_web_content(self) -> str | None:
        text_content = self.raw_data.get("text_content", "")
        if not text_content:
            return None

        return f"## Content\n\n{text_content}\n"

    def _format_web_links(self) -> str | None:
        links = self.raw_data.get("links", [])
        if not links:
            return None

        parts = ["## Links\n"]
        max_links = 20

        for link in links[:max_links]:
            url = link.get("url", "")
            text = link.get("text", "Link")
            if url:
                parts.append(f"- [{text}]({url})\n")

        if len(links) > max_links:
            parts.append(f"\n*... and {len(links) - max_links} more links*\n")

        return "".join(parts)

    def _format_web_crawl_metadata(self) -> str | None:
        crawled_at = self.raw_data.get("crawled_at")
        return f"\n*Crawled at: {crawled_at}*\n" if crawled_at else None

    def _parse_search_results(self) -> str:
        """Parse search results to markdown.

        Returns:
            Markdown formatted content
        """
        parts = []

        # Title
        query = self.raw_data.get("query", "Search Results")
        parts.append(f"# Search Results: {query}\n")

        # Summary
        total_results = self.raw_data.get("total_results", 0)
        parts.append(f"**Total Results:** {total_results}\n")

        target = self.raw_data.get("target")
        if target:
            parts.append(f"**Query:** {target}\n")

        # Results
        data = self.raw_data.get("data", [])
        if data:
            parts.append("## Results\n")

            for i, item in enumerate(data, 1):
                parts.append(f"### Result {i}\n")

                if item.get("source") == "arxiv":
                    parts.append(self._format_arxiv_paper(item))
                else:
                    # Generic result formatting
                    title = item.get("title", f"Result {i}")
                    parts.append(f"**Title:** {title}\n")

                    summary = item.get("summary") or item.get("description")
                    if summary:
                        parts.append(f"**Summary:** {summary}\n")

                parts.append("\n---\n")

        return "\n".join(parts)

    def _parse_generic_data(self) -> str:
        """Parse generic crawler data to markdown.

        Returns:
            Markdown formatted content
        """
        parts = []

        # Title
        parts.append(self._format_title())

        # Target & Source
        parts.append(self._format_target())
        parts.append(self._format_source())

        # Data content
        parts.append(self._format_data())

        # Metadata
        parts.append(self._format_metadata())

        # Remove None entries
        parts = [p for p in parts if p]

        return "\n".join(parts)

    def _format_title(self) -> str:
        data_type = self.raw_data.get("type", "Crawler Data")
        return f"# {data_type.replace('_', ' ').title()}\n"

    def _format_target(self) -> str | None:
        target = self.raw_data.get("target")
        return f"**Target:** {target}\n" if target else None

    def _format_source(self) -> str | None:
        source = self.raw_data.get("source")
        return f"**Source:** {source}\n" if source else None

    def _format_data(self) -> str | None:
        data = self.raw_data.get("data")
        if not data:
            return None

        parts = ["## Data\n"]

        if isinstance(data, str):
            parts.append(f"{data}\n")

        elif isinstance(data, dict):
            parts.extend(self._format_dict_data(data))

        elif isinstance(data, list):
            parts.extend(self._format_list_data(data))

        return "\n".join(parts)

    def _format_dict_data(self, data: dict) -> list[str]:
        parts = []
        for key, value in data.items():
            title = key.replace("_", " ").title()
            if isinstance(value, (str, int, float)):
                parts.append(f"**{title}:** {value}\n")
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(i, str) for i in value)
            ):
                parts.append(f"**{title}:** {', '.join(value)}\n")
        return parts

    def _format_list_data(self, items: list) -> list[str]:
        return [f"{i}. {item}\n" for i, item in enumerate(items, 1)]

    def _format_metadata(self) -> str | None:
        crawled_at = self.raw_data.get("crawled_at")
        return f"\n*Crawled at: {crawled_at}*\n" if crawled_at else None

    def get_parsed_data(self) -> Optional[MarkdownOutputVo]:
        """Get parsed data.

        Returns:
            Parsed data or None if not yet parsed
        """
        return self.parsed_data

    def get_raw_data(self) -> Optional[Dict[str, Any]]:
        """Get raw crawler data.

        Returns:
            Raw data dictionary or None if not loaded
        """
        return self.raw_data
