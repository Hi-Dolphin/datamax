"""
Crawl a web page or search query and save result to a file.
"""
import json
from datamax.crawler import crawl


def main():
    # URL or keyword
    target = "https://example.com"
    result = crawl(target, engine="web")

    with open("crawl_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Saved crawl_result.json")


if __name__ == "__main__":
    main()

