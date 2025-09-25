"""
Generate multimodal QA pairs from a Markdown file with images.
Requires OPENAI_API_KEY or provide explicitly.
"""
import os
from datamax.generator import generate_multimodal_qa_pairs


def main():
    md_path = "examples/generate/example.md"  # Ensure this MD contains image links
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

    qa = generate_multimodal_qa_pairs(
        file_path=md_path,
        api_key=api_key,
        model_name=model,
        question_number=2,
        max_workers=5,
    )
    print(f"Generated {len(qa)} multimodal QA pairs")


if __name__ == "__main__":
    main()

