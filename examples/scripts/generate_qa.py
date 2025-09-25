"""
Generate QA pairs from text with domain tree labeling.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or provide explicitly.
"""
import os
from datamax import DataMax


def main():
    input_path = "examples/generate/sample_document.md"

    api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
    model = os.getenv("QA_MODEL", "qwen-max")

    dm = DataMax(file_path=input_path, to_markdown=True)
    qa = dm.get_pre_label(
        api_key=api_key,
        base_url=base_url,
        model_name=model,
        question_number=5,
        max_workers=3,
        use_tree_label=True,
        interactive_tree=False,
    )
    dm.save_label_data(qa, "train")
    print("Saved train.jsonl")


if __name__ == "__main__":
    main()

