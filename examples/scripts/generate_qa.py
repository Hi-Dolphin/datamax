"""
Generate QA pairs from text with domain tree labeling.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or provide explicitly.
"""
import os

from datamax import DataMax

api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
model = os.getenv("QA_MODEL", "YOUR QA MODEL")


def main():
    parent_path = "/mnt/f/datamax/data"
    save_parent_path = "/mnt/f/datamax/train"
    input_names = ["cargo_type", "continent", "countries", "map_node", "port", "sea_area", "special_region", "trade_area", "vessel_type", "shipping_term"]
    for input_name in input_names:
        input_path = os.path.join(parent_path, f"{input_name}.json")
        save_path = os.path.join(save_parent_path, f"{input_name}_train")

        dm = DataMax(file_path=input_path, to_markdown=False)

        data = dm.get_data()

        content = data.get("content")

        qa = dm.get_pre_label(
            content=content,
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            question_number=40,  # question_number_per_chunk
            max_workers=10,
            debug=False,
            structured_data=True,  # enable structured output
            auto_self_review_mode=True,
            checkpoint_path="/mnt/f/datamax/checkpoints.jsonl",
            resume_from_checkpoint=True
        )

        dm.save_label_data(qa, save_path)


if __name__ == "__main__":
    main()

# nohup python examples/scripts/generate_qa.py > logs/generate_qa_2.out 2>&1 & echo $! > logs/generate_qa_2.pid