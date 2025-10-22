"""
Generate QA pairs from text with domain tree labeling.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or provide explicitly.
"""
    # parent_path = "/mnt/f/datamax/data"
    # save_parent_path = "/mnt/f/datamax/train"
    # input_names = ["cargo_type", "continent", "countries", "map_node", "port", "sea_area", "special_region", "trade_area", "vessel_type", "shipping_term"]

import os

from datamax import DataMax

api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
model = os.getenv("QA_MODEL", "YOUR QA MODEL")
root_dir = "/mnt/f/datamax"
save_parent_path = os.path.join(root_dir, "train")
file_paths = [
    os.path.join(root, filename)
    for root, _, files in os.walk("data/Step1")
    for filename in files
]

def main():

    os.makedirs(save_parent_path, exist_ok=True)

    for input_name in file_paths:
        input_path = os.path.join(root_dir, input_name)
        relative_stem = os.path.splitext(input_name)[0]
        save_path = os.path.join(save_parent_path, f"{relative_stem}_train")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        dm = DataMax(file_path=input_path, to_markdown=True)

        data = dm.get_data()

        content = data.get("content")

        qa = dm.get_pre_label(
            content=content,
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            question_number=50,  # question_number_per_chunk
            max_qps=100.0,
            debug=False,
            structured_data=True,  # enable structured output
            auto_self_review_mode=True,
            review_max_qps=100.0
        )

        dm.save_label_data(qa, save_path)
        break

if __name__ == "__main__":
    main()

# nohup python examples/scripts/generate_qa.py > generate_qa.out 2>&1 & echo $! > generate_qa.pid
