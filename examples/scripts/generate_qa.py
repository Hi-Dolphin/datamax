"""
Generate QA pairs from text with domain tree labeling.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or provide explicitly.
"""
import os
from datamax import DataMax
from loguru import logger

api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
model = os.getenv("QA_MODEL", "YOUR QA MODEL")

def main():
    parent_path = "/mnt/f/datamax/data"
    save_parent_path = "/mnt/f/datamax/train"
    input_names = ["cargo_type"] # "continent", "countries", "port", "special_region", "trade_area", "vessel_type", "sea_area", "map_node" 
    for input_name in input_names:
        input_path = os.path.join(parent_path, input_name) + '.json'

        save_path = os.path.join(save_parent_path, input_name) + '_train'
        # 初始化类
        dm = DataMax(file_path=input_path, to_markdown=False)
        # 解析数据
        data = dm.get_data()
        # 获取纯文本内容
        content = data.get("content")
        # 开始标注
        qa = dm.get_pre_label(
            content=content,
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            question_number=30,  # question_number_per_chunk
            max_workers=1,
            debug=False,
            structured_data=True,  # 结构化数据开启
            auto_self_review_mode=True
        )
        # 保存标注数据
        dm.save_label_data(qa, save_path)

if __name__ == "__main__":
    main()

