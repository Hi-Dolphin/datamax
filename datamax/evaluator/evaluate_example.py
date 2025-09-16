import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import json
from datamax import DataMax
from datamax.evaluator import TextQualityEvaluator, MultimodalConsistencyEvaluator
from datamax.generator.multimodal_qa_generator import parse_markdown_and_associate_images
from PIL import Image

def setup_mock_environment():
    if not os.path.exists("./output/images"):
        os.makedirs("./output/images")
    Image.new('RGB', (100, 100), color = 'red').save('./output/images/sample_image.png')
    
    content = """
# 关于机器人的研究报告

## 摘要
本报告探讨了现代机器人的发展。

![一个在仓库中工作的红色机器人](./images/sample_image.png)
*图 1: 一个用于分拣的红色机器人*

机器人技术结合了计算机科学和工程学。上图展示了一个典型的分拣机器人。
"""
    cleaned_md_path = "./output/cleaned_report.md"
    with open(cleaned_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    return cleaned_md_path


def run_multimodal_pipeline_with_evaluation():
    """完整的多模态数据集生成与评估流程"""
    
    print("--- 阶段 1 & 2: 数据解析与清洗 ---")
    cleaned_md_path = setup_mock_environment()
    print(f"✅ 模拟的源文件已准备就绪: {cleaned_md_path}")
    print("-" * 50)

    print("--- 阶段 3: 图文对构建 ---")
    image_text_pairs = parse_markdown_and_associate_images(cleaned_md_path, chunk_size=500, chunk_overlap=100)
    if not image_text_pairs:
        print("❌ 未能构建图文对。")
        return
    print(f"✅ 成功构建 {len(image_text_pairs)} 个图文对。")
    print("-" * 50)
    
    print("--- 阶段 4: 数据集生成 (模拟) ---")
    generated_dataset = [
        {
            "image_path": image_text_pairs[0]['images'][0],
            "generated_captions": [
                "一个红色的方形机器人在仓库环境中。",
                "图片展示了一个用于包裹分拣的工业机器人。",
                "仓库里有一个红色的机器人。",
            ]
        },
        {
            "image_path": image_text_pairs[0]['images'][0],
            "generated_captions": ["一张蓝色的天空照片。"] # 图文不符
        }
    ]
    print("✅ 成功模拟生成2组待评估数据。")
    print("-" * 50)
    
    print("--- 阶段 5: 数据质量量化 (使用 Evaluator 模块) ---")
    
    text_evaluator = TextQualityEvaluator(lang="zh")
    multimodal_evaluator = MultimodalConsistencyEvaluator()
    
    final_dataset = []
    
    for i, data_entry in enumerate(generated_dataset):
        print(f"\n--- 正在评估第 {i+1} 组数据 ---")
        image_path = data_entry["image_path"]
        captions = data_entry["generated_captions"]
        
        # 5.1 评估文本多样性，避免生成内容单一的数据
        diversity_score = text_evaluator.calculate_self_cider_diversity(captions)
        print(f"  - 文本多样性 (Self-CIDEr): {diversity_score:.4f}")
        if diversity_score < 0.5: # 设定多样性阈值
             print("  ⚠️ 警告: 该组标题语义多样性较低。")

        # 5.2 评估图文一致性，过滤掉“图文不符”的劣质数据
        print("  - 图文一致性 (CLIP Score):")
        high_quality_captions = []
        for caption in captions:
            try:
                clip_score_result = multimodal_evaluator.evaluate_clip_score(image_path, caption)
                similarity = clip_score_result.get("cosine_similarity", 0)
                print(f"    - '{caption[:20]}...': {similarity:.4f}")
                
                # 设定一个阈值来过滤低质量的图文对
                if similarity > 0.2: # 阈值可以根据经验调整
                    high_quality_captions.append(caption)
            except Exception as e:
                print(f"    - 评估 '{caption[:20]}...' 时出错: {e}")

        
        if high_quality_captions:
            print(f"  ✅ 保留了 {len(high_quality_captions)} 个高质量标题。")
            final_dataset.append({
                "image_path": image_path,
                "captions": high_quality_captions
            })
        else:
            print(f"  ❌ 该组数据的所有标题都未能通过一致性检查，将被丢弃。")

    print("\n" + "-" * 50)
    print("--- 评估完成 ---")
    print(f"✅ 最终筛选出 {len(final_dataset)} 组高质量数据。")
    print("最终数据集内容:")
    print(json.dumps(final_dataset, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_multimodal_pipeline_with_evaluation()