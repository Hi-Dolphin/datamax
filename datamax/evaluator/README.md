# DataMax Evaluator Module

The Evaluator module provides a comprehensive toolkit for the quantitative assessment of data quality, with a special focus on multimodal datasets. It allows you to programmatically measure the quality of generated text, the fidelity of images, and the consistency between image-text pairs.

## 核心组件

- **`TextQualityEvaluator`**: 评估生成文本的质量，包括与参考文本的相似度、流畅性和语义多样性。
- **`MultimodalConsistencyEvaluator`**: 评估图像和文本之间的一致性，确保图文对是匹配和相关的。
- **`ImageQualityEvaluator`**: 提供评估图像本身质量的框架（如视觉保真度）。

## ✨ 主要功能

-   **文本质量评估**:
    -   `BERTScore`: 基于 BERT 嵌入的语义相似度评估，关注精确率、召回率和 F1 分数。
    -   `ROUGE` & `BLEU`: 经典的 n-gram 文本相似度指标，分别侧重于召回率和精确率。

-   **多模态一致性评估**:
    -   `CLIPScore`: 利用 CLIP 模型计算图像和文本的相似度得分，是筛选图文对质量的核心指标。
    -   `VQAScore`: (高级) 通过 VQA 模型验证文本描述是否与图像内容一致，提供更强的组合推理评估。

-   **图像质量评估**:
    -   `Visual Fidelity`: 框架性支持，用于未来集成图像清晰度、噪声、对比度等经典指标。

## 📦 安装依赖

`evaluator` 模块依赖一些额外的第三方库。您可以通过以下命令安装它们：

```bash
"module 'cv2' has no attribute 'imshow'"
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

pip install bert-score rouge-score sacrebleu pycocoevalcap torch torchvision transformers magic-pdf==1.2.1