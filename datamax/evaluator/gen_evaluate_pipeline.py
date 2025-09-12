import os
import re
import sys
import json
import asyncio
import requests
from pathlib import Path
from loguru import logger
from PIL import Image
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datamax.crawler import ArxivCrawler
from datamax.parser import DataMax
from datamax.generator.multimodal_qa_generator import generatr_qa_pairs as generate_multimodal_qa_pairs
# from datamax.evaluator import MultimodalConsistencyEvaluator, TextQualityEvaluator


class PipelineConfig:
    SEARCH_QUERY = "intermodality shipping"
    MAX_PAPERS_TO_CRAWL = 1

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-61139bb843e543d9b261c0b366f80c9e")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QA_MODEL_NAME = "qwen-vl-plus" # Must be VLLM model

    QUESTIONS_PER_CHUNK = 2
    MAX_WORKERS = 4

    CLIP_SCORE_THRESHOLD = 0.2

    OUTPUT_BASE_DIR = Path("/workspace/volume/datasets-dev/intermodality/eva_multimodal")
    RAW_DATA_DIR = OUTPUT_BASE_DIR / "01_raw_data"
    PARSED_MD_DIR = OUTPUT_BASE_DIR / "02_parsed_markdown"
    GENERATED_QA_DIR = OUTPUT_BASE_DIR / "03_generated_qa"
    EVALUATED_DATA_DIR = OUTPUT_BASE_DIR / "04_evaluated_data"


class IntermodalityPipeline:
    """
    一个完整的多模态数据处理流水线，包括：
    抓取 -> 解析 -> 生成 -> 评估
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        # self.evaluator = MultimodalConsistencyEvaluator()
        self._setup_directories()

    def _setup_directories(self):
        """创建所有需要的输出目录"""
        logger.info(f"确保输出目录存在于: {self.config.OUTPUT_BASE_DIR}")
        for path in [self.config.RAW_DATA_DIR, self.config.PARSED_MD_DIR,
                     self.config.GENERATED_QA_DIR, self.config.EVALUATED_DATA_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    async def step_1_crawl_data(self) -> list[Path]:
        """步骤 1: 从 ArXiv 抓取关于多式联运的论文并下载真实的PDF文件"""
        logger.info("--- 步骤 1: 开始抓取并下载真实PDF数据 ---")
        crawler = ArxivCrawler(config={'storage': {'base_path': str(self.config.RAW_DATA_DIR)}})
        
        try:
            logger.info(f"正在搜索 ArXiv, 关键词: '{self.config.SEARCH_QUERY}'...")
            search_results = await crawler.crawl_async(
                self.config.SEARCH_QUERY,
                max_results=self.config.MAX_PAPERS_TO_CRAWL
            )
            
            if not search_results or not search_results.get('data'):
                logger.warning("未能从 ArXiv 抓取到任何论文。")
                return []

            papers = search_results['data']
            logger.success(f"成功找到 {len(papers)} 篇相关论文。现在开始下载...")
            
            downloaded_pdf_paths = []
            for paper in papers:
                pdf_url = paper.get('pdf_url')
                arxiv_id = paper.get('arxiv_id', 'unknown_id').replace('.', '_')
                
                if not pdf_url:
                    logger.warning(f"论文 '{arxiv_id}' 没有提供PDF链接，跳过。")
                    continue

                pdf_filename = f"{arxiv_id}.pdf"
                local_pdf_path = self.config.RAW_DATA_DIR / pdf_filename
                
                try:
                    # Download the actual PDF file
                    logger.info(f"正在下载: {pdf_url} -> {local_pdf_path}")
                    response = requests.get(pdf_url, stream=True, timeout=30)
                    response.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)

                    with open(local_pdf_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloaded_pdf_paths.append(local_pdf_path)
                    logger.success(f"成功下载论文PDF到: {local_pdf_path}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"下载 '{pdf_url}' 失败: {e}")
                except Exception as e:
                    logger.error(f"处理 '{local_pdf_path}' 时发生未知错误: {e}")

            return downloaded_pdf_paths
            
        except Exception as e:
            logger.error(f"数据抓取步骤失败: {e}")
            return []


    def step_2_parse_to_multimodal_markdown(self, pdf_paths: list[Path]) -> list[Path]:
        """步骤 2: 将PDF解析为包含图片和文本的Markdown文件"""
        logger.info("--- 步骤 2: 开始解析PDF为多模态Markdown ---")
        md_paths = []
        for pdf_path in pdf_paths:
            try:
                logger.info(f"正在解析: {pdf_path.name}")
                dm = DataMax(
                    file_path=str(pdf_path), 
                    use_qwen_vl_ocr=True,
                    api_key=self.config.DASHSCOPE_API_KEY,
                    base_url=self.config.DASHSCOPE_BASE_URL,
                    model_name=self.config.QA_MODEL_NAME
                )
                parsed_result = dm.get_data()

                if not parsed_result or not parsed_result.get('content'):
                    logger.warning(f"解析失败或内容为空: {pdf_path.name}")
                    continue

                md_content = parsed_result['content']
                md_filename = pdf_path.stem + ".md"
                md_path = self.config.PARSED_MD_DIR / md_filename


                relative_image_dir = Path("images") / pdf_path.stem
                
                def path_replacer(match):
                    image_filename = match.group(1)
                    new_path = (relative_image_dir / image_filename).as_posix()
                    return f"![image]({new_path})"

                image_pattern = r'!\[[^\]]*\]\(([^)]+)\)'
                md_content_corrected = re.sub(image_pattern, path_replacer, md_content)
                
                logger.info("已修正Markdown文件中的图片相对路径。")
                
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                
                md_paths.append(md_path)
                logger.success(f"成功将 '{pdf_path.name}' 解析并保存到 '{md_path}'")
                
                temp_image_dir = Path("__temp__") / "images" / pdf_path.stem
                if temp_image_dir.exists():
                    target_image_dir = self.config.PARSED_MD_DIR / "images" / pdf_path.stem
                    target_image_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(temp_image_dir, target_image_dir, dirs_exist_ok=True)
                    logger.info(f"已将相关图片从 '{temp_image_dir}' 复制到 '{target_image_dir}'")

            except Exception as e:
                logger.error(f"解析文件 '{pdf_path.name}' 失败: {e}")
        
        return md_paths

    def step_3_generate_multimodal_qa(self, md_paths: list[Path]) -> list[dict]:
        """步骤 3: 从Markdown文件生成多模态问答对"""
        logger.info("--- 步骤 3: 开始生成多模态问答对 ---")
        all_qa_pairs = []
        for md_path in md_paths:
            try:
                logger.info(f"正在为 '{md_path.name}' 生成问答对...")
                qa_pairs = generate_multimodal_qa_pairs(
                    file_path=str(md_path),
                    api_key=self.config.DASHSCOPE_API_KEY,
                    model_name=self.config.QA_MODEL_NAME,
                    question_number=self.config.QUESTIONS_PER_CHUNK,
                    max_workers=self.config.MAX_WORKERS,
                    debug=True
                )
                
                if not qa_pairs:
                    logger.warning(f"未能为 '{md_path.name}' 生成任何问答对。")
                    continue

                for pair in qa_pairs:
                    pair['source_file'] = md_path.name
                
                all_qa_pairs.extend(qa_pairs)
                
                output_path = self.config.GENERATED_QA_DIR / (md_path.stem + "_qa.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
                
                logger.success(f"成功为 '{md_path.name}' 生成 {len(qa_pairs)} 个问答对，已保存。")
                
            except Exception as e:
                logger.error(f"为 '{md_path.name}' 生成问答对失败: {e}")

        logger.info(f"总共生成 {len(all_qa_pairs)} 个多模态问答对。")
        return all_qa_pairs

    # def step_4_evaluate_and_filter(self, generated_qa: list[dict]) -> list[dict]:
    #     """步骤 4: 使用量化指标评估并筛选生成的问答对"""
    #     logger.info("--- 步骤 4: 开始评估和筛选数据 ---")
    #     high_quality_data = []
    #     evaluation_report = []

    #     if not generated_qa:
    #         logger.warning("没有可供评估的数据。")
    #         return [], {}

    #     logger.info(f"评估阈值 (CLIP Score) > {self.config.CLIP_SCORE_THRESHOLD}")
        
    #     for i, qa_item in enumerate(generated_qa):
    #         user_message = qa_item['messages'][0]['content']
    #         assistant_message = qa_item['messages'][1]['content']
    #         images = qa_item['images']
            
    #         if not images:
    #             logger.warning(f"问答对 {i+1} 缺少图片，跳过评估。")
    #             continue

    #         # 目前的生成逻辑是一个QA对关联一个或多个图片，我们主要评估和第一张图的一致性
    #         image_path = images[0]
            
    #         try:
    #             # 评估问题与图片的一致性
    #             question_text = user_message.replace("<image>", "").strip()
    #             clip_score_q = self.evaluator.evaluate_clip_score(image_path, question_text)
                
    #             # 评估答案与图片的一致性
    #             clip_score_a = self.evaluator.evaluate_clip_score(image_path, assistant_message)

    #             similarity_q = clip_score_q.get("cosine_similarity", 0)
    #             similarity_a = clip_score_a.get("cosine_similarity", 0)
    #             avg_similarity = (similarity_q + similarity_a) / 2

    #             report_entry = {
    #                 "qa_index": i + 1,
    #                 "source_file": qa_item.get('source_file'),
    #                 "image": image_path,
    #                 "question": question_text,
    #                 "answer": assistant_message,
    #                 "question_clip_score": similarity_q,
    #                 "answer_clip_score": similarity_a,
    #                 "average_clip_score": avg_similarity,
    #                 "passed": avg_similarity > self.config.CLIP_SCORE_THRESHOLD
    #             }
    #             evaluation_report.append(report_entry)

    #             if report_entry["passed"]:
    #                 qa_item['evaluation_scores'] = {
    #                     "question_clip_score": similarity_q,
    #                     "answer_clip_score": similarity_a,
    #                     "average_clip_score": avg_similarity
    #                 }
    #                 high_quality_data.append(qa_item)
    #                 logger.debug(f"QA #{i+1} 通过评估, 平均分: {avg_similarity:.4f}")
    #             else:
    #                 logger.debug(f"QA #{i+1} 未通过评估, 平均分: {avg_similarity:.4f}")

    #         except Exception as e:
    #             logger.error(f"评估问答对 #{i+1} 时出错: {e}")
    #             evaluation_report.append({"qa_index": i + 1, "error": str(e)})

    #     report_path = self.config.EVALUATED_DATA_DIR / "evaluation_report.json"
    #     with open(report_path, "w", encoding="utf-8") as f:
    #         json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
    #     logger.info(f"评估报告已保存到: {report_path}")
        
    #     final_data_path = self.config.EVALUATED_DATA_DIR / "high_quality_multimodal_data.jsonl"
    #     with open(final_data_path, "w", encoding="utf-8") as f:
    #         for item in high_quality_data:
    #             f.write(json.dumps(item, ensure_ascii=False) + "\n")
    #     logger.success(f"筛选出的 {len(high_quality_data)} 条高质量数据已保存到: {final_data_path}")
        
    #     return high_quality_data, evaluation_report

    async def run(self):
        logger.info("🚀 启动多式联运多模态数据生成与验证流水线...")
        
        if self.config.DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY":
            logger.error("请在脚本顶部或环境变量中设置您的 DASHSCOPE_API_KEY。")
            return

        pdf_files = await self.step_1_crawl_data()
        if not pdf_files:
            logger.error("流水线因数据抓取失败而终止。")
            return

        md_files = self.step_2_parse_to_multimodal_markdown(pdf_files)
        if not md_files:
            logger.error("流水线因PDF解析失败而终止。")
            return

        generated_qa = self.step_3_generate_multimodal_qa(md_files)
        if not generated_qa:
            logger.error("流水线因未能生成任何问答对而终止。")
            return

        # final_data, report = self.step_4_evaluate_and_filter(generated_qa)

        logger.info("--- 流水线执行总结 ---")
        total_generated = len(generated_qa)
        # total_passed = len(final_data)
        # pass_rate = (total_passed / total_generated * 100) if total_generated > 0 else 0
        logger.success(f"总共生成 {total_generated} 个问答对。")
        # logger.success(f"经过评估，筛选出 {total_passed} 个高质量问答对 (通过率: {pass_rate:.2f}%)。")
        logger.success(f"最终数据集保存在: {self.config.EVALUATED_DATA_DIR}")
        logger.info("🎉 流水线执行完毕！")


if __name__ == "__main__":
    pipeline = IntermodalityPipeline(config=PipelineConfig())
    asyncio.run(pipeline.run())