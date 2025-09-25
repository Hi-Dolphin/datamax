import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

import requests
from loguru import logger
from PIL import Image

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datamax.crawler import ArxivCrawler
from datamax.evaluator import MultimodalConsistencyEvaluator, TextQualityEvaluator
from datamax.generator.multimodal_qa_generator import (
    generatr_qa_pairs as generate_multimodal_qa_pairs,
)
from datamax.parser import DataMax


class PipelineConfig:
    SEARCH_QUERY = "intermodality shipping"
    MAX_PAPERS_TO_CRAWL = 1

    # Ensure your API Key is set as an environment variable or replace the default value
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/XXXXXX/"
    QA_MODEL_NAME = "qwen-vl-plus"  # Must be a VLLM model

    # MODIFIED: Use the DashScope model name for embeddings
    CLIP_MODEL_NAME = "multimodal-embedding-v1"
    VQA_MODEL_NAME = "qwen2.5-vl-7b-instruct"

    QUESTIONS_PER_CHUNK = 2
    MAX_WORKERS = 4

    CLIP_SCORE_THRESHOLD = 0.2

    OUTPUT_BASE_DIR = Path("intermodality/eva_multimodal")
    RAW_DATA_DIR = OUTPUT_BASE_DIR / "01_raw_data"
    PARSED_MD_DIR = OUTPUT_BASE_DIR / "02_parsed_markdown"
    IMAGES_DIR = OUTPUT_BASE_DIR / "03_images"
    GENERATED_QA_DIR = OUTPUT_BASE_DIR / "04_generated_qa"
    EVALUATED_DATA_DIR = OUTPUT_BASE_DIR / "05_evaluated_data"


class IntermodalityPipeline:
    """
    A complete multimodal data processing pipeline:
    Crawl -> Parse -> Generate -> Evaluate
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        # MODIFIED: Pass API key and URL to the evaluator
        self.evaluator = MultimodalConsistencyEvaluator(
            clip_model_name=str(self.config.CLIP_MODEL_NAME),
            vqa_model_name=str(self.config.VQA_MODEL_NAME),
            dashscope_api_key=self.config.DASHSCOPE_API_KEY,
            dashscope_base_url=self.config.DASHSCOPE_BASE_URL,
        )
        self.text_evaluator = TextQualityEvaluator()  # Initialize TextQualityEvaluator
        self._setup_directories()

    def _setup_directories(self):
        """Create all necessary output directories."""
        logger.info(
            f"Ensuring output directories exist in: {self.config.OUTPUT_BASE_DIR}"
        )
        for path in [
            self.config.RAW_DATA_DIR,
            self.config.PARSED_MD_DIR,
            self.config.IMAGES_DIR,
            self.config.GENERATED_QA_DIR,
            self.config.EVALUATED_DATA_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    async def step_1_crawl_data(self) -> list[Path]:
        """Step 1: Crawl papers from ArXiv and download the actual PDF files."""
        logger.info("--- Step 1: Starting to crawl and download real PDF data ---")
        crawler = ArxivCrawler(
            config={"storage": {"base_path": str(self.config.RAW_DATA_DIR)}}
        )
        try:
            search_results = await crawler.crawl_async(
                self.config.SEARCH_QUERY, max_results=self.config.MAX_PAPERS_TO_CRAWL
            )
            if not search_results or not search_results.get("data"):
                logger.warning("Failed to crawl any papers from ArXiv.")
                return []
            papers = search_results["data"]
            logger.success(
                f"Successfully found {len(papers)} relevant papers. Starting download..."
            )
            downloaded_pdf_paths = []
            for paper in papers:
                pdf_url = paper.get("pdf_url")
                arxiv_id = paper.get("arxiv_id", "unknown_id").replace(".", "_")
                if not pdf_url:
                    logger.warning(
                        f"Paper '{arxiv_id}' does not have a PDF link, skipping."
                    )
                    continue
                pdf_filename = f"{arxiv_id}.pdf"
                local_pdf_path = self.config.RAW_DATA_DIR / pdf_filename
                try:
                    response = requests.get(pdf_url, stream=True, timeout=30)
                    response.raise_for_status()
                    with open(local_pdf_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_pdf_paths.append(local_pdf_path)
                    logger.success(f"Successfully downloaded PDF to: {local_pdf_path}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download '{pdf_url}': {e}")
                except Exception as e:
                    logger.error(
                        f"An unknown error occurred while processing '{local_pdf_path}': {e}"
                    )
            return downloaded_pdf_paths
        except Exception as e:
            logger.error(f"Data crawling step failed: {e}")
            return []

    def step_2_parse_to_multimodal_markdown(self, pdf_paths: list[Path]) -> list[Path]:
        """Step 2: Parse PDFs into Markdown files using absolute paths for images."""
        logger.info(
            "--- Step 2: Starting to parse PDFs into multimodal Markdown with absolute image paths ---"
        )
        md_paths = []
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Parsing: {pdf_path.name}")
                dm = DataMax(
                    file_path=str(pdf_path), use_mineru=True
                )  # use MinerU to parse
                parsed_result = dm.get_data()
                if not parsed_result or not parsed_result.get("content"):
                    logger.warning(
                        f"Parsing failed or content is empty for: {pdf_path.name}"
                    )
                    continue
                md_content = parsed_result["content"]
                md_filename = pdf_path.stem + ".md"
                md_path = self.config.PARSED_MD_DIR / md_filename

                target_image_dir = self.config.IMAGES_DIR / pdf_path.stem
                temp_image_source_dir = Path("__temp__") / "images" / pdf_path.stem
                if temp_image_source_dir.exists():
                    shutil.copytree(
                        temp_image_source_dir, target_image_dir, dirs_exist_ok=True
                    )
                    logger.info(
                        f"Copied images from '{temp_image_source_dir}' to '{target_image_dir}'"
                    )

                def path_replacer(match):
                    original_image_filename = Path(match.group(1)).name
                    absolute_image_path = (
                        (target_image_dir / original_image_filename)
                        .resolve()
                        .as_posix()
                    )
                    return f"![image]({absolute_image_path})"

                image_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
                md_content_corrected = re.sub(image_pattern, path_replacer, md_content)

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content_corrected)
                md_paths.append(md_path)
                logger.success(
                    f"Successfully parsed and saved '{pdf_path.name}' to '{md_path}' with absolute image paths."
                )
            except Exception as e:
                logger.error(
                    f"Failed to parse file '{pdf_path.name}': {e}", exc_info=True
                )
        return md_paths

    def step_3_generate_multimodal_qa(self, md_paths: list[Path]) -> list[dict]:
        """Step 3: Generate multimodal QA pairs directly, as Markdown files now contain absolute image paths."""
        logger.info("--- Step 3: Starting to generate multimodal QA pairs ---")
        all_qa_pairs = []
        for md_path in md_paths:
            try:
                logger.info(f"Generating QA pairs for '{md_path.name}'...")
                qa_pairs = generate_multimodal_qa_pairs(
                    file_path=str(md_path),
                    api_key=self.config.DASHSCOPE_API_KEY,
                    model_name=self.config.QA_MODEL_NAME,
                    question_number=self.config.QUESTIONS_PER_CHUNK,
                    max_workers=self.config.MAX_WORKERS,
                    debug=True,
                )
                if not qa_pairs:
                    logger.warning(f"No QA pairs generated for '{md_path.name}'.")
                    continue
                for pair in qa_pairs:
                    pair["source_file"] = md_path.name
                all_qa_pairs.extend(qa_pairs)
                output_path = self.config.GENERATED_QA_DIR / (md_path.stem + "_qa.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
                logger.success(
                    f"Successfully generated {len(qa_pairs)} QA pairs for '{md_path.name}' and saved."
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate QA pairs for '{md_path.name}': {e}",
                    exc_info=True,
                )
        logger.info(f"Generated a total of {len(all_qa_pairs)} multimodal QA pairs.")
        return all_qa_pairs

    def step_4_evaluate_and_filter(
        self, generated_qa: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Step 4: Quantitatively evaluate and filter the generated QA pairs."""
        logger.info("--- Step 4: Starting data evaluation and filtering ---")
        high_quality_data = []
        evaluation_report = []

        if not generated_qa:
            logger.warning("No data to evaluate.")
            return [], []

        logger.info(
            f"Evaluation threshold (CLIP Score) > {self.config.CLIP_SCORE_THRESHOLD}"
        )

        for i, qa_item in enumerate(generated_qa):
            conversation = qa_item.get("conversations", [])
            if not conversation or len(conversation) < 2:
                continue

            user_message = conversation[0].get("value", "")
            assistant_message = conversation[1].get("value", "")
            images = qa_item.get("images", [])

            if not images:
                logger.warning(
                    f"QA pair #{i+1} is missing images, skipping evaluation."
                )
                continue

            image_path = images[0]
            try:
                question_text = user_message.replace("<image>", "").strip()

                # clipscore [question and answer]
                clip_score_q = self.evaluator.evaluate_clip_score(
                    image_path, question_text
                )
                clip_score_a = self.evaluator.evaluate_clip_score(
                    image_path, assistant_message
                )

                vqa_scores = self.evaluator.evaluate_vqa_score(
                    image_path, [question_text, assistant_message]
                )
                print("vqa_scores", vqa_scores)

                similarity_q = clip_score_q.get("cosine_similarity", 0)
                similarity_a = clip_score_a.get("cosine_similarity", 0)
                avg_similarity = (similarity_q + similarity_a) / 2

                report_entry = {
                    "qa_index": i + 1,
                    "source_file": qa_item.get("source_file"),
                    "image": image_path,
                    "question": question_text,
                    "answer": assistant_message,
                    "question_clip_score": similarity_q,
                    "answer_clip_score": similarity_a,
                    "average_clip_score": avg_similarity,
                    "vqa_score_question": vqa_scores[0] if vqa_scores else "N/A",
                    "vqa_score_answer": vqa_scores[1] if len(vqa_scores) > 1 else "N/A",
                    "passed": avg_similarity > self.config.CLIP_SCORE_THRESHOLD,
                }
                evaluation_report.append(report_entry)

                if report_entry["passed"]:
                    qa_item["evaluation_scores"] = {
                        "question_clip_score": similarity_q,
                        "answer_clip_score": similarity_a,
                        "average_clip_score": avg_similarity,
                        "vqa_score_question": vqa_scores[0] if vqa_scores else "N/A",
                        "vqa_score_answer": (
                            vqa_scores[1] if len(vqa_scores) > 1 else "N/A"
                        ),
                    }
                    high_quality_data.append(qa_item)
                    logger.debug(
                        f"QA #{i+1} PASSED evaluation, average score: {avg_similarity:.4f}"
                    )
                else:
                    logger.debug(
                        f"QA #{i+1} FAILED evaluation, average score: {avg_similarity:.4f}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating QA pair #{i+1}: {e}", exc_info=True)
                evaluation_report.append({"qa_index": i + 1, "error": str(e)})

        report_path = self.config.EVALUATED_DATA_DIR / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved to: {report_path}")

        final_data_path = (
            self.config.EVALUATED_DATA_DIR / "high_quality_multimodal_data.jsonl"
        )
        with open(final_data_path, "w", encoding="utf-8") as f:
            for item in high_quality_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.success(
            f"Saved {len(high_quality_data)} high-quality data entries to: {final_data_path}"
        )

        return high_quality_data, evaluation_report

    async def run(self):
        logger.info(
            "🚀 Starting the intermodality multimodal data generation and validation pipeline..."
        )

        if (
            not self.config.DASHSCOPE_API_KEY
            or self.config.DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY"
        ):
            logger.error(
                "Please set your DASHSCOPE_API_KEY in the script or as an environment variable."
            )
            return

        pdf_files = await self.step_1_crawl_data()
        if not pdf_files:
            logger.error("Pipeline terminated due to data crawling failure.")
            return

        md_files = self.step_2_parse_to_multimodal_markdown(pdf_files)
        if not md_files:
            logger.error("Pipeline terminated due to PDF parsing failure.")
            return

        generated_qa = self.step_3_generate_multimodal_qa(md_files)
        if not generated_qa:
            logger.error("Pipeline terminated because no QA pairs were generated.")
            return

        generated_qa = [
            {
                "conversations": [
                    {"from": "user", "value": "<image>\n图片右下角是什么？"}
                ]
            }
        ]

        final_data, report = self.step_4_evaluate_and_filter(generated_qa)

        logger.info("--- Pipeline Execution Summary ---")
        total_generated = len(generated_qa)
        total_passed = len(final_data)
        pass_rate = (total_passed / total_generated * 100) if total_generated > 0 else 0
        logger.success(f"Generated a total of {total_generated} QA pairs.")
        logger.success(
            f"After evaluation, {total_passed} high-quality QA pairs were selected (Pass rate: {pass_rate:.2f}%)."
        )
        logger.success(
            f"The final dataset is saved in: {self.config.EVALUATED_DATA_DIR}"
        )
        logger.info("🎉 Pipeline execution complete!")


if __name__ == "__main__":
    pipeline = IntermodalityPipeline(config=PipelineConfig())
    asyncio.run(pipeline.run())
