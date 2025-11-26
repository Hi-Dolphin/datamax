# datamax/generator/multimodal_qa_generator.py

import base64
import json
import math
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

lock = threading.Lock()

_qps_lock = threading.Lock()
_last_request_ts = 0.0


def _consume_futures_tqdm(futures, results):
    with tqdm(
        as_completed(futures), total=len(futures), desc="Generating multimodal QA"
    ) as pbar:
        for future in pbar:
            chunk_output = future.result()
            if chunk_output:
                with lock:
                    results.extend(chunk_output)
                pbar.set_postfix({"Generated": len(results)})


def _log_initial_params(
    debug,
    file_path,
    api_key,
    model_name,
    chunk_size,
    chunk_overlap,
    question_number,
    max_qps,
    kwargs,
):
    if not debug:
        return
    logger.debug("generatr_qa_pairs called with parameters:")
    logger.debug(f"  file_path: {file_path}")
    logger.debug(f"  api_key: {'***' if api_key else None}")
    logger.debug(f"  model_name: {model_name}")
    logger.debug(f"  chunk_size: {chunk_size}")
    logger.debug(f"  chunk_overlap: {chunk_overlap}")
    logger.debug(f"  question_number: {question_number}")
    logger.debug(f"  max_qps: {max_qps}")
    logger.debug(f"  kwargs: {kwargs}")


def _calc_min_interval(max_qps: float) -> float:
    return 1.0 / max_qps if max_qps and max_qps > 0 else 0.0


def _process_all_chunks(
    chunks, api_key, model_name, question_number, min_interval, worker_count, debug
):
    final_results = []
    executor_workers = max(1, worker_count)

    def task(chunk):
        return _process_single_chunk(
            chunk, api_key, model_name, question_number, min_interval, debug
        )

    with ThreadPoolExecutor(max_workers=executor_workers) as executor:
        futures = [executor.submit(task, chunk) for chunk in chunks]

        if debug:
            _consume_futures_debug(futures, final_results)
        else:
            _consume_futures_tqdm(futures, final_results)

    logger.success(
        f"Processing completed! Generated a total of {len(final_results)} multimodal Q&A pairs."
    )
    return final_results


def _process_single_chunk(
    chunk_data, api_key, model_name, question_number, min_interval, debug
):
    context_text = chunk_data["text"]
    images = chunk_data["images"]

    if debug:
        logger.debug(
            f"Processing chunk: text_len={len(context_text)}, image_count={len(images)}"
        )

    instruction_prompt = get_instruction_prompt(question_number)

    _respect_qps_limit(min_interval)

    dialogs = generate_multimodal_qa_with_openai(
        api_key=api_key,
        model=model_name,
        instruction_prompt=instruction_prompt,
        context_text=context_text,
        image_paths=images,
    )

    return _format_dialogs(dialogs, images, debug)


def _format_dialogs(dialogs, images, debug):
    if not dialogs or not isinstance(dialogs, list):
        if debug:
            logger.debug("No valid dialogs returned")
        return []

    qa_pairs = []
    img_tokens = "<image>" * len(images)

    for dialog in dialogs:
        if (
            not isinstance(dialog, dict)
            or "user" not in dialog
            or "assistant" not in dialog
        ):
            continue

        qa_pairs.append(
            {
                "messages": [
                    {"role": "user", "content": img_tokens + dialog["user"]},
                    {"role": "assistant", "content": dialog["assistant"]},
                ],
                "images": images,
            }
        )

    return qa_pairs


def _consume_futures_debug(futures, results):
    for i, future in enumerate(as_completed(futures), 1):
        chunk_output = future.result()
        if chunk_output:
            with lock:
                results.extend(chunk_output)
        logger.debug(
            f"Processed chunk {i}/{len(futures)}; total results={len(results)}"
        )


def _respect_qps_limit(min_interval: float) -> None:
    """Throttle outbound multimodal requests to honor the configured QPS budget."""
    if min_interval <= 0:
        return
    global _last_request_ts
    with _qps_lock:
        now = time.perf_counter()
        wait_time = min_interval - (now - _last_request_ts)
        if wait_time > 0:
            time.sleep(wait_time)
            now = time.perf_counter()
        _last_request_ts = now


def _derive_worker_count(max_qps: float, total_items: int) -> int:
    if total_items <= 0:
        return 0
    if max_qps <= 0:
        return 1
    scaled = max(1, int(math.ceil(max_qps * 3)))
    return max(1, min(total_items, 32, scaled))


from .prompt_templates import get_instruction_prompt


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded


def parse_markdown_and_associate_images(
    md_path: str, chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    """
    Parse Markdown files, extract images, and associate them with text blocks.
    """
    logger.info(f"Starting to parse Markdown file: {md_path}")

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        image_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
        image_paths_original = re.findall(image_pattern, content)

        if not image_paths_original:
            logger.warning(f"No Markdown format image links found in file {md_path}.")
            return []

        logger.info(f"Found {len(image_paths_original)} image links in the file.")

        placeholder_template = "||image_placeholder_{}||"
        path_iter = iter(range(len(image_paths_original)))

        def unique_replacer(match):
            return placeholder_template.format(next(path_iter))

        content_with_unique_placeholders = re.sub(
            image_pattern, unique_replacer, content
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(content_with_unique_placeholders)

        processed_chunks = []
        placeholder_regex = re.compile(r"\|\|image_placeholder_(\d+)\|\|")
        md_dir = os.path.dirname(
            os.path.abspath(os.sep.join(md_path.split(os.sep)[:-1]))
        )

        for chunk_text in chunks:
            found_indices = [int(idx) for idx in placeholder_regex.findall(chunk_text)]
            if not found_indices:
                continue

            clean_chunk_text = re.sub(placeholder_regex, "", chunk_text).strip()
            unique_indices = sorted(list(set(found_indices)))

            chunk_image_paths = [
                os.path.abspath(os.path.join(md_dir, image_paths_original[i]))
                for i in unique_indices
            ]

            processed_chunks.append(
                {"text": clean_chunk_text, "images": chunk_image_paths}
            )

        logger.info(
            f"Successfully parsed and associated {len(processed_chunks)} text blocks containing images."
        )
        return processed_chunks
    except Exception as e:
        logger.error(f"Failed to process Markdown file {md_path}: {e}")

        import traceback

        traceback.print_exc()
        return []


def generate_multimodal_qa_with_openai(
    api_key: str,
    model: str,
    instruction_prompt: str,
    context_text: str,
    image_paths: List[str],
    temperature: float = 0.7,
) -> List[Dict[str, str]]:
    """
    Generate content and parse JSON output using the OpenAI multimodal dialogue API
    """
    try:
        client = OpenAI(api_key=api_key)

        user_content = []
        for path in image_paths:
            base64_image = encode_image_to_base64(path)
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "image/jpeg"  # default
            image_url = f"data:{mime_type};base64,{base64_image}"
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        user_content.append(
            {
                "type": "text",
                "text": f"This is the context text you need to process:\n\n---\n{context_text}\n---",
            }
        )

        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": user_content},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        text_content = response.choices[0].message.content

        if not text_content:
            logger.error("Failed to extract valid text from API return content.")
            return []

        json_match = re.search(r"```json\n([\s\S]*?)\n```", text_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text_content

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nOriginal output: {json_str}")
            return []

    except Exception as e:
        logger.error(f"Exception occurred during LLM API call: {e}")
        import traceback

        traceback.print_exc()
        return []


def generatr_qa_pairs(
    file_path: str,
    api_key: str,
    base_url: str | None = None,
    model_name: str = "gpt-4-vision-preview",
    chunk_size: int = 2000,
    chunk_overlap: int = 300,
    question_number: int = 2,
    max_qps: float = 5.0,
    debug: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generate multimodal QA pairs from a Markdown file with associated images.
    """

    _log_initial_params(
        debug,
        file_path,
        api_key,
        model_name,
        chunk_size,
        chunk_overlap,
        question_number,
        max_qps,
        kwargs,
    )

    chunks_with_images = parse_markdown_and_associate_images(
        file_path, chunk_size, chunk_overlap
    )

    if not chunks_with_images:
        logger.warning(
            "Failed to parse any text blocks containing images from the file."
        )
        return []

    logger.info(
        f"Starting to generate Q&A pairs for {len(chunks_with_images)} text blocks "
        f"(max_qps={max_qps})..."
    )

    min_interval = _calc_min_interval(max_qps)
    worker_count = _derive_worker_count(max_qps, len(chunks_with_images))

    return _process_all_chunks(
        chunks_with_images,
        api_key,
        model_name,
        question_number,
        min_interval,
        worker_count,
        debug,
    )
