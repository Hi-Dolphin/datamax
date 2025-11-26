import json
import math
import os
import os.path
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveJsonSplitter
from loguru import logger
from tqdm import tqdm

from datamax.utils.performance_monitor import PerformanceMonitor

from .domain_tree import DomainTree  # for cache domain tree
from .prompt_templates import (
    get_system_prompt_for_answer,
    get_system_prompt_for_domain_tree,
    get_system_prompt_for_match_label,
    get_system_prompt_for_question,
    get_system_prompt_for_review,
)

lock = threading.Lock()

DEFAULT_REQUEST_TIMEOUT = 200


DEFAULT_MAX_RETRIES = int(os.getenv("DATAMAX_LLM_MAX_RETRIES", "5"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("DATAMAX_LLM_RETRY_BASE_DELAY", "1.0"))
RETRY_BACKOFF_FACTOR = float(os.getenv("DATAMAX_LLM_BACKOFF_FACTOR", "2.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("DATAMAX_LLM_MAX_BACKOFF_SECONDS", "30"))
MIN_REQUEST_INTERVAL_SECONDS = float(
    os.getenv("DATAMAX_LLM_MIN_INTERVAL_SECONDS", "0.6")
)

_rate_limit_lock = threading.Lock()
_last_request_timestamp = 0.0

QUESTION_STAGE = "question_generation"
ANSWER_STAGE = "answer_generation"
REVIEW_STAGE = "qa_review"


class QAProgressTracker:
    """Handle incremental QA pair persistence and resume."""

    def __init__(self, path: Optional[str], resume: bool = True):
        self.path = path
        self.lock = threading.Lock()
        self.entries_by_key: Dict[str, dict] = {}
        self.order: list[str] = []

        if self.path:
            dir_name = os.path.dirname(self.path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            if resume and os.path.exists(self.path):
                self._load_existing()
            elif not resume and os.path.exists(self.path):
                os.remove(self.path)

    def _make_key(self, entry: dict) -> Optional[str]:
        return entry.get("qid") or entry.get("instruction")

    def _load_existing(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping invalid checkpoint line in {self.path}: {line[:80]}..."
                        )
                        continue
                    key = self._make_key(entry)
                    if not key:
                        continue
                    self.entries_by_key[key] = entry
                    if key not in self.order:
                        self.order.append(key)
            logger.info(
                f"Loaded {len(self.entries_by_key)} checkpointed QA pairs from {self.path}"
            )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"Failed to load checkpoint {self.path}: {exc}")

    def _rewrite_file_locked(self):
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            for key in self.order:
                entry = self.entries_by_key.get(key)
                if entry:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record(self, entry: dict):
        key = self._make_key(entry)
        if not key:
            return
        with self.lock:
            is_new = key not in self.entries_by_key
            self.entries_by_key[key] = entry
            if is_new:
                self.order.append(key)
            if not self.path:
                return
            if is_new:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                self._rewrite_file_locked()

    def existing_answers(self) -> Dict[str, str]:
        answers: Dict[str, str] = {}
        for entry in self.entries_by_key.values():
            question = entry.get("instruction")
            answer = entry.get("output")
            if question and answer is not None:
                answers[question] = answer
        return answers

    def has_entry(self, *, qid: Optional[str], question: str) -> bool:
        key = qid or question
        return bool(key and key in self.entries_by_key)

    def total_entries(self) -> int:
        return len(self.entries_by_key)


def _should_log_debug(debug: bool, attempt: int) -> bool:
    return debug and attempt == 1


def _log_llm_metadata(response: requests.Response, usage: dict) -> None:
    logger.debug("LLM response received")
    logger.debug("-" * 40)
    logger.debug(f"Status code: {response.status_code}")
    if usage:
        logger.debug("Token usage:")
        logger.debug(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
        logger.debug(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        logger.debug(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
    logger.debug("-" * 40)


def _log_llm_raw_output(raw_output: str) -> None:
    logger.debug("Raw LLM response:")
    for line in raw_output.split("\n"):
        if line.strip():
            logger.debug(f"  {line}")
    logger.debug("-" * 40)


def _log_returning_raw_content(raw_output: str) -> None:
    logger.debug(f"Returning raw content (length: {len(raw_output)})")
    logger.debug("=" * 60)


def _batch_iterable(it: Iterable[Any], batch_size: int):
    iterator = iter(it)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def _build_indexed_batches(page_content: list, worker_count: int) -> list:
    total_pages = len(page_content)
    target_batches = max(1, worker_count * 3)
    batch_size = max(1, (total_pages + target_batches - 1) // target_batches)
    return [
        (idx, batch)
        for idx, batch in enumerate(_batch_iterable(page_content, batch_size))
    ]


def _submit_question_jobs(
    executor,
    indexed_batches,
    question_number,
    max_retries,
    api_key,
    model,
    base_url,
    message,
    debug,
    perf_monitor,
    stage_name,
    interval,
):
    return {
        executor.submit(
            _process_question_batch,
            batch_pages,
            question_number,
            max_retries,
            api_key,
            model,
            base_url,
            message,
            debug,
            perf_monitor,
            stage_name,
            interval,
        ): batch_idx
        for batch_idx, batch_pages in indexed_batches
    }


def _collect_results_debug(futures, batch_results):
    for future in as_completed(futures):
        batch_idx = futures[future]
        try:
            batch_results[batch_idx] = future.result()
        except Exception:
            batch_results[batch_idx] = []


def _collect_results_progress(futures, batch_results):
    with tqdm(total=len(futures), desc="Generating questions") as progress_bar:
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results[batch_idx] = future.result()
            except Exception:
                batch_results[batch_idx] = []
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "Generated questions": sum(
                        len(batch or []) for batch in batch_results
                    )
                }
            )


def _record_perf_if_needed(
    perf_monitor: Optional[PerformanceMonitor],
    perf_stage: Optional[str],
    type_str: str,
    messages_payload: list,
    raw_output: str,
    prompt: Union[str, None],
    model: str,
    base_url: str,
    temperature: float,
    top_p: float,
    attempt: int,
    status_code: int,
) -> None:
    if not perf_monitor:
        return

    perf_monitor.record_call(
        stage=perf_stage,
        call_type=type_str or "unknown",
        messages=messages_payload,
        response=raw_output,
        prompt=prompt,
        metadata={
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "top_p": top_p,
            "attempt": attempt,
            "status_code": status_code,
        },
    )


def _extract_pending_items(question_items: list, existing: dict) -> list:
    pending = []
    for item in question_items:
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict question item: {item!r}")
            continue

        question = item.get("question")
        if question is None:
            logger.warning(f"Skipping question item without 'question' key: {item!r}")
            continue

        if question not in existing:
            pending.append(item)

    return pending


def _submit_answer_jobs(
    executor,
    pending_items,
    max_retries,
    api_key,
    model,
    base_url,
    message,
    debug,
    perf_monitor,
    stage_name,
    interval,
):
    return {
        executor.submit(
            _generate_answer_with_retry,
            item,
            max_retries,
            api_key,
            model,
            base_url,
            message,
            debug,
            perf_monitor,
            stage_name,
            interval,
        ): item
        for item in pending_items
    }


def _submit_review_jobs(
    executor,
    qa_pairs,
    source_text,
    api_key,
    model,
    base_url,
    score_threshold,
    max_retries,
    debug,
    user_prompt,
    perf_monitor,
    interval,
):
    return [
        executor.submit(
            _review_single_qa_pair,
            idx,
            qa_pair,
            source_text,
            api_key,
            model,
            base_url,
            score_threshold,
            max_retries,
            debug,
            user_prompt,
            perf_monitor,
            interval,
        )
        for idx, qa_pair in enumerate(qa_pairs, start=1)
    ]


def _log_review_debug(dbg, idx, accepted, score, reason):
    if dbg:
        status = "passed" if accepted else "rejected"
        dbg.log(f"QA pair {idx} {status} (score: {score}): {reason}")


def _handle_review_result(
    result,
    reviewed_qa_pairs,
    rejected_count,
    dbg,
):
    (
        idx,
        qa_pair,
        accepted,
        score,
        reason,
        severity,
        error_message,
    ) = result

    _log_review_debug(dbg, idx, accepted, score, reason)

    if accepted:
        reviewed_qa_pairs.append(qa_pair)
    else:
        rejected_count += 1
        if severity == "warning":
            logger.warning(
                f"Failed to parse review result for QA pair {idx}: {error_message}"
            )
        elif severity == "error":
            logger.error(f"Error reviewing QA pair {idx}: {error_message}")

    return rejected_count


def _update_progress_bar(progress_bar, reviewed_count, rejected_count):
    if progress_bar:
        progress_bar.update(1)
        progress_bar.set_postfix({"passed": reviewed_count, "rejected": rejected_count})


def _respect_rate_limit(min_interval: float):
    if min_interval <= 0:
        return
    global _last_request_timestamp
    with _rate_limit_lock:
        now = time.perf_counter()
        wait_time = min_interval - (now - _last_request_timestamp)
        if wait_time > 0:
            time.sleep(wait_time)
            now = time.perf_counter()
        _last_request_timestamp = now


def _determine_worker_count(
    max_qps: float,
    total_items: int,
    *,
    multiplier: float = 3.0,
    hard_cap: int = 64,
) -> int:
    """
    Derive a practical worker count from the user-specified QPS budget.

    We deliberately avoid exposing raw thread counts. Instead, we size a pool large
    enough to keep the request pipeline saturated for typical I/O bound latencies.
    """
    if total_items <= 0:
        return 0
    if max_qps <= 0:
        return 1
    scaled = max(1, int(math.ceil(max_qps * multiplier)))
    return max(1, min(total_items, hard_cap, scaled))


def _extract_error_detail(response: Optional[requests.Response]) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            message = payload.get("message")
            if not message and isinstance(payload.get("error"), dict):
                message = payload["error"].get("message")
            if message:
                return str(message)
            return json.dumps(payload, ensure_ascii=False)
    except ValueError:
        pass
    return response.text or ""


def _get_retry_after(response: Optional[requests.Response]) -> Optional[float]:
    if response is None:
        return None
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _should_retry(status_code: Optional[int], error_detail: str) -> bool:
    if status_code is None:
        return True
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if status_code == 400 and "rate" in error_detail.lower():
        return True
    return False


def _calculate_retry_delay(
    attempt: int,
    retry_after: Optional[float],
    backoff_factor: float,
) -> float:
    if retry_after is not None:
        return min(max(retry_after, 0.0), MAX_BACKOFF_SECONDS)
    delay = RETRY_BASE_DELAY_SECONDS * (backoff_factor ** (attempt - 1))
    return min(max(delay, 0.5), MAX_BACKOFF_SECONDS)


# ====== API settings======
# set your api key and base url in .env file
API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")


def complete_api_url(base_url: str) -> str:
    """
    Normalize the given base_url so that it ends with the OpenAI-style
    chat completions endpoint.
    E.g. if user passes "https://api.provider.com/v1" it will become
    "https://api.provider.com/v1/chat/completions".
    """
    url = base_url.rstrip("/")
    # If it doesn't end with /chat/completions, append it automatically
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    return url


def load_and_split_markdown(md_path: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Parse Markdown using UnstructuredMarkdownLoader
    Chunking strategy that preserves original paragraph structure

    Args:
        md_path: Path to the markdown file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    try:
        # Use LangChain's MarkdownLoader to load Markdown file
        file_name = os.path.basename(md_path)
        logger.info(f"Starting to split Markdown file: {file_name}")
        loader = UnstructuredMarkdownLoader(md_path)
        documents = loader.load()
        # Further split documents if needed
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        pages = splitter.split_documents(documents)
        page_content = [i.page_content for i in pages]
        logger.info(
            f"馃搫 Markdown file '{file_name}' split into {len(page_content)} chunks"
        )
        return page_content

    except Exception as e:
        logger.error(f"Failed to load {Path(md_path).name}: {str(e)}")
        return []


def load_and_split_text(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    use_mineru: bool = False,
    use_qwen_vl_ocr: bool = False,
) -> list:
    """
    Parse other formats to markdown and split

    Args:
        file_path: Path to the markdown file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        use_mineru: Whether to use MinerU for PDF parsing
        use_qwen_vl_ocr: Whether to use Qwen-VL OCR for PDF parsing

    Returns:
        List of document chunks
    """
    try:
        from datamax.core import DataMax

        # Get file extension for logging
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        logger.info(f"寮€濮嬪鐞嗘枃浠? {file_name} (绫诲瀷: {file_ext})")

        # 浣跨敤DataMax瑙ｆ瀽鏂囦欢锛屼紶閫抲se_mineru鍜寀se_qwen_vl_ocr鍙傛暟
        dm = DataMax(
            file_path=file_path,
            to_markdown=True,
            use_mineru=use_mineru,
            use_qwen_vl_ocr=use_qwen_vl_ocr,
        )
        parsed_data = dm.get_data()

        if not parsed_data:
            logger.error(f"File parsing failed: {file_name}")
            return []

        # Get parsed content
        if isinstance(parsed_data, list):
            # If multiple files, take the first one
            content = parsed_data[0].get("content", "")
        else:
            content = parsed_data.get("content", "")

        if not content:
            logger.error(f"File content is empty: {file_name}")
            return []

        # Use LangChain's text splitter for chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Directly split text content
        page_content = splitter.split_text(content)

        # 鏍规嵁鏂囦欢绫诲瀷鎻愪緵涓嶅悓鐨勬棩蹇椾俊鎭?
        if file_ext == ".pdf":
            if use_qwen_vl_ocr:
                logger.info(
                    f"馃搫 PDF鏂囦欢 '{file_name}' 浣跨敤Qwen-VL OCR瑙ｆ瀽锛岃鍒嗚В涓?{len(page_content)} 涓猚hunk"
                )
            elif use_mineru:
                logger.info(
                    f"馃搫 PDF鏂囦欢 '{file_name}' 浣跨敤MinerU瑙ｆ瀽锛岃鍒嗚В涓?{len(page_content)} 涓猚hunk"
                )
            else:
                logger.info(
                    f"馃搫 PDF file '{file_name}' parsed with PyMuPDF, split into {len(page_content)} chunks"
                )
        else:
            logger.info(
                f"馃搫 {file_ext.upper()} file '{file_name}' split into {len(page_content)} chunks"
            )

        return page_content

    except Exception as e:
        logger.error(f"Failed to process file {Path(file_path).name}: {str(e)}")
        return []


# ------------llm generator-------------------
def extract_json_from_llm_output(output: str):
    """
    Extract JSON content from LLM output, handling multiple possible formats

    Args:
        output: Raw output string from LLM

    Returns:
        Parsed JSON list if successful, None otherwise
    """
    # Try to parse the entire output directly
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Try to extract content wrapped in ```json ```
    json_match = re.search(r"```json\n([\s\S]*?)\n```", output)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    # Try to extract the most JSON-like part
    json_start = output.find("[")
    json_end = output.rfind("]") + 1
    if json_start != -1 and json_end != 0:
        try:
            return json.loads(output[json_start:json_end])
        except json.JSONDecodeError:
            pass

    logger.error(f"Model output not in standard format: {output}")
    return None


def _prepare_llm_messages(prompt: Union[str, None], message: Union[list, None]) -> list:
    if message:
        return list(message)
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Please follow the generation instructions strictly.",
        },
    ]


def _log_llm_request(
    debug: bool,
    attempt: int,
    model: str,
    base_url: str,
    temperature: float,
    top_p: float,
    type_str: str,
    messages_payload: list,
):
    if debug and attempt == 1:
        logger.debug("=" * 60)
        logger.debug("LLM request details")
        logger.debug("=" * 60)
        logger.debug(f"Model: {model}")
        logger.debug(f"API URL: {base_url}")
        logger.debug(f"Temperature: {temperature}")
        logger.debug(f"Top-P: {top_p}")
        logger.debug(f"Request type: {type_str}")
        logger.debug("-" * 40)
        logger.debug("Messages:")
        for idx, msg in enumerate(messages_payload, 1):
            logger.debug(f"{idx}. {msg['role'].upper()}:")
            content_value = msg.get("content", "")
            if not isinstance(content_value, str):
                content_value = str(content_value)
            content_lines = content_value.split("\n")
            for line in content_lines:
                if line.strip():
                    logger.debug(f"   {line}")
        logger.debug("-" * 40)
        logger.debug("Sending request to LLM...")


def process_questions(
    api_key: str,
    model: str,
    base_url: str,
    page_content: list,
    question_number: int,
    max_qps: float = 5.0,
    message: list = None,
    max_retries: int = 10,
    debug: bool = False,
    perf_monitor: Optional[PerformanceMonitor] = None,
) -> list:
    """Generate questions using multi-threading with retry mechanism (refactored to reduce complexity)."""

    message = message or []
    stage_name = QUESTION_STAGE
    stage_ctx = perf_monitor.stage(stage_name) if perf_monitor else nullcontext()
    interval = 1.0 / max_qps if max_qps and max_qps > 0 else 0.0

    if not page_content:
        return []

    worker_count = _determine_worker_count(max_qps, len(page_content))
    worker_count = max(1, worker_count)
    logger.info(
        f"Starting question generation (max_qps={max_qps}, worker_count={worker_count}, "
        f"retries: {max_retries})..."
    )

    indexed_batches = _build_indexed_batches(page_content, worker_count)
    batch_results = [None] * len(indexed_batches)

    with stage_ctx:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = _submit_question_jobs(
                executor,
                indexed_batches,
                question_number,
                max_retries,
                api_key,
                model,
                base_url,
                message,
                debug,
                perf_monitor,
                stage_name,
                interval,
            )

            if debug:
                _collect_results_debug(futures, batch_results)
            else:
                _collect_results_progress(futures, batch_results)

    ordered_results = [q for batch in batch_results if batch for q in batch]

    if perf_monitor:
        perf_monitor.add_stage_items(stage_name, len(ordered_results))

    return ordered_results


def _prepare_request_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _prepare_request_body(model, messages_payload, temperature, top_p):
    return {
        "model": model,
        "messages": messages_payload,
        "temperature": temperature,
        "top_p": top_p,
    }


def _record_perf_for_success(perf_monitor, perf_stage, call_start, response):
    if not perf_monitor:
        return

    result = response.json()
    usage = result.get("usage") or {}

    perf_monitor.record_request(
        stage=perf_stage,
        duration_seconds=time.perf_counter() - call_start,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _record_no_choices(
    perf_monitor,
    perf_stage,
    type,
    messages_payload,
    prompt,
    model,
    base_url,
    temperature,
    top_p,
    attempt,
    response,
):
    if not perf_monitor:
        return

    perf_monitor.record_call(
        stage=perf_stage,
        call_type=type or "unknown",
        messages=messages_payload,
        response="",
        prompt=prompt,
        metadata={
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "top_p": top_p,
            "attempt": attempt,
            "status_code": response.status_code,
            "note": "No choices returned",
        },
    )


def _record_perf_for_error(
    perf_monitor,
    perf_stage,
    type,
    messages_payload,
    prompt,
    model,
    base_url,
    temperature,
    top_p,
    attempt,
    status_code,
    error_message,
):
    if not perf_monitor:
        return

    perf_monitor.record_call(
        stage=perf_stage,
        call_type=type or "unknown",
        messages=messages_payload or [],
        response="",
        prompt=prompt,
        metadata={
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "top_p": top_p,
            "attempt": attempt,
            "status_code": status_code,
            "error": error_message,
        },
    )


def _extract_exception_info(e):
    response = getattr(e, "response", None)
    status_code = response.status_code if response is not None else None

    if isinstance(e, requests.exceptions.HTTPError):
        detail = _extract_error_detail(response)
        error_message = f"HTTP {status_code}: {detail}"
    else:
        detail = str(e)
        error_message = detail

    return response, status_code, detail, error_message


def _should_retry_attempt(e, status_code, error_detail):
    if isinstance(e, requests.exceptions.HTTPError):
        return _should_retry(status_code, error_detail)
    return True


def llm_generator(
    api_key: str,
    model: str,
    base_url: str,
    prompt: Union[str, None] = None,
    type: str = "normal",
    message: Union[list, None] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    debug: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_factor: float = RETRY_BACKOFF_FACTOR,
    min_interval_seconds: Union[float, None] = None,
    perf_monitor: Optional[PerformanceMonitor] = None,
    perf_stage: Optional[str] = None,
) -> list:
    """Generate content using LLM API with automatic throttling and retries."""

    attempts = max(1, max_retries)
    request_interval = (
        min_interval_seconds
        if min_interval_seconds is not None
        else MIN_REQUEST_INTERVAL_SECONDS
    )
    last_error_message = ""

    for attempt in range(1, attempts + 1):
        call_start = time.perf_counter()

        if request_interval:
            _respect_rate_limit(request_interval)

        messages_payload = _prepare_llm_messages(prompt, message)
        _log_llm_request(
            debug, attempt, model, base_url, temperature, top_p, type, messages_payload
        )

        try:
            response = requests.post(
                base_url,
                headers=_prepare_request_headers(api_key),
                json=_prepare_request_body(model, messages_payload, temperature, top_p),
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            _record_perf_for_success(perf_monitor, perf_stage, call_start, response)

            output = _process_llm_response(
                response,
                debug,
                attempt,
                type,
                perf_monitor,
                perf_stage,
                messages_payload,
                prompt,
                model,
                base_url,
                temperature,
                top_p,
            )

            if output is not None:
                return output

            _record_no_choices(
                perf_monitor,
                perf_stage,
                type,
                messages_payload,
                prompt,
                model,
                base_url,
                temperature,
                top_p,
                attempt,
                response,
            )

            if debug and attempt == 1:
                logger.debug("No valid choices returned by LLM")
                logger.debug("=" * 60)

            return []

        except (
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
            Exception,
        ) as e:
            response, status_code, detail, last_error_message = _extract_exception_info(
                e
            )

            _record_perf_for_error(
                perf_monitor,
                perf_stage,
                type,
                messages_payload,
                prompt,
                model,
                base_url,
                temperature,
                top_p,
                attempt,
                status_code,
                last_error_message,
            )

            should_retry = _should_retry_attempt(e, status_code, detail)
            if should_retry and attempt < attempts:
                wait_time = _calculate_retry_delay(
                    attempt, _get_retry_after(response), retry_backoff_factor
                )
                logger.warning(
                    f"LLM request failed ({last_error_message}); retrying in {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                continue

            logger.error(f"LLM request failed: {last_error_message}")
            if getattr(e, "__traceback__", None):
                logger.error(f"Error line number: {e.__traceback__.tb_lineno}")
            return []

    logger.error(f"LLM request failed after {attempts} attempts: {last_error_message}")
    return []


# ------------thread_process-------------
def process_match_tags(
    api_key: str,
    model: str,
    base_url: str,
    questions: list,
    tags_json: list,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_qps: float = 3.0,
    debug: bool = False,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not questions:
        logger.info("No questions supplied for tag matching; skipping")
        return []

    interval = 1.0 / max_qps if max_qps and max_qps > 0 else 0.0
    worker_count = _determine_worker_count(max_qps, len(questions))
    logger.info(
        "Starting concurrent question-tag matching... "
        f"(max_qps={max_qps}, worker_count={worker_count})"
    )
    results = []

    def match_one_question(q):
        prompt = get_system_prompt_for_match_label(tags_json, [q])
        match = llm_generator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            prompt=prompt,
            type="question",
            debug=debug,
            min_interval_seconds=interval,
        )
        return match[0] if match else {"question": q, "label": "鍏朵�?"}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_q = {executor.submit(match_one_question, q): q for q in questions}
        for future in as_completed(future_to_q):
            res = future.result()
            # print(f"Question: {res.get('question', '')} | Matched label: {res.get('label', '')}")
            results.append(res)
    logger.success(
        f"Question-tag matching completed successfully, generated {len(results)} questions"
    )
    return results


def _attempt_domain_tree_generation(
    api_key: str,
    model: str,
    base_url: str,
    prompt: str,
    temperature: float,
    top_p: float,
    debug: bool,
) -> Optional[DomainTree]:
    message = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Please analyze the document and return a structured domain tree in JSON.",
        },
    ]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": message,
        "temperature": temperature,
        "top_p": top_p,
    }
    response = requests.post(
        base_url,
        headers=headers,
        json=data,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()

    if debug:
        logger.debug(f"API Response Status: {response.status_code}")
        if "usage" in result:
            logger.debug(f"Token Usage: {result['usage']}")

    if "choices" in result and len(result["choices"]) > 0:
        output = result["choices"][0]["message"]["content"]
        if debug:
            logger.debug(f"Raw Response: {output[:500]}...")
        if output:
            json_output = extract_json_from_llm_output(output)
            if debug:
                logger.debug(f"Parsed JSON: {json_output}")
            if json_output is not None:
                domain_tree = DomainTree()
                domain_tree.from_json(json_output)
                if debug:
                    logger.debug(f"Generated Domain Tree: {domain_tree.visualize()}")
                logger.info(
                    f"Domain tree generated successfully, created {len(json_output)} main tags"
                )
                return domain_tree
    return None


def process_domain_tree(
    api_key: str,
    model: str,
    base_url: str,
    text: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_retries: int = 10,
    debug: bool = False,
) -> DomainTree:
    prompt = get_system_prompt_for_domain_tree(text)
    logger.info("Domain tree generation started...")

    if debug:
        logger.debug("=" * 80)
        logger.debug("馃尦 DOMAIN TREE GENERATION DEBUG INFO")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt: {prompt[:200]}...")
        logger.debug(f"Model: {model}")
        logger.debug(f"API URL: {base_url}")
        logger.debug(f"Temperature: {temperature}")
        logger.debug(f"Top-P: {top_p}")
        logger.debug(f"Max Retries: {max_retries}")
        logger.debug("=" * 80)

    for attempt in range(max_retries):
        try:
            domain_tree = _attempt_domain_tree_generation(
                api_key, model, base_url, prompt, temperature, top_p, debug
            )
            if domain_tree:
                return domain_tree

            logger.warning(
                f"Domain tree generation failed (attempt {attempt + 1}/{max_retries}): Invalid response"
            )

        except Exception as e:
            logger.error(
                f"Domain tree generation error (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                logger.error(f"Error line number: {e.__traceback__.tb_lineno}")

            if attempt == max_retries - 1:
                break

            logger.info(f"Waiting for retry... ({attempt + 2}/{max_retries})")
            time.sleep(2)

    error_msg = "Tree generation failed! Please check network or switch LLM model! Will continue with plain text generation"
    logger.error(
        f"Domain tree generation failed after {max_retries} retries: {error_msg}"
    )
    return None


def _generate_questions_with_retry(
    page: str,
    question_number: int,
    max_retries: int,
    api_key: str,
    model: str,
    base_url: str,
    message: list,
    debug: bool,
    perf_monitor: Optional[PerformanceMonitor],
    stage_name: str,
    interval: float,
) -> List[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            prompt = get_system_prompt_for_question(page, question_number)
            questions = llm_generator(
                api_key=api_key,
                model=model,
                base_url=base_url,
                message=message,
                prompt=prompt,
                type="question",
                debug=debug,
                perf_monitor=perf_monitor,
                perf_stage=stage_name,
                min_interval_seconds=interval,
            )
            if questions:
                return [{"question": question, "page": page} for question in questions]
            logger.warning(
                f"Question generation failed (attempt {attempt + 1}/{max_retries}): Empty result"
            )
        except Exception as e:
            logger.error(
                f"Question generation error (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                logger.error(f"Error line number: {e.__traceback__.tb_lineno}")

        if attempt < max_retries - 1:
            logger.info(f"Waiting for retry... ({attempt + 2}/{max_retries})")
            time.sleep(2)

    logger.error(f"Question generation failed after {max_retries} retries")
    return []


def _process_question_batch(
    pages_batch: List[Any],
    question_number: int,
    max_retries: int,
    api_key: str,
    model: str,
    base_url: str,
    message: list,
    debug: bool,
    perf_monitor: Optional[PerformanceMonitor],
    stage_name: str,
    interval: float,
) -> List[Dict[str, Any]]:
    batch_output: List[Dict[str, Any]] = []
    for page in pages_batch:
        result = _generate_questions_with_retry(
            page,
            question_number,
            max_retries,
            api_key,
            model,
            base_url,
            message,
            debug,
            perf_monitor,
            stage_name,
            interval,
        )
        if result:
            batch_output.extend(result)
    return batch_output


def _generate_answer_with_retry(
    item: dict,
    max_retries: int,
    api_key: str,
    model: str,
    base_url: str,
    message: list,
    debug: bool,
    perf_monitor: Optional[PerformanceMonitor],
    stage_name: str,
    interval: float,
) -> Optional[Tuple[str, str]]:
    for attempt in range(max_retries):
        try:
            prompt = get_system_prompt_for_answer(item["page"], item["question"])
            answer = llm_generator(
                api_key=api_key,
                model=model,
                base_url=base_url,
                prompt=prompt,
                message=message,
                type="answer",
                debug=debug,
                perf_monitor=perf_monitor,
                perf_stage=stage_name,
                min_interval_seconds=interval,
            )
            if answer and len(answer) > 0:
                return item["question"], answer[0]
            logger.warning(
                f"Answer generation failed (attempt {attempt + 1}/{max_retries}): Empty result"
            )
        except Exception as e:
            logger.error(
                f"Answer generation error (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                logger.error(f"Error line number: {e.__traceback__.tb_lineno}")

        if attempt < max_retries - 1:
            logger.info(f"Waiting for retry... ({attempt + 2}/{max_retries})")
            time.sleep(2)

    question_text = (
        item["question"][:20] + "..."
        if len(item["question"]) > 20
        else item["question"]
    )
    logger.error(
        f"Network status is poor! Discarded QA pair for question: ({question_text})"
    )
    return None


def process_answers(
    api_key: str,
    model: str,
    base_url: str,
    question_items: list,
    message: list | None = None,
    max_qps: float = 5.0,
    max_retries: int = 10,
    debug: bool = False,
    existing_answers: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    perf_monitor: Optional[PerformanceMonitor] = None,
) -> dict:
    """Generate answers using multi-threading (refactored to reduce complexity)."""

    qa_pairs: Dict[str, str] = existing_answers.copy() if existing_answers else {}
    message = message or []
    stage_name = ANSWER_STAGE

    if existing_answers:
        logger.info(
            f"Loaded {len(existing_answers)} answers from checkpoint, "
            "skipping regeneration for them"
        )

    if not question_items:
        logger.warning("No question items supplied for answer generation")
        if perf_monitor:
            perf_monitor.add_stage_items(stage_name, len(qa_pairs))
        return qa_pairs

    pending_items = _extract_pending_items(question_items, qa_pairs)

    if not pending_items:
        logger.info("All questions already have answers from checkpoint")
        if perf_monitor:
            perf_monitor.add_stage_items(stage_name, len(qa_pairs))
        return qa_pairs

    interval = 1.0 / max_qps if max_qps > 0 else 0.0
    worker_count = max(1, _determine_worker_count(max_qps, len(pending_items)))
    logger.info(
        f"Starting answer generation (max_qps={max_qps}, worker_count={worker_count}, "
        f"retries: {max_retries}, pending: {len(pending_items)})..."
    )

    stage_ctx = perf_monitor.stage(stage_name) if perf_monitor else nullcontext()

    with stage_ctx:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = _submit_answer_jobs(
                executor,
                pending_items,
                max_retries,
                api_key,
                model,
                base_url,
                message,
                debug,
                perf_monitor,
                stage_name,
                interval,
            )

            if debug:
                _collect_results_debug(futures, qa_pairs, progress_callback)
            else:
                _collect_results_progress(futures, qa_pairs, progress_callback)

    if perf_monitor:
        perf_monitor.add_stage_items(stage_name, len(qa_pairs))

    return qa_pairs


def _review_single_qa_pair(
    idx: int,
    qa_pair: dict,
    source_text: Optional[str],
    api_key: str,
    model: str,
    base_url: str,
    score_threshold: int,
    max_retries: int,
    debug: bool,
    user_prompt: str,
    perf_monitor: Optional[PerformanceMonitor],
    interval: float,
) -> Tuple[int, dict, bool, int, str, Optional[str], Optional[str]]:
    last_error_message = ""
    reference_text = qa_pair.get("context") or source_text or ""
    for attempt in range(max_retries):
        try:
            qa_pair_json = json.dumps(qa_pair, ensure_ascii=False, indent=2)
            review_prompt = get_system_prompt_for_review(reference_text, qa_pair_json)
            review_messages = [
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": user_prompt},
            ]
            review_result_list = llm_generator(
                api_key=api_key,
                model=model,
                base_url=base_url,
                type="review",
                message=review_messages,
                debug=debug,
                perf_monitor=perf_monitor,
                perf_stage=REVIEW_STAGE,
                min_interval_seconds=interval,
            )
            review_result = review_result_list[0] if review_result_list else ""
            if not review_result:
                last_error_message = "Empty review result"
                continue

            try:
                review_json = json.loads(review_result)
            except json.JSONDecodeError as exc:
                last_error_message = f"JSON decode error: {exc}"
                return (
                    idx,
                    qa_pair,
                    False,
                    0,
                    "Failed to parse review result",
                    "warning",
                    last_error_message,
                )

            score = review_json.get("score", 0)
            reason = review_json.get("reason", "No reason provided")

            if score >= score_threshold:
                return idx, qa_pair, True, score, reason, None, None

            return idx, qa_pair, False, score, reason, "info", None

        except Exception as exc:  # noqa: BLE001
            last_error_message = str(exc)
            if attempt < max_retries - 1:
                time.sleep(1)
    return (
        idx,
        qa_pair,
        False,
        0,
        "Review request failed",
        "error",
        last_error_message,
    )


def review_qa_pairs(
    qa_pairs: list[dict],
    *,
    source_text: Optional[str],
    api_key: str,
    model: str,
    base_url: str,
    score_threshold: int = 4,
    max_qps: float = 5.0,
    max_retries: int = 3,
    debug: bool = False,
    user_prompt: str = "请进行评分",
    progress_desc: str = "Reviewing QA pairs",
    dbg: Any = None,
    perf_monitor: Optional[PerformanceMonitor] = None,
) -> tuple[list[dict], int]:

    if not qa_pairs:
        logger.info("No QA pairs supplied for review; skipping review process")
        if perf_monitor:
            perf_monitor.add_stage_items(REVIEW_STAGE, 0)
        return [], 0

    interval = 1.0 / max_qps if max_qps > 0 else 0.0
    reviewed_qa_pairs: list[dict] = []
    rejected_count = 0

    stage_ctx = perf_monitor.stage(REVIEW_STAGE) if perf_monitor else nullcontext()
    progress_bar = (
        None if debug else tqdm(total=len(qa_pairs), desc=progress_desc, unit="pair")
    )

    worker_count = _determine_worker_count(max_qps, len(qa_pairs))
    logger.info(
        f"Starting QA review (max_qps={max_qps}, worker_count={worker_count}, "
        f"retries: {max_retries}, total: {len(qa_pairs)})..."
    )

    with stage_ctx:
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            futures = _submit_review_jobs(
                executor,
                qa_pairs,
                source_text,
                api_key,
                model,
                base_url,
                score_threshold,
                max_retries,
                debug,
                user_prompt,
                perf_monitor,
                interval,
            )

            for future in as_completed(futures):
                result = future.result()
                rejected_count = _handle_review_result(
                    result,
                    reviewed_qa_pairs,
                    rejected_count,
                    dbg,
                )
                _update_progress_bar(
                    progress_bar, len(reviewed_qa_pairs), rejected_count
                )

        if progress_bar:
            progress_bar.close()

    if perf_monitor:
        perf_monitor.add_stage_items(REVIEW_STAGE, len(qa_pairs))

    return reviewed_qa_pairs, rejected_count


def find_tagpath_by_label(domain_tree: DomainTree, label: str):
    return domain_tree.find_path(label)


def _process_llm_response(
    response: requests.Response,
    debug: bool,
    attempt: int,
    type_str: str,
    perf_monitor: Optional[PerformanceMonitor],
    perf_stage: Optional[str],
    messages_payload: list,
    prompt: Union[str, None],
    model: str,
    base_url: str,
    temperature: float,
    top_p: float,
) -> Union[list, None]:
    result = response.json()
    usage = result.get("usage") or {}

    choices = result.get("choices") or []
    if not choices:
        return None

    debug_enabled = _should_log_debug(debug, attempt)

    if debug_enabled:
        _log_llm_metadata(response, usage)

    raw_output = choices[0]["message"].get("content", "") or ""

    if debug_enabled and raw_output:
        _log_llm_raw_output(raw_output)

    _record_perf_if_needed(
        perf_monitor=perf_monitor,
        perf_stage=perf_stage,
        type_str=type_str,
        messages_payload=messages_payload,
        raw_output=raw_output,
        prompt=prompt,
        model=model,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        attempt=attempt,
        status_code=response.status_code,
    )

    if type_str == "question":
        fmt_output = extract_json_from_llm_output(raw_output)
        if debug_enabled:
            logger.debug(f"Parsed questions: {fmt_output}")
            logger.debug(f"Question count: {len(fmt_output) if fmt_output else 0}")
            logger.debug("=" * 60)
        return fmt_output if fmt_output is not None else []

    if debug_enabled:
        _log_returning_raw_content(raw_output)

    return [raw_output] if raw_output else []


def generatr_qa_pairs(
    question_info: list,
    api_key: str,
    base_url: str,
    model_name: str,
    question_number: int = 5,
    message: list = None,
    max_qps: float = 5.0,
    domain_tree: DomainTree = None,
    debug: bool = False,
    progress_tracker: Optional["QAProgressTracker"] = None,
    perf_monitor: Optional[PerformanceMonitor] = None,
) -> list:
    if message is None:
        message = []
    if domain_tree is None:
        from datamax.generator.domain_tree import DomainTree

        domain_tree = DomainTree([])
    existing_answers = progress_tracker.existing_answers() if progress_tracker else None
    question_lookup = {item["question"]: item for item in question_info}

    def _on_answer_generated(question: str, answer: str):
        if not progress_tracker:
            return
        question_item = question_lookup.get(question)
        if not question_item:
            return
        label = question_item.get("label", "")
        entry = {
            "qid": question_item.get("qid", ""),
            "instruction": question,
            "input": "",
            "output": answer,
            "label": label,
            "tag-path": (
                find_tagpath_by_label(domain_tree, label) if domain_tree else ""
            ),
            "context": question_item.get("page", ""),
        }
        progress_tracker.record(entry)

    qa_pairs = process_answers(
        question_items=question_info,
        message=message,
        max_qps=max_qps,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        debug=debug,
        existing_answers=existing_answers,
        progress_callback=_on_answer_generated,
        perf_monitor=perf_monitor,
    )
    logger.success(f"Completed! Generated {len(qa_pairs)} QA pairs in total")
    res_list = []
    for question_item in question_info:
        question = question_item["question"]
        # only add question with answer
        if question in qa_pairs:
            label = question_item.get("label", "")
            answer = qa_pairs[question]
            tag_path = find_tagpath_by_label(domain_tree, label) if domain_tree else ""
            qid = question_item.get("qid", "")
            qa_entry = {
                "qid": qid,
                "instruction": question,
                "input": "",
                "output": answer,
                "label": label,
                "tag-path": tag_path,
                "context": question_item.get("page", ""),
            }
            res_list.append(qa_entry)
    return res_list


def _parse_node_name(part: str) -> Optional[str]:
    """Extract node name from the first segment."""
    name_part = part.split(":", 1)
    if len(name_part) != 2:
        print("Format error: missing node name")
        return None
    return name_part[1].strip()


def _parse_parent_child(parts: list[str]) -> tuple[str, str]:
    """Extract parent and child names from the remaining parts."""
    parent = ""
    child = ""
    for segment in parts:
        key_value = [s.strip() for s in segment.split(":", 1)]
        if len(key_value) != 2:
            continue
        key = key_value[0].lower()
        val = key_value[1]
        if key == "parent":
            parent = val
        elif key == "child":
            child = val
    return parent, child


def _handle_add_root(domain_tree, node_name: str):
    if domain_tree.add_node(node_name):
        print(f"Successfully added node '{node_name}' as root node")
    else:
        print("Add failed: unknown error")


def _handle_add_between(domain_tree, node_name: str, parent: str, child: str):
    if domain_tree.insert_node_between(node_name, parent, child):
        print(
            f"Successfully inserted node '{node_name}' between '{parent}' and '{child}'"
        )
    else:
        print("Insert failed: please check parent and child relationship")


def _handle_add_under_parent(domain_tree, node_name: str, parent: str):
    if domain_tree.add_node(node_name, parent):
        print(f"Successfully added node '{node_name}' under parent node '{parent}'")
    else:
        print(f"Add failed: parent node '{parent}' not found")


def _handle_add_node(domain_tree, parts):
    node_name = _parse_node_name(parts[0])
    if not node_name:
        return

    parent_name, child_name = _parse_parent_child(parts[1:])

    if not parent_name:
        _handle_add_root(domain_tree, node_name)
    elif child_name:
        _handle_add_between(domain_tree, node_name, parent_name, child_name)
    else:
        _handle_add_under_parent(domain_tree, node_name, parent_name)


def _handle_delete_node(domain_tree, user_input):
    name_part = user_input.split(":", 1)
    node_name = name_part[1].strip() if len(name_part) == 2 else ""
    if not node_name:
        print("Format error: please provide node name")
        return
    if domain_tree.remove_node(node_name):
        print(f"Successfully deleted node '{node_name}' and all its descendant nodes")
    else:
        print(f"Delete failed: node '{node_name}' not found")


def _handle_rename_node(domain_tree, user_input):
    parts = [segment.strip() for segment in user_input.split(";")]
    if len(parts) != 2:
        print(
            "Format error: please use 'Rename node: <new name>; Original: <old name>'"
        )
        return
    new_part = parts[0].split(":", 1)
    old_part = parts[1].split(":", 1)
    if len(new_part) != 2 or len(old_part) != 2:
        print(
            "Format error: please use 'Rename node: <new name>; Original: <old name>'"
        )
        return
    new_name = new_part[1].strip()
    old_name = old_part[1].strip()
    if domain_tree.update_node(old_name, new_name):
        print(f"Successfully updated node '{old_name}' to '{new_name}'")
    else:
        print(f"Update failed: node '{old_name}' not found")


def _print_tree_help():
    print("\nDo you need to modify the tree?")
    print("Supported operations:")
    print("1. Add node: <name>; Parent: <parent> (parent optional; blank adds as root)")
    print("2. Add node: <name>; Parent: <parent>; Child: <child>")
    print("3. Delete node: <name>")
    print("4. Rename node: <new name>; Original: <old name>")
    print("5. Finish")
    print(
        "Note: Node format is usually like '1.1 Logistics Planning' or '1 Transportation Systems'."
    )
    print("\nPlease enter operation command (enter 'Finish' to exit):")


def _handle_user_add(domain_tree, user_input):
    parts = [segment.strip() for segment in user_input.split(";")]
    if not parts:
        print("Format error: please use 'Add node: <name>; Parent: <parent>'")
        return
    _handle_add_node(domain_tree, parts)


def _handle_user_command(domain_tree, user_input):
    """Dispatch user commands based on prefixes."""
    lower = user_input.lower()

    if lower.startswith("add node"):
        _handle_user_add(domain_tree, user_input)
    elif lower.startswith("delete node"):
        _handle_delete_node(domain_tree, user_input)
    elif lower.startswith("rename node"):
        _handle_rename_node(domain_tree, user_input)
    else:
        print("Unknown operation. Please follow the listed formats.")


def _print_tree_and_prompt(domain_tree):
    print("\nCurrent tree structure:")
    print(domain_tree.visualize())
    print("\nPlease enter next operation command:")


def _interactive_tree_modification(domain_tree):
    """Interactive custom domain tree structure modification."""
    _print_tree_help()

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            if user_input.lower() == "finish":
                print("Tree operations completed, continuing QA pair generation...")
                break

            _handle_user_command(domain_tree, user_input)
            _print_tree_and_prompt(domain_tree)

        except KeyboardInterrupt:
            print("\nOperation interrupted, continuing QA pair generation...")
            break
        except Exception as e:
            print(f"Operation error: {e}")
            print("Please re-enter operation command:")

    return domain_tree


def _prepare_page_content(
    content: str,
    structured_data: bool,
    chunk_size: int,
    chunk_overlap: int,
    use_mineru: bool,
) -> Tuple[List[str], str]:
    if structured_data is False:
        content_type = "Text"
        if content.strip().startswith("#") or "**" in content or "```" in content:
            content_type = "Markdown"
            logger.info("Detected Markdown format content")
        elif any(keyword in content.lower() for keyword in ["pdf", "page", "document"]):
            content_type = "PDF converted content"
            logger.info("Detected PDF converted content")
            if use_mineru:
                logger.info("Using MinerU parsed PDF content")
            else:
                logger.info("Using PDF parsed PDF content")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_text(content), content_type

    elif structured_data is True:
        content_type = "Dict"
        logger.info("Detected Dict format content")
        import json
        import re

        json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        json_matches = re.findall(json_pattern, content)

        valid_json_objects = []
        for match in json_matches:
            try:
                json.loads(match)  # Check validity
                valid_json_objects.append(match)
            except json.JSONDecodeError:
                continue

        return valid_json_objects, content_type
    return [], "Unknown"


def _prepare_domain_tree(
    use_tree_label: bool,
    custom_domain_tree: list,
    api_key: str,
    base_url: str,
    model_name: str,
    page_content: list,
    debug: bool,
    interactive_tree: bool,
):
    domain_tree = None
    if use_tree_label:
        from datamax.generator.domain_tree import DomainTree

        if custom_domain_tree is not None:
            domain_tree = DomainTree(custom_domain_tree)
            logger.info("Using user-uploaded custom domain tree structure")
            print(
                "Using your uploaded custom domain tree structure for pre-labeling..."
            )
        else:
            domain_tree = process_domain_tree(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                text="\n".join(page_content),
                temperature=0.7,
                top_p=0.9,
                debug=debug,
            )
            if domain_tree is None:
                logger.info(
                    "Domain tree generation failed, using plain text generation strategy"
                )
                use_tree_label = False

        if interactive_tree and domain_tree and domain_tree.tree:
            tree_source = "Custom" if custom_domain_tree is not None else "Generated"
            print("\n" + "=" * 60)
            print(f"Domain tree source: {tree_source}")
            print("=" * 60)
            print(domain_tree.visualize())
            print("=" * 60)
            if custom_domain_tree is not None:
                print("You can modify the custom tree, or enter to use it directly")
            domain_tree = _interactive_tree_modification(domain_tree)

    return domain_tree, use_tree_label


def full_qa_labeling_process(
    content: str = None,
    file_path: str = None,
    api_key: str = None,
    base_url: str = None,
    model_name: str = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    question_number: int = 5,
    max_qps: float = 5.0,
    use_tree_label: bool = False,
    messages: list = None,
    interactive_tree: bool = False,
    custom_domain_tree: list = None,
    use_mineru: bool = False,  # Add use_mineru parameter
    debug: bool = False,
    structured_data: bool = False,
    checkpoint_path: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    perf_monitor: Optional[PerformanceMonitor] = None,
):
    """
    Complete QA generation workflow, including splitting, domain tree generation and interaction,
    question generation, label tagging, and answer generation.
    """
    monitor = perf_monitor or PerformanceMonitor()

    if not content:
        logger.error(
            "content parameter is required. Check content is null or not. Check file_path is null or not."
        )
        return []

    if not api_key or not base_url or not model_name:
        logger.error("api_key, base_url, and model_name parameters are required")
        return []

    logger.info("Using text content for splitting")
    page_content, content_type = _prepare_page_content(
        content, structured_data, chunk_size, chunk_overlap, use_mineru
    )

    domain_tree, use_tree_label = _prepare_domain_tree(
        use_tree_label,
        custom_domain_tree,
        api_key,
        base_url,
        model_name,
        page_content,
        debug,
        interactive_tree,
    )

    question_info = process_questions(
        api_key=api_key,
        model=model_name,
        base_url=base_url,
        page_content=page_content,
        question_number=question_number,
        max_qps=max_qps,
        message=messages,
        debug=debug,
        perf_monitor=monitor,
    )
    for question_item in question_info:
        if "qid" not in question_item:
            question_item["qid"] = str(uuid.uuid4())

    if (
        use_tree_label
        and domain_tree
        and hasattr(domain_tree, "to_json")
        and domain_tree.to_json()
    ):
        q_match_list = process_match_tags(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            tags_json=domain_tree.to_json(),
            questions=[q["question"] for q in question_info],
            max_qps=max_qps,
            debug=debug,
        )
        label_map = {item["question"]: item.get("label", "") for item in q_match_list}
        for question_item in question_info:
            question_item["label"] = label_map.get(question_item["question"], "")
    else:
        for question_item in question_info:
            question_item["label"] = ""

    progress_tracker = None
    if checkpoint_path:
        progress_tracker = QAProgressTracker(
            checkpoint_path, resume=resume_from_checkpoint
        )
        logger.info(
            f"Checkpointing QA generation to {checkpoint_path} "
            f"(resume={'enabled' if resume_from_checkpoint else 'disabled'})"
        )

    qa_list = generatr_qa_pairs(
        question_info=question_info,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        question_number=question_number,
        max_qps=max_qps,
        domain_tree=domain_tree if use_tree_label else None,
        debug=debug,
        progress_tracker=progress_tracker,
        perf_monitor=monitor,
    )

    result = {
        "qa_pairs": qa_list,
        "metadata": {
            "content_type": content_type,
            "question_number": question_number,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_tree_label": use_tree_label,
            "structured_data": structured_data,
            "total_questions_generated": len(question_info),
            "total_qa_pairs": len(qa_list),
        },
    }

    if use_tree_label and domain_tree:
        result["domain_tree"] = domain_tree.to_json()
        result["domain_tree_source"] = (
            "custom" if custom_domain_tree is not None else "generated"
        )
    else:
        result["domain_tree"] = None

    result["performance"] = monitor.build_report()

    return result
