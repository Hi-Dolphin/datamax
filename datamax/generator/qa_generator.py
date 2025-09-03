import json
import os.path
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from loguru import logger
from pyexpat.errors import messages
from tqdm import tqdm  
from dotenv import load_dotenv
from .domain_tree import DomainTree   # for cache domain tree
from .prompt_templates import (
    get_system_prompt_for_match_label,
    get_system_prompt_for_domain_tree,
    get_system_prompt_for_question,
    get_system_prompt_for_answer
)

lock = threading.Lock()

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
        logger.info(f"📄 Markdown file '{file_name}' split into {len(page_content)} chunks")
        return page_content

    except Exception as e:
        logger.error(f"Failed to load {Path(md_path).name}: {str(e)}")
        return []


def load_and_split_text(file_path: str, chunk_size: int, chunk_overlap: int, use_mineru: bool = False, use_qwen_vl_ocr: bool = False) -> list:
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
        from datamax.parser.core import DataMax
        
        # Get file extension for logging
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        logger.info(f"开始处理文件: {file_name} (类型: {file_ext})")
        
        # 使用DataMax解析文件，传递use_mineru和use_qwen_vl_ocr参数
        dm = DataMax(file_path=file_path, to_markdown=True, use_mineru=use_mineru, use_qwen_vl_ocr=use_qwen_vl_ocr)
        parsed_data = dm.get_data()

        if not parsed_data:
            logger.error(f"File parsing failed: {file_name}")
            return []
            
        # Get parsed content
        if isinstance(parsed_data, list):
            # If multiple files, take the first one
            content = parsed_data[0].get('content', '')
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

        # 根据文件类型提供不同的日志信息
        if file_ext == '.pdf':
            if use_qwen_vl_ocr:
                logger.info(f"📄 PDF文件 '{file_name}' 使用Qwen-VL OCR解析，被分解为 {len(page_content)} 个chunk")
            elif use_mineru:
                logger.info(f"📄 PDF文件 '{file_name}' 使用MinerU解析，被分解为 {len(page_content)} 个chunk")
            else:
                logger.info(f"📄 PDF file '{file_name}' parsed with PyMuPDF, split into {len(page_content)} chunks")
        else:
            logger.info(f"📄 {file_ext.upper()} file '{file_name}' split into {len(page_content)} chunks")
            
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


def llm_generator(
    api_key: str,
    model: str,
    base_url: str,
    prompt: str,
    type: str,
    message: list = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    debug: bool = False,
) -> list:
    """Generate content using LLM API"""
    try:
        if not message:
            logger.warning("No message provided, using default system prompt")
            message = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请严格按照要求生成内容"},
            ]
        
        if debug:
            logger.debug("=" * 80)
            logger.debug("🚀 大模型请求详细信息")
            logger.debug("=" * 80)
            logger.debug(f"📍 模型: {model}")
            logger.debug(f"🌐 API地址: {base_url}")
            logger.debug(f"🌡️  温度参数: {temperature}")
            logger.debug(f"🎯 Top-P参数: {top_p}")
            logger.debug(f"📝 请求类型: {type}")
            logger.debug("-" * 40)
            logger.debug("💬 消息内容:")
            for i, msg in enumerate(message, 1):
                role_emoji = "🤖" if msg["role"] == "system" else "👤" if msg["role"] == "user" else "🔧"
                logger.debug(f"  {i}. {role_emoji} {msg['role'].upper()}:")
                content_lines = msg["content"].split('\n')
                for line in content_lines:
                    if line.strip():
                        logger.debug(f"     {line}")
                logger.debug("")
            logger.debug("-" * 40)
        
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

        if debug:
            logger.debug("📤 发送请求到大模型...")
        
        response = requests.post(base_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if debug:
            logger.debug("✅ 大模型响应成功")
            logger.debug("-" * 40)
            logger.debug("📥 响应详细信息:")
            logger.debug(f"  📊 状态码: {response.status_code}")
            if "usage" in result:
                usage = result["usage"]
                logger.debug(f"  🔢 Token使用情况:")
                logger.debug(f"     输入Token: {usage.get('prompt_tokens', 'N/A')}")
                logger.debug(f"     输出Token: {usage.get('completion_tokens', 'N/A')}")
                logger.debug(f"     总Token: {usage.get('total_tokens', 'N/A')}")
            logger.debug("-" * 40)

        # Parse LLM response
        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            
            if debug:
                logger.debug("📋 大模型原始回答:")
                output_lines = output.split('\n')
                for line in output_lines:
                    if line.strip():
                        logger.debug(f"  {line}")
                logger.debug("-" * 40)
            
            if type == "question":
                fmt_output = extract_json_from_llm_output(output)
                if debug:
                    logger.debug(f"🔄 解析后的问题列表: {fmt_output}")
                    logger.debug(f"📈 解析出 {len(fmt_output) if fmt_output else 0} 个问题")
                    logger.debug("=" * 80)
                return fmt_output if fmt_output is not None else []
            else:
                if debug:
                    logger.debug(f"📝 返回原始内容 (长度: {len(output) if output else 0} 字符)")
                    logger.debug("=" * 80)
                return [output] if output else []
        
        if debug:
            logger.debug("⚠️  响应中没有有效的choices内容")
            logger.debug("=" * 80)
        return []

    except Exception as e:
        logger.error(f"LLM keyword extraction failed: {e}")
        if hasattr(e, "__traceback__") and e.__traceback__ is not None:
            logger.error(f"Error line number: {e.__traceback__.tb_lineno}")
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
    max_workers: int = 3,
    debug: bool = False,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger.info(f"Starting concurrent question-tag matching... (max_workers={max_workers})")
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
        )
        # llm_generator return a list, only one question is passed, take the first one
        return match[0] if match else {"question": q, "label": "其他"}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_q = {executor.submit(match_one_question, q): q for q in questions}
        for future in as_completed(future_to_q):
            res = future.result()
            #print(f"Question: {res.get('question', '')} | Matched label: {res.get('label', '')}")
            results.append(res)
    logger.success(f"Question-tag matching completed successfully, generated {len(results)} questions")
    return results


def process_domain_tree(
    api_key: str,
    model: str,
    base_url: str,
    text: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_retries: int = 3,
    debug: bool = False,
) -> DomainTree:
    prompt = get_system_prompt_for_domain_tree(text)
    logger.info(f"Domain tree generation started...")
    
    if debug:
        logger.debug("=" * 80)
        logger.debug("🌳 DOMAIN TREE GENERATION DEBUG INFO")
        logger.debug("=" * 80)
        logger.debug(f"📝 System Prompt: {prompt[:200]}...")
        logger.debug(f"🔧 Model: {model}")
        logger.debug(f"🌐 API URL: {base_url}")
        logger.debug(f"🌡️ Temperature: {temperature}")
        logger.debug(f"🎯 Top-P: {top_p}")
        logger.debug(f"🔄 Max Retries: {max_retries}")
        logger.debug("=" * 80)
    
    for attempt in range(max_retries):
        try:
            message = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请严格按照要求生成内容"},
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
            response = requests.post(base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if debug:
                logger.debug(f"📡 API Response Status: {response.status_code}")
                if "usage" in result:
                    logger.debug(f"🔢 Token Usage: {result['usage']}")

            # Parse LLM response
            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0]["message"]["content"]
                if debug:
                    logger.debug(f"📄 Raw Response: {output[:500]}...")
                if output:
                    json_output = extract_json_from_llm_output(output)
                    if debug:
                        logger.debug(f"🔍 Parsed JSON: {json_output}")
                    if json_output is not None:
                        domain_tree = DomainTree()
                        domain_tree.from_json(json_output)
                        if debug:
                            logger.debug(f"🌳 Generated Domain Tree: {domain_tree.visualize()}")
                        logger.info(f"Domain tree generated successfully, created {len(json_output)} main tags")
                        return domain_tree
                    else:
                        logger.warning(f"Domain tree generation failed (attempt {attempt + 1}/{max_retries}): Unable to parse JSON output")
                else:
                    logger.warning(f"Domain tree generation failed (attempt {attempt + 1}/{max_retries}): Empty output")
            else:
                logger.warning(f"Domain tree generation failed (attempt {attempt + 1}/{max_retries}): Invalid response format")
                
        except Exception as e:
            logger.error(f"Domain tree generation error (attempt {attempt + 1}/{max_retries}): {e}")
            if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                logger.error(f"Error line number: {e.__traceback__.tb_lineno}")
            
            if attempt == max_retries - 1:
                error_msg = "Tree generation failed! Please check network or switch LLM model! Will continue with plain text generation"
                print(f"❌ {error_msg}")
                logger.error(f"Domain tree generation failed after {max_retries} retries: {error_msg}")
                return None
            else:
                logger.info(f"Waiting for retry... ({attempt + 2}/{max_retries})")
                import time
                time.sleep(2)  # Wait 2 seconds before retry
    
    error_msg = "Tree generation failed! Please check network or switch LLM model! Will continue with plain text generation"
    print(f"❌ {error_msg}")
    logger.error(f"Domain tree generation failed after {max_retries} retries: {error_msg}")
    return None


def process_questions(
    api_key: str,
    model: str,
    base_url: str,
    page_content: list,
    question_number: int,
    max_workers: int = 5,
    message: list = None,
    max_retries: int = 3,
    debug: bool = False,
) -> list:
    """Generate questions using multi-threading with retry mechanism"""
    total_questions = []
    if message is None:
        message = []

    def _generate_questions_with_retry(page):
        """Inner function for question generation with retry"""
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
                )
                if questions:
                    return [
                        {"question": question, "page": page} for question in questions
                    ]
                else:
                    logger.warning(f"Question generation failed (attempt {attempt + 1}/{max_retries}): Empty result")
            except Exception as e:
                logger.error(f"Question generation error (attempt {attempt + 1}/{max_retries}): {e}")
                if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                    logger.error(f"Error line number: {e.__traceback__.tb_lineno}")
            
            if attempt < max_retries - 1:
                logger.info(f"Waiting for retry... ({attempt + 2}/{max_retries})")
                import time
                time.sleep(2)  # Wait 2 seconds before retry
        
        logger.error(f"Question generation failed after {max_retries} retries")
        return []

    logger.info(f"Starting question generation (threads: {max_workers}, retries: {max_retries})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_questions_with_retry, page) for page in page_content]
        if debug:
            # Debug模式下禁用进度条，避免与debug日志冲突
            for future in as_completed(futures):
                result = future.result()
                if result:
                    with lock:
                        total_questions.extend(result)
        else:
            with tqdm(as_completed(futures), total=len(futures), desc="Generating questions") as pbar:
                for future in pbar:
                    result = future.result()
                    if result:
                        with lock:
                            total_questions.extend(result)
                        pbar.set_postfix({"Generated questions": len(total_questions)})
    return total_questions


def process_answers(
    api_key: str,
    model: str,
    base_url: str,
    question_items: list,
    message: list | None = None,
    max_workers=5,
    max_retries: int = 3,
    debug: bool = False,
) -> dict:
    """Generate answers using multi-threading"""
    qa_pairs = {}
    if message is None:
        message = []

    def _generate_answer_with_retry(item):
        """Inner function for answer generation with retry"""
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
                )
                if answer and len(answer) > 0:
                    return item["question"], answer[0]  # llm_generator returns a list
                else:
                    logger.warning(f"Answer generation failed (attempt {attempt + 1}/{max_retries}): Empty result")
            except Exception as e:
                logger.error(f"Answer generation error (attempt {attempt + 1}/{max_retries}): {e}")
                if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                    logger.error(f"Error line number: {e.__traceback__.tb_lineno}")
            
            if attempt < max_retries - 1:
                logger.info(f"Waiting for retry... ({attempt + 2}/{max_retries})")
                import time

                time.sleep(2)  # retry after 2 seconds

        # all retries failed
        question_text = item["question"][:20] + "..." if len(item["question"]) > 20 else item["question"]
        logger.error(f"Network status is poor! Discarded QA pair for question: ({question_text})")
        return None  # return None to discard the question with answer

    logger.info(f"Starting answer generation (threads: {max_workers}, retries: {max_retries})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_answer_with_retry, item): item
            for item in question_items
        }

        if debug:
            # Debug模式下禁用进度条，避免与debug日志冲突
            for future in as_completed(futures):
                result = future.result()
                if result is not None:  # only add question with answer
                    question, answer = result
                    with lock:
                        qa_pairs[question] = answer
        else:
            with tqdm(as_completed(futures), total=len(futures), desc="Generating answers") as pbar:
                for future in pbar:
                    result = future.result()
                    if result is not None:  # only add question with answer
                        question, answer = result
                        with lock:
                            qa_pairs[question] = answer
                        pbar.set_postfix({"Generated answers": len(qa_pairs)})
    return qa_pairs


# find tagpath by label


def find_tagpath_by_label(domain_tree: DomainTree, label: str):
    return domain_tree.find_path(label)


def generatr_qa_pairs(
    question_info: list,
    api_key: str,
    base_url: str,
    model_name: str,
    question_number: int = 5,
    message: list = None,
    max_workers: int = 5,
    domain_tree: DomainTree = None,
    debug: bool = False,
) -> list:
    if message is None:
        message = []
    if domain_tree is None:
        from datamax.generator.domain_tree import DomainTree
        domain_tree = DomainTree([])
    qa_pairs = process_answers(
        question_items=question_info,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        debug=debug,
    )
    logger.success(
        f"Completed! Generated {len(qa_pairs)} QA pairs in total"
    )
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
                
            }
            res_list.append(qa_entry)
    return res_list


def _interactive_tree_modification(domain_tree):
    """
    Interactive custom domain tree structure modification
    :param domain_tree: DomainTree instance
    :return: Modified DomainTree instance
    """
    print("\n Do you need to modify the tree?")
    print("Supported operations:")
    print("1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）")
    print("2. 增加节点：xxx；父节点：xxx；子节点：xxx")
    print("3. 删除节点：xxx")
    print("4. 更新节点：新名称；原先节点：旧名称")
    print("5. 结束树操作")
    print("Note: Node format is usually: x.xx xxxx, like: '1.1 货物运输组织与路径规划' or '1 运输系统组织'")
    print("\nPlease enter operation command (enter '结束树操作' to exit):")
    while True:
        try:
            user_input = input("> ").strip()
            if user_input == "结束树操作":
                print("✅ Tree operations completed, continuing QA pair generation...")
                break
            elif user_input.startswith("增加节点："):
                parts = user_input.split("；")
                if len(parts) >= 2:
                    node_name = parts[0].replace("增加节点：", "").strip()
                    parent_name = parts[1].replace("父节点：", "").strip()
                    if not parent_name:
                        if domain_tree.add_node(node_name):
                            print(f"✅ Successfully added node '{node_name}' as root node")
                        else:
                            print(f"❌ Add failed: Unknown error")
                    elif len(parts) == 2:
                        if domain_tree.add_node(node_name, parent_name):
                            print(f"✅ Successfully added node '{node_name}' under parent node '{parent_name}'")
                        else:
                            print(f"❌ Add failed: Parent node '{parent_name}' not found")
                    elif len(parts) == 3:
                        child_name = parts[2].replace("子节点：", "").strip()
                        if domain_tree.insert_node_between(node_name, parent_name, child_name):
                            print(f"✅ Successfully inserted node '{node_name}' between '{parent_name}' and '{child_name}'")
                        else:
                            print(f"❌ Insert failed: Please check parent and child node relationship")
                    else:
                        print("❌ Format error: Please use correct format")
                else:
                    print("❌ Format error: Please use correct format")
            elif user_input.startswith("删除节点："):
                node_name = user_input.replace("删除节点：", "").strip()
                if domain_tree.remove_node(node_name):
                    print(f"✅ Successfully deleted node '{node_name}' and all its descendant nodes")
                else:
                    print(f"❌ Delete failed: Node '{node_name}' not found")
            elif user_input.startswith("更新节点："):
                parts = user_input.split("；")
                if len(parts) == 2:
                    new_name = parts[0].replace("更新节点：", "").strip()
                    old_name = parts[1].replace("原先节点：", "").strip()
                    if domain_tree.update_node(old_name, new_name):
                        print(f"✅ Successfully updated node '{old_name}' to '{new_name}'")
                    else:
                        print(f"❌ Update failed: Node '{old_name}' not found")
                else:
                    print("❌ Format error: Please use correct format, like: 更新节点：新名称；原先节点：旧名称")
            else:
                print("❌ Unknown operation, please use correct format")
            print("\n📝 Current tree structure:")
            print(domain_tree.visualize())
            print("\nPlease enter next operation command:")
            print("Supported operations:")
            print("1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）")
            print("2. 增加节点：xxx；父节点：xxx；子节点：xxx")
            print("3. 删除节点：xxx")
            print("4. 更新节点：新名称；原先节点：旧名称")
            print("5. 结束树操作")
            print("Note: Node format is usually: x.xx xxxx, like: '1.1 货物运输组织与路径规划' or '1 运输系统组织'")
        except KeyboardInterrupt:
            print("\n\n⚠️⚠️Operation interrupted⚠️⚠️, continuing QA pair generation...")
            break
        except Exception as e:
            print(f"❌ Operation error: {e}")
            print("Please re-enter operation command:")
    return domain_tree


def full_qa_labeling_process(
    content: str = None,
    file_path: str = None,
    api_key: str = None,
    base_url: str = None,
    model_name: str = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    question_number: int = 5,
    max_workers: int = 5,
    use_tree_label: bool = True,
    messages: list = None,
    interactive_tree: bool = True,
    custom_domain_tree: list = None,
    use_mineru: bool = False,  # Add use_mineru parameter
    debug: bool = False,
):
    """
    Complete QA generation workflow, including splitting, domain tree generation and interaction, 
    question generation, label tagging, and answer generation.
    """
    import uuid

    from datamax.generator.qa_generator import (
        generatr_qa_pairs,
        process_domain_tree,
        process_match_tags,
        process_questions,
    )

    # Validate required parameters
    if not content:
        logger.error("content parameter is required. Check content is null or not. Check file_path is null or not.")
        return []

    if not api_key:
        logger.error("api_key parameter is required")
        return []

    if not base_url:
        logger.error("base_url parameter is required")
        return []

    if not model_name:
        logger.error("model_name parameter is required")
        return []

    # 1. text split - only process content, not file_path
    logger.info("Using text content for splitting")
    
    # Try to detect content type
    content_type = "Text"
    if content.strip().startswith('#') or '**' in content or '```' in content:
        content_type = "Markdown"
        logger.info("📄 Detected Markdown format content")
    elif any(keyword in content.lower() for keyword in ['pdf', 'page', 'document']):
        content_type = "PDF converted content"
        logger.info("📄 Detected PDF converted content")
        if use_mineru:
            logger.info("📄 Using MinerU parsed PDF content")
        else:
            logger.info("📄 Using PyMuPDF parsed PDF content")
    
    # Directly use LangChain's text splitter for chunking without creating temporary files
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    page_content = splitter.split_text(content)
    
    # Add content chunking completion log
    if content_type == "PDF converted content":
        if use_mineru:
            logger.info(f"✅ MinerU parsed PDF content processing completed, generated {len(page_content)} text chunks")
        else:
            logger.info(f"✅ PyMuPDF parsed PDF content processing completed, generated {len(page_content)} text chunks")
    else:
        logger.info(f"✅ {content_type} content processing completed, generated {len(page_content)} text chunks")

    # 2. domain tree generation
    domain_tree = None
    if use_tree_label:
        from datamax.generator.domain_tree import DomainTree

        # if custom_domain_tree is not None, use it
        if custom_domain_tree is not None:
            domain_tree = DomainTree(custom_domain_tree)
            logger.info("🌳 Using user-uploaded custom domain tree structure")
            print("🌳 Using your uploaded custom domain tree structure for pre-labeling...")
        else:
            # otherwise, generate tree from text
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
                # tree generation failed, use text generation strategy
                logger.info("Domain tree generation failed, using plain text generation strategy")
                use_tree_label = False
        
        # Unified interactive editing logic
        if interactive_tree and domain_tree and domain_tree.tree:
            tree_source = "Custom" if custom_domain_tree is not None else "Generated"
            print("\n" + "="*60)
            print(f"🌳 {tree_source} domain tree structure:")
            print("="*60)
            print(domain_tree.visualize())
            print("=" * 60)
            if custom_domain_tree is not None:
                print("💡 You can modify the custom tree, or enter '结束树操作' to use it directly")
            domain_tree = _interactive_tree_modification(domain_tree)
    # generate questions
    question_info = process_questions(
        api_key=api_key,
        model=model_name,
        base_url=base_url,
        page_content=page_content,
        question_number=question_number,
        max_workers=max_workers,
        message=messages,
        debug=debug,
    )
    for question_item in question_info:
        if "qid" not in question_item:
            question_item["qid"] = str(uuid.uuid4())
    # 4. label tagging
    if use_tree_label and domain_tree and hasattr(domain_tree, 'to_json') and domain_tree.to_json():
        q_match_list = process_match_tags(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            tags_json=domain_tree.to_json(),
            questions=[q["question"] for q in question_info],
            max_workers=max_workers,
            debug=debug,
        )
        label_map = {item["question"]: item.get("label", "") for item in q_match_list}
        for question_item in question_info:
            question_item["label"] = label_map.get(question_item["question"], "")
    else:
        for question_item in question_info:
            question_item["label"] = ""
    # 5. generate answers
    qa_list = generatr_qa_pairs(
        question_info=question_info,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        question_number=question_number,
        max_workers=max_workers,
        domain_tree=domain_tree if use_tree_label else None,
        debug=debug,
    )
    
    # Return both qa_list and domain_tree
    return {
        'qa_pairs': qa_list,
        'domain_tree': domain_tree
    }
