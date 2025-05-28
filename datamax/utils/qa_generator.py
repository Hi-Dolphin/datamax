import json
import os.path
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from loguru import logger
from pyexpat.errors import messages
from tqdm import tqdm  # For progress bar display

lock = threading.Lock()


# ------------prompt-----------------
def get_system_prompt_for_question(query_text, question_number):
    """Generate system prompt for question generation task"""
    system_prompt = f"""
        # 角色使命
        你是一位专业的文本分析专家，擅长从复杂文本中提取关键信息并生成可用于模型微调的结构化数据（仅生成问题）。

        ## 核心任务
        根据用户提供的文本，生成不少于 ${question_number} 个高质量问题。

        ## 约束条件（重要！）
        - 必须基于文本内容直接生成
        - 问题应具有明确答案指向性
        - 需覆盖文本的不同方面
        - 禁止生成假设性、重复或相似问题
        - 确保生成得完整性

        ## 处理流程
        1. 【文本解析】分段处理内容，识别关键实体和核心概念
        2. 【问题生成】基于信息密度选择最佳提问点
        3. 【质量检查】确保：
           - 问题答案可在原文中找到依据
           - 标签与问题内容强相关
           - 无格式错误

        ## 输出格式
         - JSON 数组格式必须正确
        - 字段名使用英文双引号
        - 输出的 JSON 数组必须严格符合以下结构：
        \`\`\`json
        ["问题1", "问题2", "..."]
        \`\`\`

        ## 输出示例
        \`\`\`json
        [ "人工智能伦理框架应包含哪些核心要素？","民法典对个人数据保护有哪些新规定？"]
         \`\`\`

        ## 待处理文本
        ${query_text}

        ## 限制
        - 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
        - 生成不少于${question_number}个高质量问题
        - 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
        - 问题不得包含【报告、文章、文献、表格】中提到的这种话术，必须是一个自然的问题
    """
    return system_prompt


def get_system_prompt_for_answer(text, query_question):
    """Generate system prompt for answer generation task"""
    system_prompt = f"""
        # Role: 微调数据集生成专家
        ## Profile:
        - Description: 你是一名微调数据集生成专家，擅长从给定的内容中生成准确的问题答案，确保答案的准确性和相关性，你要直接回答用户问题，所有信息已内化为你的专业知识。

        ## Skills   :
        1. 答案必须基于给定的内容
        2. 答案必须准确，不能胡编乱造
        3. 答案必须与问题相关
        4. 答案必须符合逻辑
        5. 基于给定参考内容，用自然流畅的语言整合成一个完整答案，不需要提及文献来源或引用标记

        ## Workflow:
        1. Take a deep breath and work on this problem step-by-step.
        2. 首先，分析给定的文件内容
        3. 然后，从内容中提取关键信息
        4. 接着，生成与问题相关的准确答案
        5. 最后，确保答案的准确性和相关性

        ## 参考内容：
        ${text}

        ## 问题
        ${query_question}

        ## Constrains:
        1. 答案必须基于给定的内容
        2. 答案必须准确，必须与问题相关，不能胡编乱造
        3. 答案必须充分、详细、包含所有必要的信息、适合微调大模型训练使用
        4. 答案中不得出现 ' 参考 / 依据 / 文献中提到 ' 等任何引用性表述，只需呈现最终结果
    """
    return system_prompt


# ------------spliter----------------
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
        loader = UnstructuredMarkdownLoader(md_path)
        documents = loader.load()
        # Further split documents if needed
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"加载 {Path(md_path).name} 失败: {str(e)}")
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
            print(f"解析 JSON 时出错: {e}")

    # Try to extract the most JSON-like part
    json_start = output.find("[")
    json_end = output.rfind("]") + 1
    if json_start != -1 and json_end != 0:
        try:
            return json.loads(output[json_start:json_end])
        except json.JSONDecodeError:
            pass

    print("模型未按标准格式输出:", output)
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
    max_token: int = 2048,
) -> list:
    """Generate content using LLM API"""
    try:
        if not message:
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
            "max_tokens": max_token,
            "temperature": temperature,
            "top_p": top_p,
        }
        response = requests.post(base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Parse LLM response
        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            if type == "question":
                fmt_output = extract_json_from_llm_output(output)
            else:
                return output
            return fmt_output
        return []

    except Exception as e:
        print(f"LLM提取关键词失败: {e, e.__traceback__.tb_lineno}")
        return []


# ------------thread_process-------------


def process_questions(
    api_key: str,
    model: str,
    base_url: str,
    page_content: list,
    question_number: int,
    message: list,
    max_workers: int = 5,
) -> list:
    """Generate questions using multi-threading"""
    total_questions = []

    def _generate_questions(page):
        """Inner function for question generation"""
        prompt = get_system_prompt_for_question(page, question_number)
        questions = llm_generator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            message=message,
            prompt=prompt,
            type="question",
        )
        return [{"question": q, "page": page} for q in questions] if questions else []

    logger.info(f"开始生成问题 (线程数: {max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_questions, page) for page in page_content]

        with tqdm(as_completed(futures), total=len(futures), desc="生成问题") as pbar:
            for future in pbar:
                result = future.result()
                if result:
                    with lock:
                        total_questions.extend(result)
                    pbar.set_postfix({"已生成问题": len(total_questions)})

    return total_questions


def process_answers(
    api_key: str,
    model: str,
    base_url: str,
    question_items: list,
    message: list = None,
    max_workers=5,
) -> dict:
    """Generate answers using multi-threading"""
    qa_pairs = {}

    def _generate_answer(item):
        """Inner function for answer generation"""
        prompt = get_system_prompt_for_answer(item["page"], item["question"])
        answer = llm_generator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            prompt=prompt,
            message=message,
            type="answer",
        )
        return item["question"], answer

    logger.info(f"开始生成答案 (线程数: {max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_answer, item): item for item in question_items
        }

        with tqdm(as_completed(futures), total=len(futures), desc="生成答案") as pbar:
            for future in pbar:
                question, answer = future.result()
                if answer:
                    with lock:
                        qa_pairs[question] = answer
                    pbar.set_postfix({"已生成答案": len(qa_pairs)})
    return qa_pairs


def generatr_qa_pairs(
    file_path: str,
    api_key: str,
    base_url: str,
    model_name: str,
    chunk_size=500,
    chunk_overlap=100,
    question_number=5,
    message: list = None,
    max_workers=5,
):
    """Main function to generate QA pairs from markdown file"""
    # 1. Split markdown text into chunks
    pages = load_and_split_markdown(
        md_path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    page_content = [i.page_content for i in pages]
    logger.info(f"markdown被分解了{len(page_content)}个chunk")

    # 2. Generate questions using multi-threading
    questions = process_questions(
        page_content=page_content,
        message=message,
        question_number=question_number,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    if not questions:
        logger.error("未能生成任何问题，请检查输入文档和API设置")

    # 3. Generate answers using multi-threading
    qa_pairs = process_answers(
        question_items=questions,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )

    # 4. Save results
    res_list = []
    with open(
        f"{os.path.basename(file_path).strip('.md')}.jsonl", "w", encoding="utf-8"
    ) as f:
        for question, answer in qa_pairs.items():
            # Build properly formatted JSON object
            qa_entry = {"instruction": question, "input": "", "output": answer}
            res_list.append(qa_entry)
            # Write to JSONL file (one JSON object per line)
            f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")

    logger.success(
        f"完成! 共生成 {len(qa_pairs)} 个问答对，已保存到 {os.path.basename(file_path).strip('.md')}.jsonl"
    )

    return res_list


if __name__ == "__main__":
    generatr_qa_pairs(
        file_path=r"C:\Users\cykro\Desktop\文档整理\知识图谱\知识图谱概要设计.md",
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_name="qwen-max",
        chunk_size=500,
        chunk_overlap=100,
        question_number=5,
        max_workers=5,
        # message=[]
    )
