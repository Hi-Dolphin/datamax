"""
QA Generator for DataMax

This module provides functionality to generate question-answer pairs from processed content.

Features:
- Chinese QA generation (default): Uses Chinese prompts for generating Chinese Q&A pairs
- English QA generation (enhanced): Uses reinforced English-only prompts with strict language constraints
  to ensure pure English output without any Chinese characters

Language Selection:
- language="zh": Generates Chinese questions and answers (default)
- language="en": Generates English questions and answers with enhanced prompts that strictly
  enforce English-only output to prevent any Chinese character leakage

Enhanced English Mode Features:
- Multiple language enforcement checks in prompts
- Explicit "ENGLISH ONLY" requirements in system prompts
- Enhanced user messages for English generation
- Stricter quality controls for English output

Usage:
    # For Chinese QA pairs
    qa_data = generate_qa_from_content(content=text, language="zh", ...)

    # For English QA pairs (enhanced)
    qa_data = generate_qa_from_content(content=text, language="en", ...)
"""

import json
import os.path
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pyexpat.errors import messages

import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm  # For progress bar display

lock = threading.Lock()


# ------------prompt-----------------
def complete_api_url(base_url: str) -> str:
    """
    Normalize the given base_url so that it ends with the OpenAI-style
    chat completions endpoint.

    E.g. if user passes "https://api.provider.com/v1" it will become
    "https://api.provider.com/v1/chat/completions".
    """
    url = base_url.rstrip("/")
    # 如果还没以 /chat/completions 结尾，就自动拼上
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    return url

def get_system_prompt_for_question(query_text, question_number):
    """Generate system prompt for question generation task (Chinese)"""
    system_prompt = f"""
        # 角色使命
        你是一位专业的文本分析专家，擅长从复杂文本中提取关键信息并生成可用于模型微调的结构化数据（仅生成问题）。

        ## 核心任务
        根据用户提供的文本，生成不少于 {question_number} 个高质量问题。

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
        ```json
        ["问题1", "问题2", "..."]
        ```

        ## 输出示例
        ```json
        [ "人工智能伦理框架应包含哪些核心要素？","民法典对个人数据保护有哪些新规定？"]
         ```

        ## 待处理文本
        {query_text}

        ## 限制
        - 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
        - 生成不少于{question_number}个高质量问题
        - 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
        - 问题不得包含【报告、文章、文献、表格】中提到的这种话术，必须是一个自然的问题
    """
    return system_prompt


def get_system_prompt_for_question_en(query_text, question_number):
    """Generate system prompt for question generation task (English)"""
    system_prompt = f"""
        # Role Mission
        You are a professional text analysis expert, skilled in extracting key information from complex texts and generating structured data suitable for model fine-tuning (question generation only).

        ## CRITICAL REQUIREMENT: GENERATE ONLY ENGLISH QUESTIONS
        - ALL questions MUST be in English language only
        - NO Chinese characters or other languages allowed
        - Output format must be English JSON array

        ## Core Task
        Based on the text provided by the user, generate at least {question_number} high-quality questions IN ENGLISH ONLY.

        ## Constraints (Important!)
        - Must be generated directly based on text content
        - Questions should have clear answer directionality
        - Need to cover different aspects of the text
        - Prohibit generating hypothetical, repetitive or similar questions
        - Ensure generation completeness
        - **MANDATORY: All questions must be in English language**

        ## Processing Flow
        1. [Text Analysis] Process content in segments, identify key entities and core concepts
        2. [Question Generation] Select optimal questioning points based on information density
        3. [Quality Check] Ensure:
           - Question answers can be found in the original text
           - Labels are strongly related to question content
           - No format errors
           - **All questions are in English**

        ## Output Format
         - JSON array format must be correct
        - Field names use English double quotes
        - Output JSON array must strictly conform to the following structure:
        - **ALL CONTENT MUST BE IN ENGLISH**
        ```json
        ["English Question 1", "English Question 2", "..."]
        ```

        ## Output Example
        ```json
        ["What are the core elements that should be included in an AI ethics framework?", "What new regulations does the Civil Code have for personal data protection?", "How do machine learning algorithms impact data privacy?"]
         ```

        ## Text to Process
        {query_text}

        ## Restrictions
        - Must output according to the specified JSON format, do not output any other unrelated content
        - Generate at least {question_number} high-quality questions **IN ENGLISH ONLY**
        - Questions should not be related to the material itself, for example, prohibit questions about authors, chapters, catalogs, etc.
        - Questions must not contain phrases like "mentioned in [report, article, literature, table]", must be natural questions
        - **CRITICAL: Absolutely no Chinese characters or non-English content allowed**
        - All questions must be grammatically correct English
    """
    return system_prompt


def get_system_prompt_for_answer(text, query_question):
    """Generate system prompt for answer generation task (Chinese)"""
    system_prompt = f"""
        # Role: 微调数据集生成专家
        ## Profile:
        - Description: 你是一名微调数据集生成专家，擅长从给定的内容中生成准确的问题答案，确保答案的准确性和相关性，你要直接回答用户问题，所有信息已内化为你的专业知识。

        ## Skills:
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
        {text}

        ## 问题
        {query_question}

        ## Constraints:
        1. 答案必须基于给定的内容
        2. 答案必须准确，必须与问题相关，不能胡编乱造
        3. 答案必须充分、详细、包含所有必要的信息、适合微调大模型训练使用
        4. 答案中不得出现 ' 参考 / 依据 / 文献中提到 ' 等任何引用性表述，只需呈现最终结果
    """
    return system_prompt


def get_system_prompt_for_answer_en(text, query_question):
    """Generate system prompt for answer generation task (English)"""
    system_prompt = f"""
        # Role: Fine-tuning Dataset Generation Expert
        
        ## CRITICAL REQUIREMENT: GENERATE ONLY ENGLISH ANSWERS
        - ALL answers MUST be in English language only
        - NO Chinese characters or other languages allowed
        - Response must be in fluent, natural English
        
        ## Profile:
        - Description: You are a fine-tuning dataset generation expert, skilled in generating accurate question answers from given content, ensuring answer accuracy and relevance. You should directly answer user questions in ENGLISH ONLY, with all information internalized as your professional knowledge.

        ## Skills:
        1. Answers must be based on the given content
        2. Answers must be accurate and not fabricated
        3. Answers must be relevant to the questions
        4. Answers must be logical
        5. Based on the given reference content, integrate into a complete answer using natural and fluent **ENGLISH** language, without mentioning literature sources or citation marks
        6. **MANDATORY: All responses must be in English language only**

        ## Workflow:
        1. Take a deep breath and work on this problem step-by-step.
        2. First, analyze the given file content
        3. Then, extract key information from the content
        4. Next, generate accurate answers related to the questions **IN ENGLISH ONLY**
        5. Finally, ensure the accuracy and relevance of the answers in proper English

        ## Reference Content:
        {text}

        ## Question
        {query_question}

        ## Constraints:
        1. Answers must be based on the given content
        2. Answers must be accurate and relevant to the questions, not fabricated
        3. Answers must be comprehensive, detailed, contain all necessary information, and be suitable for fine-tuning large model training
        4. Answers must not contain any referential expressions like 'referenced / based on / mentioned in literature', only present the final results
        5. **CRITICAL: ALL answers must be in English language only - no Chinese characters or other languages allowed**
        6. Use proper English grammar, vocabulary, and sentence structure
        7. Ensure the response flows naturally in English
        
        ## IMPORTANT REMINDER:
        Your response must be entirely in English. Do not include any Chinese characters, phrases, or words in your answer.
    """
    return system_prompt


# ------------spliter----------------
def split_content_to_chunks(content: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Split content into chunks, replacing the file reading approach

    Args:
        content: Processed text content
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    try:
        # Handle potential escaped newlines
        content = content.replace("\\n", "\n")

        # Create document object for the splitter
        document = Document(
            page_content=content, metadata={"source": "processed_content"}
        )

        # Split the document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents([document])
    except Exception as e:
        logger.error(f"Failed to split content: {str(e)}")
        return []


def load_and_split_markdown(md_path: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Parse Markdown file and split into chunks
    This function is kept for backward compatibility

    Args:
        md_path: Path to the markdown file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    try:
        # Read markdown file directly to avoid loader issues
        with open(md_path, "r", encoding="utf-8") as file:
            content = file.read()

        return split_content_to_chunks(content, chunk_size, chunk_overlap)
    except Exception as e:
        logger.error(f"Failed to load {Path(md_path).name}: {str(e)}")
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

    print("Model output not in standard format:", output)
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
            # Determine if this is English mode based on prompt content
            is_english_mode = (
                "GENERATE ONLY ENGLISH" in prompt or "ENGLISH ONLY" in prompt
            )

            if is_english_mode:
                user_message = "Please generate content strictly according to requirements. IMPORTANT: Generate ONLY English content - no Chinese characters or other languages allowed."
            else:
                user_message = "请严格按照要求生成内容"

            message = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
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
        print(f"LLM keyword extraction failed: {e, e.__traceback__.tb_lineno}")
        return []


# ------------thread_process-------------


def process_questions(
    api_key: str,
    model: str,
    base_url: str,
    page_content: list,
    question_number: int,
    language: str = "zh",
    message: list = None,
    max_workers: int = 5,
) -> list:
    """Generate questions using multi-threading"""
    total_questions = []

    def _generate_questions(page):
        """Inner function for question generation"""
        if language.lower() == "en":
            prompt = get_system_prompt_for_question_en(page, question_number)
        else:
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

    logger.info(
        f"Starting question generation (threads: {max_workers}, language: {language})..."
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_questions, page) for page in page_content]

        with tqdm(
            as_completed(futures), total=len(futures), desc="Generating questions"
        ) as pbar:
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
    language: str = "zh",
    message: list = None,
    max_workers=5,
) -> dict:
    """Generate answers using multi-threading"""
    qa_pairs = {}

    def _generate_answer(item):
        """Inner function for answer generation"""
        if language.lower() == "en":
            prompt = get_system_prompt_for_answer_en(item["page"], item["question"])
        else:
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

    logger.info(
        f"Starting answer generation (threads: {max_workers}, language: {language})..."
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_answer, item): item for item in question_items
        }

        with tqdm(
            as_completed(futures), total=len(futures), desc="Generating answers"
        ) as pbar:
            for future in pbar:
                question, answer = future.result()
                if answer:
                    with lock:
                        qa_pairs[question] = answer
                    pbar.set_postfix({"Generated answers": len(qa_pairs)})
    return qa_pairs


def generate_qa_from_content(
    content: str,
    api_key: str,
    base_url: str,
    model_name: str,
    chunk_size=500,
    chunk_overlap=100,
    question_number=5,
    language: str = "zh",
    message: list = None,
    max_workers=5,
):
    """
    Generate QA pairs from processed content

    Args:
        content: Processed text content (from get_data() content field)
        api_key: API key
        base_url: API base URL
        model_name: Model name
        chunk_size: Chunk size
        chunk_overlap: Overlap length
        question_number: Number of questions generated per chunk
        language: Language for QA generation ("zh" for Chinese, "en" for English)
        message: Custom message
        max_workers: Number of concurrent workers

    Returns:
        List of QA pairs
    """
    # 1. Split content into chunks
    pages = split_content_to_chunks(
        content=content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    page_content = [i.page_content for i in pages]
    logger.info(f"Content split into {len(page_content)} chunks")

    # 2. Generate questions using multi-threading
    questions = process_questions(
        page_content=page_content,
        question_number=question_number,
        language=language,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    if not questions:
        logger.error(
            "Failed to generate any questions, please check input content and API settings"
        )
        return []

    # 3. Generate answers using multi-threading
    qa_pairs = process_answers(
        question_items=questions,
        language=language,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )

    logger.success(f"Completed! Generated {len(qa_pairs)} QA pairs in {language}")

    # Format results
    res_list = []
    for question, answer in qa_pairs.items():
        qa_entry = {"instruction": question, "input": "", "output": answer}
        res_list.append(qa_entry)
    return res_list


def generatr_qa_pairs(
    file_path: str,
    api_key: str,
    base_url: str,
    model_name: str,
    chunk_size=500,
    chunk_overlap=100,
    question_number=5,
    language: str = "zh",
    message: list = None,
    max_workers=5,
):
    """Main function to generate QA pairs from markdown file (kept for backward compatibility)"""
    # 1. Split markdown text into chunks
    pages = load_and_split_markdown(
        md_path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    page_content = [i.page_content for i in pages]
    logger.info(f"Markdown split into {len(page_content)} chunks")

    # 2. Generate questions using multi-threading
    questions = process_questions(
        page_content=page_content,
        question_number=question_number,
        language=language,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    if not questions:
        logger.error(
            "Failed to generate any questions, please check input document and API settings"
        )

    # 3. Generate answers using multi-threading
    qa_pairs = process_answers(
        question_items=questions,
        language=language,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )

    logger.success(f"Completed! Generated {len(qa_pairs)} QA pairs in {language}")

    # Format results
    res_list = []
    for question, answer in qa_pairs.items():
        qa_entry = {"instruction": question, "input": "", "output": answer}
        res_list.append(qa_entry)
    return res_list


if __name__ == "__main__":
    # Example 1: Generate Chinese QA pairs (default)
    print("Generating Chinese QA pairs...")
    generatr_qa_pairs(
        file_path=r"C:\Users\example\Desktop\document\knowledge_graph\knowledge_graph_design.md",
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_name="qwen-max",
        chunk_size=500,
        chunk_overlap=100,
        question_number=5,
        language="zh",  # Chinese QA pairs
        max_workers=5,
        # message=[]
    )

    # Example 2: Generate English QA pairs (Enhanced with strict English-only prompts)
    print("\nGenerating English QA pairs with enhanced prompts...")
    # generatr_qa_pairs(
    #     file_path=r"C:\Users\example\Desktop\document\knowledge_graph\knowledge_graph_design.md",
    #     api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    #     model_name="qwen-max",
    #     chunk_size=500,
    #     chunk_overlap=100,
    #     question_number=5,
    #     language="en",  # English QA pairs with enhanced prompts
    #     max_workers=5,
    # )
