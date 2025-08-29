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
from datamax.utils.domain_tree import DomainTree   # for cache domain tree

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


# ------------prompt-----------------
def get_system_prompt_for_match_label(tags_json, question):
    system_prompt = f"""
    # Role: 标签匹配专家
    - Description: 你是一名标签匹配专家，擅长根据给定的标签数组和问题数组，将问题打上最合适的领域标签。你熟悉标签的层级结构，并能根据问题的内容优先匹配二级标签，若无法匹配则匹配一级标签，若无法匹配最后打上"其他"标签。

    ### Skill:
    1. 熟悉标签层级结构，能够准确识别一级和二级标签。
    2. 能够根据问题的内容，智能匹配最合适的标签。
    3. 能够处理复杂的标签匹配逻辑，确保每个问题都能被打上正确的标签。
    4. 能够按照规定的输出格式生成结果，确保不改变原有数据结构。
    5. 能够处理大规模数据，确保高效准确的标签匹配。

    ## Goals:
    1. 将问题数组中的每个问题打上最合适的领域标签。
    2. 优先匹配二级标签，若无法匹配则匹配一级标签，最后打上"其他"标签。
    3. 确保输出格式符合要求，不改变原有数据结构。
    4. 提供高效的标签匹配算法，确保处理大规模数据时的性能。
    5. 确保标签匹配的准确性和一致性。

    ## OutputFormat:
    1. 输出结果必须是一个数组，每个元素包含 question、和 label 字段。
    2. label 字段必须是根据标签数组匹配到的标签，若无法匹配则打上"其他"标签。
    3. 不改变原有数据结构，只新增 label 字段。

    ## 标签json：

    ${tags_json}

    ## 问题数组：

    ${question}


    ## Workflow:
    1. Take a deep breath and work on this problem step-by-step.
    2. 首先，仔细分析每个问题的核心内容和关键词。
    3. 然后，遍历问题数组中的每个问题，根据问题的内容匹配标签数组中的标签。
    4. 优先匹配二级标签，若无法匹配则匹配一级标签，最后打上"其他"标签。
    5. 将匹配到的标签添加到问题对象中，确保不改变原有数据结构。
    6. 最后，输出结果数组，确保格式符合要求。


    ## Constrains:
    1. 只新增一个 label 字段，不改变其他任何格式和数据。
    2. 必须按照规定格式返回结果。
    3. 优先匹配二级标签，若无法匹配则匹配一级标签，最后打上"其他"标签。尽量不匹配"其他"标签。
    4. 确保标签匹配的准确性和一致性。
    5. 匹配的标签必须来自标签数组，如果无法匹配任何标签，就打上"其他"标签。
    6. 输出结果必须是一个数组，每个元素包含 question、label 字段（只输出这个，不要输出任何其他无关内容）。
    7. 仔细分析问题内容，寻找与标签的语义关联。
    8. 如果问题内容与多个标签相关，选择最匹配的一个。
    9. 考虑问题的核心主题和关键词，进行精确匹配。

    ## Output Example:
    ```json
        [
            {{
                "question": "XSS为什么会在2003年后引起人们更多关注并被OWASP列为威胁榜首？",
                "label": "2.2 XSS攻击"
            }},
            {{
                "question": "这个问题与现有标签都不相关",
                "label": "其他"
            }}
        ]
    ```
    """
    return system_prompt


def get_system_prompt_for_domain_tree(text):
    """Generate system prompt for domain tree task"""
    system_prompt = f"""
        #  Role: 领域分类专家 & 知识图谱专家
        - Description:
        作为一名资深的领域分类专家和知识图谱专家，擅长从文本内容中提取核心主题，构建分类体系，
        并输出规定 JSON 格式的标签树。

        ## Skills:
        1. 精通文本主题分析和关键词提取
        2. 擅长构建分层知识体系
        3. 熟练掌握领域分类方法论
        4. 具备知识图谱构建能力
        5. 精通JSON数据结构

        ## Goals:
        1. 分析书籍目录内容
        2. 识别核心主题和关键领域
        3. 构建两级分类体系
        4. 确保分类逻辑合理
        5. 生成规范的JSON输出

        ## Workflow:
        1. 仔细阅读完整的书籍目录内容
        2. 提取关键主题和核心概念
        3. 对主题进行分组和归类
        4. 构建一级领域标签
        5. 为适当的一级标签添加二级标签
        6. 检查分类逻辑的合理性
        7. 生成符合格式的JSON输出
        

        ## 需要分析的目录
        ${text}

        ## 限制
        1. 一级领域标签数量5-10个
        2. 二级领域标签数量1-10个
        3. 最多两层分类层级
        4. 分类必须与原始目录内容相关
        5. 输出必须符合指定 JSON 格式，不要输出 JSON 外其他任何不相关内容
        6. 标签的名字最多不要超过 6 个字
        7. 在每个标签前加入序号（序号不计入字数）

        ## OutputFormat:
        ```json
        [
            {{
                "label": "1 一级领域标签",
                "child": [
                    {{"label": "1.1 二级领域标签1"}},
                    {{"label": "1.2 二级领域标签2"}}
                ]
            }},
            {{
                "label": "2 一级领域标签(无子标签)"
            }}
        ]
        ```
    """
    return system_prompt


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
        ```json
        ["问题1", "问题2", "..."]
        ```

        ## 输出示例
        ```json
        [ "人工智能伦理框架应包含哪些核心要素？","民法典对个人数据保护有哪些新规定？"]
        ```

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
            "temperature": temperature,
            "top_p": top_p,
        }

        response = requests.post(base_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()

        # Parse LLM response
        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            if type == "question":
                fmt_output = extract_json_from_llm_output(output)
                return fmt_output if fmt_output is not None else []
            else:
                return [output] if output else []
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
) -> DomainTree:
    prompt = get_system_prompt_for_domain_tree(text)
    logger.info(f"Domain tree generation started...")
    
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

            # Parse LLM response
            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0]["message"]["content"]
                if output:
                    json_output = extract_json_from_llm_output(output)
                    if json_output is not None:
                        domain_tree = DomainTree()
                        domain_tree.from_json(json_output)
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
) -> list:
    if message is None:
        message = []
    if domain_tree is None:
        from datamax.utils.domain_tree import DomainTree
        domain_tree = DomainTree([])
    qa_pairs = process_answers(
        question_items=question_info,
        message=message,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    logger.success(
        f"Completed! Generated {len(qa_pairs)} QA pairs in total"
    )
    res_list = []
    for question_item in question_info:
        question = question_item["question"]
        relative_chuck = question_item["page"]
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
                "relative_chuck": relative_chuck,
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
):
    """
    Complete QA generation workflow, including splitting, domain tree generation and interaction, 
    question generation, label tagging, and answer generation.
    """
    import uuid

    from datamax.utils.qa_generator import (
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
        from datamax.utils.domain_tree import DomainTree

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
    )
    return qa_list
