import json
import os.path
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Any
import uuid

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from loguru import logger
from pyexpat.errors import messages
from tqdm import tqdm  
from dotenv import load_dotenv
from datamax.utils.domain_tree import DomainTree   #for cache domain tree

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
    # 如果还没以 /chat/completions 结尾，就自动拼上
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
        logger.info(f"开始切分markdown文件...")
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
        logger.info(f"markdown被分解了{len(page_content)}个chunk")
        return page_content


    except Exception as e:
        logger.error(f"加载 {Path(md_path).name} 失败: {str(e)}")
        return []


def load_and_split_text(file_path: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Parse other formats to markdown and split
    
    Args:
        file_path: Path to the markdown file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    try:
        from datamax.parser.core import DataMax
        
        logger.info(f"开始处理文件: {file_path}")
        
        # 使用DataMax解析文件，自动转换为markdown格式
        dm = DataMax(file_path=file_path, to_markdown=True)
        parsed_data = dm.get_data()
        
        if not parsed_data:
            logger.error(f"文件解析失败: {file_path}")
            return []
            
        # 获取解析后的内容
        if isinstance(parsed_data, list):
            # 如果是多个文件，取第一个
            content = parsed_data[0].get('content', '')
        else:
            content = parsed_data.get('content', '')
            
        if not content:
            logger.error(f"文件内容为空: {file_path}")
            return []
            
        # 使用LangChain的文本分割器进行切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # 直接分割文本内容
        page_content = splitter.split_text(content)
        logger.info(f"文件被分解了{len(page_content)}个chunk")
        return page_content
        
    except Exception as e:
        logger.error(f"处理文件 {Path(file_path).name} 失败: {str(e)}")
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

    logger.error(f"模型未按标准格式输出: {output}")
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
        logger.error(f"LLM提取关键词失败: {e}")
        if hasattr(e, "__traceback__") and e.__traceback__ is not None:
            logger.error(f"错误行号: {e.__traceback__.tb_lineno}")
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
    max_workers: int = 3
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger.info(f"开始并发生成问题匹配标签... (max_workers={max_workers})")
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
            #print(f"问题: {res.get('question', '')} | 匹配标签: {res.get('label', '')}")
            results.append(res)
    logger.success(f"问题匹配标签生成成功, 共生成 {len(results)} 个问题")
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
    logger.info(f"领域树生成开始...")
    
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
                        logger.info(f"领域树生成成功, 共生成 {len(json_output)} 个大标签")
                        return domain_tree
                    else:
                        logger.warning(f"领域树生成失败 (尝试 {attempt + 1}/{max_retries}): 无法解析JSON输出")
                else:
                    logger.warning(f"领域树生成失败 (尝试 {attempt + 1}/{max_retries}): 空输出")
            else:
                logger.warning(f"领域树生成失败 (尝试 {attempt + 1}/{max_retries}): 无效响应格式")
                
        except Exception as e:
            logger.error(f"领域树生成异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                logger.error(f"错误行号: {e.__traceback__.tb_lineno}")
            
            if attempt == max_retries - 1:
                error_msg = "树生成失败！请检查网络或更换大模型！后续将依据纯文本生成"
                print(f"❌ {error_msg}")
                logger.error(f"领域树生成失败，已重试 {max_retries} 次: {error_msg}")
                return None
            else:
                logger.info(f"等待重试... ({attempt + 2}/{max_retries})")
                import time
                time.sleep(2)  # 等待2秒后重试
    
    error_msg = "树生成失败！请检查网络或更换大模型！后续将依据纯文本生成"
    print(f"❌ {error_msg}")
    logger.error(f"领域树生成失败，已重试 {max_retries} 次: {error_msg}")
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
                    return [{"question": question, "page": page} for question in questions]
                else:
                    logger.warning(f"问题生成失败 (尝试 {attempt + 1}/{max_retries}): 空结果")
            except Exception as e:
                logger.error(f"问题生成异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                    logger.error(f"错误行号: {e.__traceback__.tb_lineno}")
            
            if attempt < max_retries - 1:
                logger.info(f"等待重试... ({attempt + 2}/{max_retries})")
                import time
                time.sleep(2)  # 等待2秒后重试
        
        logger.error(f"问题生成失败，已重试 {max_retries} 次")
        return []

    logger.info(f"开始生成问题 (线程数: {max_workers}, 重试次数: {max_retries})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_questions_with_retry, page) for page in page_content]
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
    message: Optional[list] = None,
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
                    logger.warning(f"答案生成失败 (尝试 {attempt + 1}/{max_retries}): 空结果")
            except Exception as e:
                logger.error(f"答案生成异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                if hasattr(e, "__traceback__") and e.__traceback__ is not None:
                    logger.error(f"错误行号: {e.__traceback__.tb_lineno}")
            
            if attempt < max_retries - 1:
                logger.info(f"等待重试... ({attempt + 2}/{max_retries})")
                import time
                time.sleep(2)  # retry after 2 seconds
        
        # all retries failed
        question_text = item["question"][:20] + "..." if len(item["question"]) > 20 else item["question"]
        logger.error(f"网络状态不佳！舍弃了（{question_text}）问题的对应问答对")
        return None  # 返回None表示舍弃该问答对

    logger.info(f"开始生成答案 (线程数: {max_workers}, 重试次数: {max_retries})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_answer_with_retry, item): item for item in question_items
        }

        with tqdm(as_completed(futures), total=len(futures), desc="生成答案") as pbar:
            for future in pbar:
                result = future.result()
                if result is not None:  # 只有成功生成答案的才添加
                    question, answer = result
                    with lock:
                        qa_pairs[question] = answer
                    pbar.set_postfix({"已生成答案": len(qa_pairs)})
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
        f"完成! 共生成 {len(qa_pairs)} 个问答对"
    )
    res_list = []
    for question_item in question_info:
        question = question_item["question"]
        # 只有成功生成答案的问题才会被添加到结果中
        if question in qa_pairs:
            label = question_item.get("label", "")
            answer = qa_pairs[question]
            tag_path = find_tagpath_by_label(domain_tree, label) if domain_tree else ""
            qid = question_item.get("qid", "")
            method = "text with tree label" if domain_tree else "text"
            qa_entry = {
                "qid": qid,
                "instruction": question,
                "input": "",
                "output": answer,
                "label": label,
                "tag-path": tag_path,
                "method": method
            }
            res_list.append(qa_entry)
    return res_list


def _interactive_tree_modification(domain_tree):
    """
    交互式自定义领域树结构
    :param domain_tree: DomainTree实例
    :return: 修改后的DomainTree实例
    """
    print("\n 是否需要进行树修改？")
    print("支持的操作:")
    print("1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）")
    print("2. 增加节点：xxx；父节点：xxx；子节点：xxx")
    print("3. 删除节点：xxx")
    print("4. 更新节点：新名称；原先节点：旧名称")
    print("5. 结束树操作")
    print("注意，节点的格式通常为：x.xx xxxx,如：‘1.1 货物运输组织与路径规划’或‘1 运输系统组织’")
    print("\n请输入操作指令（输入'结束树操作'退出）:")
    while True:
        try:
            user_input = input("> ").strip()
            if user_input == "结束树操作":
                print("✅ 树操作结束，继续QA对生成...")
                break
            elif user_input.startswith("增加节点："):
                parts = user_input.split("；")
                if len(parts) >= 2:
                    node_name = parts[0].replace("增加节点：", "").strip()
                    parent_name = parts[1].replace("父节点：", "").strip()
                    if not parent_name:
                        if domain_tree.add_node(node_name):
                            print(f"✅ 成功将节点 '{node_name}' 作为根节点添加")
                        else:
                            print(f"❌ 添加失败：未知错误")
                    elif len(parts) == 2:
                        if domain_tree.add_node(node_name, parent_name):
                            print(f"✅ 成功添加节点 '{node_name}' 到父节点 '{parent_name}' 下")
                        else:
                            print(f"❌ 添加失败：未找到父节点 '{parent_name}'")
                    elif len(parts) == 3:
                        child_name = parts[2].replace("子节点：", "").strip()
                        if domain_tree.insert_node_between(node_name, parent_name, child_name):
                            print(f"✅ 成功插入节点 '{node_name}' 到 '{parent_name}' 和 '{child_name}' 之间")
                        else:
                            print(f"❌ 插入失败：请检查父节点和子节点的关系")
                    else:
                        print("❌ 格式错误：请使用正确的格式")
                else:
                    print("❌ 格式错误：请使用正确的格式")
            elif user_input.startswith("删除节点："):
                node_name = user_input.replace("删除节点：", "").strip()
                if domain_tree.remove_node(node_name):
                    print(f"✅ 成功删除节点 '{node_name}' 及其所有子孙节点")
                else:
                    print(f"❌ 删除失败：未找到节点 '{node_name}'")
            elif user_input.startswith("更新节点："):
                parts = user_input.split("；")
                if len(parts) == 2:
                    new_name = parts[0].replace("更新节点：", "").strip()
                    old_name = parts[1].replace("原先节点：", "").strip()
                    if domain_tree.update_node(old_name, new_name):
                        print(f"✅ 成功将节点 '{old_name}' 更新为 '{new_name}'")
                    else:
                        print(f"❌ 更新失败：未找到节点 '{old_name}'")
                else:
                    print("❌ 格式错误：请使用正确的格式，如：更新节点：新名称；原先节点：旧名称")
            else:
                print("❌ 未知操作，请使用正确的格式")
            print("\n📝 当前树结构:")
            print(domain_tree.visualize())
            print("\n请输入下一个操作指令:")
            print("支持的操作:")
            print("1. 增加节点：xxx；父节点：xxx   （父节点可留空，留空则添加为根节点）")
            print("2. 增加节点：xxx；父节点：xxx；子节点：xxx")
            print("3. 删除节点：xxx")
            print("4. 更新节点：新名称；原先节点：旧名称")
            print("5. 结束树操作")
            print("注意，节点的格式通常为：x.xx xxxx,如：‘1.1 货物运输组织与路径规划’或‘1 运输系统组织’")
        except KeyboardInterrupt:
            print("\n\n⚠️⚠️操作被中断⚠️⚠️，继续QA对生成...")
            break
        except Exception as e:
            print(f"❌ 操作出错：{e}")
            print("请重新输入操作指令:")
    return domain_tree


def full_qa_labeling_process(
    content: str,
    api_key: str,
    base_url: str,
    model_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    question_number: int = 5,
    max_workers: int = 5,
    use_tree_label: bool = True,
    messages: list = None,
    interactive_tree: bool = True,
    custom_domain_tree: list = None,
):
    """
    封装完整的QA生成流程，包括分割、领域树生成与交互、问题生成、标签打标、答案生成。
    """
    from datamax.utils.qa_generator import (
        process_domain_tree,
        process_questions,
        process_match_tags,
        generatr_qa_pairs
    )
    import uuid

    # 1.split content
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    page_content = splitter.split_text(content)
    # 2.generate domain tree
    domain_tree = None
    if use_tree_label:
        from datamax.utils.domain_tree import DomainTree
        
        # if custom_domain_tree is not None, use it
        if custom_domain_tree is not None:
            domain_tree = DomainTree(custom_domain_tree)
            logger.info("🌳 使用用户上传的自定义领域树结构")
            print("🌳 正在使用您上传的自定义领域树结构进行预标注...")
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
                logger.info("领域树生成失败，采用纯文本生成策略")
                use_tree_label = False
        
        # 统一的交互式编辑逻辑
        if interactive_tree and domain_tree and domain_tree.tree:
            tree_source = "自定义" if custom_domain_tree is not None else "生成"
            print("\n" + "="*60)
            print(f"🌳 {tree_source}的领域树结构:")
            print("="*60)
            print(domain_tree.visualize())
            print("="*60)
            if custom_domain_tree is not None:
                print("💡 您可以对自定义树进行修改，或输入'结束树操作'直接使用")
            domain_tree = _interactive_tree_modification(domain_tree)
    #generate questions
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
    # 4.label tagging
    if use_tree_label and domain_tree and hasattr(domain_tree, 'to_json') and domain_tree.to_json():
        q_match_list = process_match_tags(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            tags_json=domain_tree.to_json(),
            questions=[q["question"] for q in question_info],
            max_workers=max_workers
        )
        label_map = {item["question"]: item.get("label", "") for item in q_match_list}
        for question_item in question_info:
            question_item["label"] = label_map.get(question_item["question"], "")
    else:
        for question_item in question_info:
            question_item["label"] = ""
    # 5.generate answers
    qa_list = generatr_qa_pairs(
        question_info=question_info,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        question_number=question_number,
        max_workers=max_workers,
        domain_tree=domain_tree if use_tree_label else None
    )
    return qa_list


if __name__ == "__main__":
    # split text into chunks
    page_content = load_and_split_markdown(
        md_path="知识图谱.md",  
        chunk_size=500,
        chunk_overlap=100,
    )

    # generate domain tree
    domain_tree = process_domain_tree(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="qwen-plus",
        text=page_content,
        temperature=0.7,
        top_p=0.9,
    )

    # generate question_info containing chuck and questions
    # question_info is the largest question set, will be adjusted according to the modification of the domain tree
    question_info = process_questions(
        page_content=page_content,
        question_number=5,  
        max_workers=10,  
        api_key=API_KEY,
        base_url=BASE_URL,
        model="qwen-plus",
    )

    # add unique id to each question
    for question_item in question_info:
        question_item["qid"] = str(uuid.uuid4())

    if not question_info:
        logger.error("未能生成任何问题，请检查输入文档和API设置")
        
    # check if domain_tree is empty
    if not domain_tree or not domain_tree.to_json():
        logger.info("领域树为空, 未进行打标")
    else:
        # use DomainTree instance to match label
        q_match_list = process_match_tags(
            api_key=API_KEY,
            base_url=BASE_URL,
            model="qwen-plus",
            tags_json=domain_tree.to_json(),
            questions= [question_item["question"] for question_item in question_info],
            max_workers=3
        )
        logger.info(f"问题匹配标签完成, 结果是: {q_match_list}")
        # merge label to question_info
        label_map = {item["question"]: item.get("label", "") for item in q_match_list}
        for question_item in question_info:
            question_item["label"] = label_map.get(question_item["question"], "")
        # get filtered question_info
        question_list = [question_item["question"] for question_item in question_info]
        question_info = [{"question": question_item["question"], "page": question_item["page"], "qid": question_item["qid"], "label": question_item["label"]} for question_item in question_info if question_item["question"] in question_list]

    # final answer
    r = generatr_qa_pairs(
        question_info=question_info,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name="qwen-plus",
        question_number=5,  
        max_workers=10,  
        domain_tree=domain_tree
        # message=[] 
    )

    print(r)
