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
from tqdm import tqdm  # For progress bar display

lock = threading.Lock()


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
        print(f"LLM提取关键词失败: {e}")
        if hasattr(e, "__traceback__") and e.__traceback__ is not None:
            print(f"错误行号: {e.__traceback__.tb_lineno}")
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
):
    prompt = get_system_prompt_for_domain_tree(text)

    logger.info(f"领域树生成开始...")

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
        # save result
        if output:
            json_output = extract_json_from_llm_output(output)
            if json_output is not None:
                with open("./tags.json", "w", encoding="utf-8") as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                logger.info(f"领域树生成成功, 共生成 {len(json_output)} 个大标签")
                return json_output

    return []


def process_questions(
    api_key: str,
    model: str,
    base_url: str,
    page_content: list,
    question_number: int,
    max_workers: int = 5,
    message: list = None,
) -> list:
    """Generate questions using multi-threading"""
    total_questions = []


    def _generate_questions(page, message):
        """Inner function for question generation"""
        prompt = get_system_prompt_for_question(page, question_number)
        if not message:
            message = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请严格按照要求生成内容"},
            ]

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
        futures = [executor.submit(_generate_questions, page, message) for page in page_content]


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


# find tagpath by label

def find_tagpath_by_label(domain_tree, label, path=None):
    if path is None:
        path = []
    
    # check if domain_tree is a list
    if not isinstance(domain_tree, list):
        logger.warning(f"domain_tree 不是列表类型: {type(domain_tree)}")
        return None
        
    for node in domain_tree:
        if not isinstance(node, dict):
            continue
        current_label = node.get('label')
        current_path = path + [current_label]
        if current_label == label:
            return '/'.join(current_path)
        # recursive find child node
        if 'child' in node and isinstance(node['child'], list):
            result = find_tagpath_by_label(node['child'], label, current_path)
            if result:
                return result
    return None



def generatr_qa_pairs(
    question_info: list,
    api_key: str,
    base_url: str,
    model_name: str,
    question_number=5,
    message: list = None,
    max_workers=5,
    domain_tree=None,  
):
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
    for q in question_info:
        question = q["question"]
        label = q.get("label", "")
        answer = qa_pairs.get(question, "")
        tag_path = find_tagpath_by_label(domain_tree, label) if domain_tree else ""
        qid = q.get("qid", "")
        qa_entry = {
            "qid": qid,
            "instruction": question,
            "input": "",
            "output": answer,
            "label": label,
            "tag-path": tag_path,
            "method": "text"
        }
        res_list.append(qa_entry)
    return res_list


if __name__ == "__main__":
    # split text into chunks
    page_content = load_and_split_markdown(
        md_path=r"C:\Users\lysnoir\OneDrive\桌面\15349549.md",
        chunk_size=500,
        chunk_overlap=100,
    )

    # generate domain tree
    domain_tree = process_domain_tree(
        api_key="sk-0079be2ceda64ece81e0810e2b6c7be9",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
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
        api_key="sk-0079be2ceda64ece81e0810e2b6c7be9",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model="qwen-plus",
    )

    # add unique id to each question
    for q in question_info:
        q["qid"] = str(uuid.uuid4())

    if not question_info:
        logger.error("未能生成任何问题，请检查输入文档和API设置")

    # if tags.json exists, filter question_info
    if not os.path.exists("./tags.json"):
        logger.info("tags.json 文件不存在, 未进行打标")
    else:
        # if tags.json exists, filter question_info
        with open("./tags.json", "r", encoding="utf-8") as f:
            tags_json = json.load(f)
        # question matching label q_match_list must be less than or equal to the questions in question_info
        q_match_list = process_match_tags(
            api_key="sk-0079be2ceda64ece81e0810e2b6c7be9",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            model="qwen-plus",
            tags_json=domain_tree,
            questions= [q["question"] for q in question_info],
            max_workers=3
        )
        logger.info(f"问题匹配标签完成, 结果是: {q_match_list}")
        # merge label to question_info
        label_map = {item["question"]: item.get("label", "") for item in q_match_list}
        for q in question_info:
            q["label"] = label_map.get(q["question"], "")
        # generate unique id-label mapping and save
        qid_label_map = {q["qid"]: q.get("label", "") for q in question_info}
        with open("question_id_label_map.json", "w", encoding="utf-8") as f:
            json.dump(qid_label_map, f, ensure_ascii=False, indent=2)
        # get filtered question_info
        q_list = [i["question"] for i in question_info]
        question_info = [{"question": q["question"], "page": q["page"], "qid": q["qid"], "label": q["label"]} for q in question_info if q["question"] in q_list]

    # final answer
    r = generatr_qa_pairs(
        question_info=question_info,
        api_key="sk-0079be2ceda64ece81e0810e2b6c7be9",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_name="qwen-plus",
        question_number=5,  
        max_workers=10,  
        domain_tree=domain_tree
        # message=[]
    )

    print(r)
