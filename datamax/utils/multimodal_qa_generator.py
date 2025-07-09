# datamax/utils/multimodal_qa_generator.py

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import dashscope
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

lock = threading.Lock()

def get_instruction_prompt(question_number: int) -> str:
    """
    Generate a general instruction to tell the model what to do.
    """
    prompt = f"""
        # 角色
        你是一位顶尖的多模态数据标注专家，专门从包含文本和图片的内容中创建高质量的视觉问答（VQA）训练数据。

        # 任务
        根据用户提供的上下文文本和图片，生成 {question_number} 组高质量、多样化且富有想象力的问答对话。

        ## 核心要求
        1.  **强视觉关联**：问题必须与图片内容紧密相关，需要用户仔细观察图片才能回答。
        2.  **对话形式**：每个问答对需以多轮对话格式呈现，至少包含一个用户问题和一个助手回答。
        3.  **多样性**：
            -   **问题类型**：涵盖细节识别（“图片右下角是什么？”）、比较分析（“两张图片有何不同？”）、概念推理（“这张图片中的事物具有什么功能？”）、逻辑分析（“使用图片和公式可以解决什么问题？”）等。
            -   **创意性**：提出一些非常规、需要深度思考或想象力才能回答的问题。
        4.  **忠于原文**：回答应基于上下文文本和合理的图片内容推断，避免捏造信息。
        5. 问题应具有明确答案指向性，覆盖内容的不同方面。
        6. 禁止生成假设性、重复或相似问题，确保生成的完整性。

        ## 处理流程
        1. 【内容解析】分段处理内容，识别关键实体和核心概念
        2. 【问题生成】基于信息密度选择最佳提问点
        3. 【质量检查】确保：
           - 问题答案可在原文中找到依据
           - 标签与问题内容强相关
           - 无格式错误
        
        ## 输出格式
        - **必须**以一个JSON数组的形式输出，数组中包含 {question_number} 个独立的问答对象。
        - 每个对象都必须严格遵循以下结构，不要添加任何额外的解释或文字。
        
        ```json
        [
          {{
            "user": "用户的第一个问题",
            "assistant": "助手的第一个回答"
          }},
          {{
            "user": "用户的第二个问题",
            "assistant": "助手的第二个回答"
          }}
        ]
        ```

        ## 约束
        - 严格按照要求的JSON格式输出，不要输出任何其他内容。
        - 生成的JSON数组必须正好包含 {question_number} 个元素。
    """
    return prompt


def parse_markdown_and_associate_images(md_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Parse Markdown files, extract images, and associate them with text blocks.
    """
    logger.info(f"开始解析Markdown文件: {md_path}")
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        image_pattern = r'!\[[^\]]*\]\(([^)]+)\)'
        image_paths_original = re.findall(image_pattern, content)
        
        if not image_paths_original:
            logger.warning(f"在文件 {md_path} 中未找到任何Markdown格式的图片链接。")
            return []
        
        logger.info(f"在文件中找到 {len(image_paths_original)} 个图片链接。")

        placeholder_template = "||image_placeholder_{}||"
        path_iter = iter(range(len(image_paths_original)))
        
        def unique_replacer(match):
            return placeholder_template.format(next(path_iter))

        content_with_unique_placeholders = re.sub(image_pattern, unique_replacer, content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(content_with_unique_placeholders)

        processed_chunks = []
        placeholder_regex = re.compile(r"\|\|image_placeholder_(\d+)\|\|")
        md_dir = os.path.dirname(os.path.abspath(os.sep.join(md_path.split(os.sep)[:-1])))

        for chunk_text in chunks:
            found_indices = [int(idx) for idx in placeholder_regex.findall(chunk_text)]
            if not found_indices:
                continue
            
            clean_chunk_text = re.sub(placeholder_regex, '', chunk_text).strip()
            unique_indices = sorted(list(set(found_indices)))
            
            chunk_image_paths = [
                os.path.abspath(os.path.join(md_dir, image_paths_original[i]))
                for i in unique_indices
            ]

            processed_chunks.append({
                "text": clean_chunk_text,
                "images": chunk_image_paths
            })
        
        logger.info(f"成功解析并关联了 {len(processed_chunks)} 个包含图片的文本块。")
        return processed_chunks
    except Exception as e:
        logger.error(f"处理Markdown文件 {md_path} 失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def generate_multimodal_qa_with_dashscope(
    api_key: str,
    model: str,
    instruction_prompt: str,
    context_text: str,
    image_paths: List[str],
    temperature: float = 0.7,
) -> List[Dict[str, str]]:
    """
    Generate content and parse JSON output using the DashScope multimodal dialogue API
    """
    try:
        dashscope.api_key = api_key
        
        user_content = []
        for path in image_paths:
            local_file_path = f'file://{os.path.abspath(path)}'
            user_content.append({'image': local_file_path})
        
        user_content.append({'text': f"这是你需要处理的上下文文本：\n\n---\n{context_text}\n---"})
        
        messages = [
            {'role': 'system', 'content': [{'text': instruction_prompt}]},
            {'role': 'user', 'content': user_content}
        ]
        
        response = dashscope.MultiModalConversation.call(
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature
        )

        if response.status_code == 200:
            output_content = response.output.choices[0].get('message', {}).get('content')

            # --- BUG修复点 ---
            # 检查返回的 content 是列表还是字符串
            if isinstance(output_content, list) and output_content:
                # 如果是列表，提取第一个元素的'text'内容
                text_content = output_content[0].get('text')
            elif isinstance(output_content, str):
                # 如果是字符串，直接使用
                text_content = output_content
            else:
                # 其他意外情况，记录错误并返回空
                logger.error(f"API返回的content格式无法识别: {type(output_content)}: {output_content}")
                return []

            if not text_content:
                logger.error("从API返回内容中未能提取到有效文本。")
                return []

            json_match = re.search(r"```json\n([\s\S]*?)\n```", text_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text_content

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}\n原始输出: {json_str}")
                return []
        else:
            logger.error(f"DashScope API调用失败: Code: {response.status_code}, Message: {response.message}")
            return []

    except Exception as e:
        logger.error(f"LLM API调用过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return []

def generatr_qa_pairs(
    file_path: str,
    api_key: str,
    model_name: str,
    chunk_size=2000,
    chunk_overlap=300,
    question_number=2,
    max_workers=5,
    **kwargs,
):
    """
    The main function for generating multimodal question-answer pairs from a Markdown file containing images.
    """
    chunks_with_images = parse_markdown_and_associate_images(
        file_path, chunk_size, chunk_overlap
    )

    if not chunks_with_images:
        logger.warning("未能从文件中解析出任何包含图片的文本块。")
        return []

    final_qa_list = []

    def _process_chunk(chunk_data):
        context_text = chunk_data["text"]
        images = chunk_data["images"]
        
        instruction_prompt = get_instruction_prompt(question_number)
        
        generated_dialogs = generate_multimodal_qa_with_dashscope(
            api_key=api_key,
            model=model_name,
            instruction_prompt=instruction_prompt,
            context_text=context_text,
            image_paths=images,
        )

        chunk_qas = []
        if generated_dialogs and isinstance(generated_dialogs, list):
            for dialog in generated_dialogs:
                if isinstance(dialog, dict) and "user" in dialog and "assistant" in dialog:
                    formatted_qa = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": "<image>"*len(images) + dialog["user"]
                            },
                            {
                                "role": "assistant", 
                                "content": dialog["assistant"]
                            },
                        ],
                        "images": images,
                    }
                    chunk_qas.append(formatted_qa)
        return chunk_qas

    logger.info(f"开始为 {len(chunks_with_images)} 个文本块生成问答对 (线程数: {max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks_with_images]

        with tqdm(as_completed(futures), total=len(futures), desc="生成多模态QA") as pbar:
            for future in pbar:
                result = future.result()
                if result:
                    with lock:
                        final_qa_list.extend(result)
                    pbar.set_postfix({"已生成QA": len(final_qa_list)})

    logger.success(f"处理完成! 共生成 {len(final_qa_list)} 个多模态问答对。")
    return final_qa_list