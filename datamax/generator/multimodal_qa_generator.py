# datamax/generator/multimodal_qa_generator.py

import base64
import json
import mimetypes
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

lock = threading.Lock()

from .prompt_templates import get_instruction_prompt


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded


def parse_markdown_and_associate_images(md_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Parse Markdown files, extract images, and associate them with text blocks.
    """
    logger.info(f"Starting to parse Markdown file: {md_path}")

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        image_pattern = r'!\[[^\]]*\]\(([^)]+)\)'
        image_paths_original = re.findall(image_pattern, content)
        
        if not image_paths_original:
            logger.warning(f"No Markdown format image links found in file {md_path}.")
            return []
        
        logger.info(f"Found {len(image_paths_original)} image links in the file.")

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
        
        logger.info(f"Successfully parsed and associated {len(processed_chunks)} text blocks containing images.")
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

        user_content.append({"type": "text", "text": f"This is the context text you need to process:\n\n---\n{context_text}\n---"})

        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": user_content}
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
    model_name: str,
    chunk_size=2000,
    chunk_overlap=300,
    question_number=2,
    max_workers=5,
    debug: bool = False,
    **kwargs,
):
    """
    The main function for generating multimodal question-answer pairs from a Markdown file containing images.
    """
    if debug:
        logger.debug(f"generatr_qa_pairs called with parameters:")
        logger.debug(f"  file_path: {file_path}")
        logger.debug(f"  api_key: {'***' if api_key else None}")
        logger.debug(f"  model_name: {model_name}")
        logger.debug(f"  chunk_size: {chunk_size}")
        logger.debug(f"  chunk_overlap: {chunk_overlap}")
        logger.debug(f"  question_number: {question_number}")
        logger.debug(f"  max_workers: {max_workers}")
        logger.debug(f"  kwargs: {kwargs}")
    
    chunks_with_images = parse_markdown_and_associate_images(
        file_path, chunk_size, chunk_overlap
    )

    if not chunks_with_images:
        logger.warning("Failed to parse any text blocks containing images from the file.")
        if debug:
            logger.debug("No chunks with images found, returning empty list")
        return []
    
    if debug:
        logger.debug(f"Found {len(chunks_with_images)} chunks with images")

    final_qa_list = []

    def _process_chunk(chunk_data):
        context_text = chunk_data["text"]
        images = chunk_data["images"]
        
        if debug:
            logger.debug(f"Processing chunk with text length: {len(context_text)}, images: {len(images)}")
        
        instruction_prompt = get_instruction_prompt(question_number)
        
        if debug:
            logger.debug(f"Generated instruction prompt: {instruction_prompt[:100]}...")
        
        generated_dialogs = generate_multimodal_qa_with_openai(
            api_key=api_key,
            model=model_name,
            instruction_prompt=instruction_prompt,
            context_text=context_text,
            image_paths=images,
        )

        chunk_qas = []
        if generated_dialogs and isinstance(generated_dialogs, list):
            if debug:
                logger.debug(f"Generated {len(generated_dialogs)} dialogs for chunk")
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
        elif debug:
            logger.debug(f"No valid dialogs generated for chunk")
        
        if debug:
            logger.debug(f"Chunk processing completed, generated {len(chunk_qas)} QA pairs")
        return chunk_qas
    logger.info(f"Starting to generate Q&A pairs for {len(chunks_with_images)} text blocks (threads: {max_workers})...")
    if debug:
        logger.debug(f"Using ThreadPoolExecutor with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks_with_images]

        # Disable tqdm progress bar in debug mode to avoid conflict with log output
        if debug:
            # Do not use progress bar in debug mode, process futures directly
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    with lock:
                        final_qa_list.extend(result)
                        logger.debug(f"Processed chunk {i}/{len(futures)}: Added {len(result)} QA pairs, total: {len(final_qa_list)}")
                else:
                    logger.debug(f"Processed chunk {i}/{len(futures)}: Future returned empty result")
        else:
            # Use progress bar in non-debug mode
            with tqdm(as_completed(futures), total=len(futures), desc="Generating multimodal QA") as pbar:
                for future in pbar:
                    result = future.result()
                    if result:
                        with lock:
                            final_qa_list.extend(result)
                        pbar.set_postfix({"Generated QA": len(final_qa_list)})
    
    logger.success(f"Processing completed! Generated a total of {len(final_qa_list)} multimodal Q&A pairs.")
    if debug:
        logger.debug(f"Returning {len(final_qa_list)} multimodal QA pairs")
    return final_qa_list
