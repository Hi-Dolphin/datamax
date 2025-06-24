import argparse
import ast
import os
import re
from datetime import datetime

import cv2
import numpy as np
import torch
from GOT.demo.process_results import punctuation_dict
from GOT.model import *
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import SeparatorStyle, conv_templates
from GOT.utils.utils import KeywordsStoppingCriteria, disable_torch_init
from paddle.utils import try_import
from PIL import Image
from transformers import AutoTokenizer

fitz = try_import("fitz")

DEFAULT_IMAGE_TOKEN = "<image>"  # nosec B105 - technical const,not a passward
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"  # nosec B105 - technical const,not a passward

DEFAULT_IM_START_TOKEN = "<img>"  # nosec B105 - technical const,not a passward
DEFAULT_IM_END_TOKEN = "</img>"  # nosec B105 - technical const,not a passward

translation_table = str.maketrans(punctuation_dict)

parser = argparse.ArgumentParser()

args = argparse.Namespace()
args.model_name = "./GOT_weights/"
args.type = "format"
args.box = ""
args.color = ""

# TODO vary old codes, NEED del
image_processor = BlipImageEvalProcessor(image_size=1024)

image_processor_high = BlipImageEvalProcessor(image_size=1024)

use_im_start_end = True

image_token_len = 256


def covert_pdf_to_image(image_path: str):
    # step1: Convert PDF to images
    imgs = []
    with fitz.open(image_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(4, 4)  # Magnify by four times throughout the process
            pm = page.get_pixmap(matrix=mat, alpha=False)
            # if pm.width > 2000 or pm.height > 2000:
            #     pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)

    img_name = datetime.now().strftime("%Y%m%d%H%M%S")
    # step2: Process images
    output = "output"
    img_paths = []
    for index, pdf_img in enumerate(imgs):
        # img processing

        gray_img = cv2.cvtColor(pdf_img, cv2.COLOR_BGR2GRAY)

        # Binarization processing
        _, binary_img = cv2.threshold(
            gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # denoise
        filtered_img = cv2.medianBlur(binary_img, 3)
        processed_img = filtered_img

        os.makedirs(os.path.join(output, img_name), exist_ok=True)
        pdf_img_path = os.path.join(
            output, img_name, img_name + "_" + str(index) + ".jpg"
        )
        cv2.imwrite(pdf_img_path, processed_img)
        img_paths.append([pdf_img_path, processed_img])

    return img_name


# def initialize_model(model_path: str = "./GOT_weights/", gpu_id: int = 6):
#     model_name = os.path.expanduser(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map=f'cuda:{gpu_id}',
#                                                use_safetensors=True, pad_token_id=151643).eval()
#     model.to(device=f'cuda:{gpu_id}', dtype=torch.bfloat16)
#     return model, tokenizer


def initialize_model(model_path: str = "./GOT_weights/", gpu_id: int = 6):
    model_name = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # load model
    model = GOTQwenForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map=f"cuda:{gpu_id}",
        use_safetensors=True,
        pad_token_id=151643,
    ).eval()

    # Ensure that both the model and the tensor are moved to the target device.
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device=device, dtype=torch.bfloat16)

    # Ensure that the output of the tokenizer is also on the target device.
    tokenizer.model_max_length = 512  # maxlength，adjust to need
    tokenizer.padding_side = "right"  # padding side，adjust to need

    return model, tokenizer


def eval_model(file: str, model, tokenizer, gpu_id: int = 6):
    # Model
    # image = load_image(args.image_file)
    image = Image.open(file).convert("RGB")

    w, h = image.size
    # print(image.size)

    disable_torch_init()

    if args.type == "format":
        qs = "OCR with format: "
    else:
        qs = "OCR: "

    if args.box:
        bbox = ast.literal_eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0] / w * 1000)
            bbox[1] = int(bbox[1] / h * 1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0] / w * 1000)
            bbox[1] = int(bbox[1] / h * 1000)
            bbox[2] = int(bbox[2] / w * 1000)
            bbox[3] = int(bbox[3] / h * 1000)
        if args.type == "format":
            qs = str(bbox) + " " + "OCR with format: "
        else:
            qs = str(bbox) + " " + "OCR: "

    if args.color:
        if args.type == "format":
            qs = "[" + args.color + "]" + " " + "OCR with format: "
        else:
            qs = "[" + args.color + "]" + " " + "OCR: "

    if use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])

    # vary old codes, no use
    image_1 = image.copy()
    image_tensor = image_processor(image)

    image_tensor_1 = image_processor_high(image_1)

    input_ids = torch.as_tensor(inputs.input_ids).to(f"cuda:{gpu_id}")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast(f"cuda:{gpu_id}", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[
                (
                    image_tensor.unsqueeze(0).half().to(f"cuda:{gpu_id}"),
                    image_tensor_1.unsqueeze(0).half().to(f"cuda:{gpu_id}"),
                )
            ],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            # streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria],
        )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
    return outputs + "\n"


def sorted_list_by_index(img_name):
    files_with_index = []
    for root, dirs, files in os.walk(f"./output/{img_name}"):
        for file in files:
            file_path = os.path.join(root, file)
            match = re.search(r"_(\d+)(?:\.\w+)?$", file)
            if match:
                index = int(match.group(1))
                files_with_index.append((file_path, index))
    files_with_index.sort(key=lambda x: x[1])
    sorted_files = [file[0] for file in files_with_index]
    return sorted_files


def convert_to_markdown(md_content, pdf_path):
    """write into markdown"""
    file_extension = os.path.splitext(pdf_path)[1].lower()
    output_file = (
        f'./got_output/{os.path.basename(pdf_path).replace(file_extension, ".mmd")}'
    )
    os.makedirs("got_output", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)


def main(image_list: str, pdf_path: str, model, tokenizer, gpu_id: int = 6):
    res_list = sorted_list_by_index(image_list)

    outputs = ""
    for file_path in res_list:
        outputs += eval_model(
            file=file_path, model=model, tokenizer=tokenizer, gpu_id=gpu_id
        )

    convert_to_markdown(outputs, pdf_path)
    return outputs


def generate_mathpix_markdown(pdf_path: str, model, tokenizer, gpu_id: int = 6):
    image_list = covert_pdf_to_image(pdf_path)
    outputs = main(
        image_list=image_list,
        pdf_path=pdf_path,
        gpu_id=gpu_id,
        model=model,
        tokenizer=tokenizer,
    )
    return outputs
