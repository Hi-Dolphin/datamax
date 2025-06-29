"""layout_recovery"""

import os
import pathlib
import sys
from copy import deepcopy
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from paddle.utils import try_import
from paddleocr import PPStructure, save_structure_res

sys.path.append("/usr/local/lib/python3.10/dist-packages/paddleocr")
from ppstructure.recovery.recovery_to_doc import convert_info_docx, sorted_layout_boxes


def recovery(img_path, output, use_gpu, gpu_id):
    """
    Convert a PDF file to a Word document with layout recovery.

    :param img_path: Path to the PDF file
    :param output: Path to the output folder
    """
    fitz = try_import("fitz")

    # step1: Convert PDF to images
    imgs = []
    with fitz.open(img_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)

    img_name = datetime.now().strftime("%Y%m%d%H%M%S")

    # step2: Process images
    img_paths = []
    for index, pdf_img in enumerate(imgs):
        os.makedirs(os.path.join(output, img_name), exist_ok=True)
        pdf_img_path = os.path.join(
            output, img_name, img_name + "_" + str(index) + ".jpg"
        )
        cv2.imwrite(pdf_img_path, pdf_img)
        img_paths.append([pdf_img_path, pdf_img])

    # step3: Convert images to DOCX
    all_res = []
    engine = PPStructure(
        recovery=True,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        det_model_dir=f"{ROOT_DIR}/ocr_model_dir/det/en/en_PP-OCRv3_det_infer",
        rec_model_dir=f"{ROOT_DIR}/ocr_model_dir/rec/ch/ch_PP-OCRv4_rec_infer",
        table_model_dir=f"{ROOT_DIR}/ocr_model_dir/table/en_ppstructure_mobile_v2.0_SLANet_infer",
        layout_model_dir=f"{ROOT_DIR}/ocr_model_dir/layout/picodet_lcnet_x1_0_fgd_layout_infer",
        formula_model_dir=f"{ROOT_DIR}/ocr_model_dir/formula/rec_latex_ocr_infer",
    )
    for index, (new_img_path, imgs) in enumerate(img_paths):
        print("processing {}/{} page:".format(index + 1, len(img_paths)))
        result = engine(imgs, img_idx=index)
        save_structure_res(result, output, img_name, index)
        h, w, _ = imgs.shape
        result_cp = deepcopy(result)
        result_sorted = sorted_layout_boxes(result_cp, w)
        all_res += result_sorted
    try:
        convert_info_docx(imgs, all_res, output, img_name)
        os.rename(
            f"./output/{img_name}_ocr.docx",
            f'./output/{os.path.basename(img_path).replace(".pdf", "")}_ocr.docx',
        )
    except Exception as e:
        raise e


def use_paddleocr(
    input_files: str, output_files: str, use_gpu: bool = False, gpu_id: int = 6
):
    try:
        if not os.path.exists(output_files):
            os.makedirs(output_files)
        try:
            recovery(
                img_path=input_files,
                output=output_files,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
            )
        except Exception as e:
            raise e
    except Exception as e:
        raise e
