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

try:
    from ppstructure.recovery.recovery_to_doc import convert_info_docx, sorted_layout_boxes
except ImportError:
    import sysconfig
    lib_path = sysconfig.get_path('purelib')
    sys.path.insert(0, os.path.join(lib_path, "paddleocr"))
    try:
        from ppstructure.recovery.recovery_to_doc import convert_info_docx, sorted_layout_boxes
    except ImportError as e:
        raise ImportError("Failed to import paddleocr modules. Ensure paddleocr and ppstructure are installed.") from e


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
        for pg in range(pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)

    if not imgs:
        raise ValueError("No images extracted from PDF")

    img_name = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # step2: Process images
    img_paths = []
    img_dir = pathlib.Path(output) / img_name
    img_dir.mkdir(exist_ok=True)
    for index, pdf_img in enumerate(imgs):
        pdf_img_path = img_dir / f"{img_name}_{index:03d}.jpg"
        cv2.imwrite(str(pdf_img_path), pdf_img)
        img_paths.append([str(pdf_img_path), pdf_img])

    # step3: Convert images to DOCX
    all_res = []
    engine = PPStructure(
        recovery=True,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        det_model_dir=str(ROOT_DIR / "ocr_model_dir/det/en/en_PP-OCRv3_det_infer"),
        rec_model_dir=str(ROOT_DIR / "ocr_model_dir/rec/ch/ch_PP-OCRv4_rec_infer"),
        table_model_dir=str(ROOT_DIR / "ocr_model_dir/table/en_ppstructure_mobile_v2.0_SLANet_infer"),
        layout_model_dir=str(ROOT_DIR / "ocr_model_dir/layout/picodet_lcnet_x1_0_fgd_layout_infer"),
        formula_model_dir=str(ROOT_DIR / "ocr_model_dir/formula/rec_latex_ocr_infer"),
    )
    for index, (img_path, img) in enumerate(img_paths):
        print(f"processing {index + 1}/{len(img_paths)} page:")
        result = engine(img, img_idx=index)
        save_structure_res(result, output, img_name, index)
        h, w, _ = img.shape
        result_sorted = sorted_layout_boxes(deepcopy(result), w)
        all_res.extend(result_sorted)

    try:
        convert_info_docx(imgs, all_res, output, img_name)
        output_file = pathlib.Path(output) / f"{img_name}_ocr.docx"
        new_name = pathlib.Path(output) / f"{pathlib.Path(img_path).stem}_ocr.docx"
        output_file.rename(new_name)
    except Exception as e:
        raise


def use_paddleocr(
    input_files: str, output_files: str, use_gpu: bool = False, gpu_id: int = 6
):
    output_path = pathlib.Path(output_files)
    output_path.mkdir(exist_ok=True)
    recovery(
        img_path=input_files,
        output=str(output_path),
        use_gpu=use_gpu,
        gpu_id=gpu_id,
    )
