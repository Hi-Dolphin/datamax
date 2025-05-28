# '''
# Example of using DataMax for parsing: Currently supported file types include: doc, docx, epub, ppt, pptx, html, pdf, txt, xls, xlsx, jpg(png|jpeg|...)
# '''
#
# # 1. Import the SDK parsing class
# from datamax import DataMax
# import time
#

# 2. Use the corresponding parsing class to parse the file
from datamax import DataMax

data = DataMax(file_path=r"/mnt/f/code/datamax/examples/网信安全风险评估报告（2024） (1).doc")

print(f"txt parser result:-->{data.get_data()}")

# # pdf type (use ocr by mineru)
# data = DataMax(file_path=r"db.pdf", use_mineru=True)
# print(f'pdf parser result:-->{data.get_data()}')


# # "XLSX files may encounter long-running calculations (e.g., complex formulas or large datasets). You can set a TTL (Time-To-Live) to enforce a timeout."
# data = DataMax(file_path=r"db.xlsx", timeout=10)
# print(f'xlsx parser result:-->{data.get_data()}')


# #
# # Data Cleaning
# """
# For specific cleaning rules, please refer to datamax/utils/data_cleaner.py
# abnormal: abnormal data cleaning
# private: privacy processing
# filter: text filtering
# """
# # Direct use: Supports direct cleaning of text content in the text parameter and returns a string
# dm = DataMax()
# data1 = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
# print(data1)
# # Process use: Supports using after get_data(), which returns the complete data structure
# dm = DataMax(file_path=r"C:\Users\cykro\Desktop\数据库开发手册.pdf", use_mineru=True)
# data2 = dm.get_data()
# print(data2)
# cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
# print(cleaned_data)


# # Data prelabeling (abandon)
# '''
# data = DataMax(
#         file_path=r"path\to\xxxxx.docx",
#         to_markdown=True,
#     )
# parsed_data = data.get_data()

# # Optionally enhance the parsed data with a large model
# # 如果不传递自定义messages，则使用sdk中默认的messages
# messages=[
#         {'role': 'system', 'content': 'You are a helpful assistant.'},
#         {'role': 'user', 'content': '你是谁？'}
#     ]
# enhanced_data = data.enhance_with_model(api_key="sk-key", base_url="https://xxxx", model_name="xxxx", messages=messages)
# print(f'Annotated result:{enhanced_data}')
# '''

# # Data prelabeling (new)
# dm = DataMax(file_path="./知识图谱概要设计.md")
# data = dm.get_pre_label(
#     api_key="sk-xxxx",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
#     model_name="qwen-max",
#     chunk_size=500,
#     chunk_overlap=100,
#     question_number=5,
#     max_workers=5,
#     # message=[]
# )
# print(data)
