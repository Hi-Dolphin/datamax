'''
Example of using DataMax for parsing: Currently supported file types include: doc, docx, epub, ppt, pptx, html, pdf, txt.
'''

# 1. Import the SDK parsing class
from datamax import DataMaxParser

# 2. Use the corresponding parsing class to parse the file
data = DataMaxParser(file_path=r"C:\Users\cykro\Desktop\纯文本船视宝.txt")
print(f'txt parser result:-->{data.get_data()}')

data = DataMaxParser(file_path=r"C:\Users\cykro\Desktop\数据工厂.pdf", use_ocr=True)
print(f'pdf parser result:-->{data.get_data()}')

data = DataMaxParser(file_path=r"C:\Users\cykro\Desktop\航海学知识点.docx", to_markdown=True)
print(f'pdf parser result:-->{data.get_data()}')


# Data Cleaning
"""
For specific cleaning rules, please refer to datamax/utils/data_cleaner.py
abnormal: abnormal data cleaning
private: privacy processing
filter: text filtering
"""
# Direct use: Supports direct cleaning of text content in the text parameter and returns a string
dm = DataMaxParser()
data = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
print(data)
# Process use: Supports using after get_data(), which returns the complete data structure
dm = DataMaxParser(file_path=r"C:\Users\cykro\Desktop\数据库开发手册.pdf", use_ocr=True)
data2 = dm.get_data()
print(data2)
cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
print(cleaned_data)