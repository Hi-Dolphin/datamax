from datamax import DataMax
from datamax.generator import domain_tree

# prepare info
FILE_PATHS = ["/mnt/f/凡人修仙传前三章.txt"]
LABEL_LLM_API_KEY = "sk-48fc797bfa7340018a67ade72769e694"
LABEL_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LABEL_LLM_MODEL_NAME = "qwen-turbo-latest"
LLM_TRAIN_OUTPUT_FILE_NAME = "train"

# init client
client = DataMax(file_path=FILE_PATHS)

# get pre label. return trainable qa list
qa_list = client.get_pre_label(
    api_key=LABEL_LLM_API_KEY,
    base_url=LABEL_LLM_BASE_URL,
    model_name=LABEL_LLM_MODEL_NAME,
    question_number=1,
    max_workers=1,
    debug=True)

    # save label data
client.save_label_data(qa_list, LLM_TRAIN_OUTPUT_FILE_NAME)


