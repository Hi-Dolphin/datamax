from datamax.parser.core import DataMax

def test_qa_generator_with_bespokelabs():
    dm = DataMax()
    results = dm.qa_generator_with_bespokelabs(
        texts=[
            "人工智能是近年来快速发展的科技领域，具有广泛的应用前景。",
            "深度学习算法能够自动从大量数据中学习有用的特征。"
        ],
        model_name="qwen-turbo",
        # model_name="gpt-3.5-turbo"
        api_key="sk-bfddf3a562fb40b38bec21199dec6d82",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # base_url="https://api.openai.com/v1"
        label_type="qa",
    )
    for r in results:
        print("Dashscope QA:", r)
    # 断言，确保结果不为空且有问题和答案
    assert len(results) == 2
    assert all("question" in item and "answer" in item for item in results)

if __name__ == "__main__":
    test_qa_generator_with_bespokelabs()