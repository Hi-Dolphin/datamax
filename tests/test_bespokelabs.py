import unittest
from datamax import DataMax

API_KEY = "sk-xxx"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "your-model-name"

test_prompt = "请解释人工智能的基本原理。"
test_content = """
人工智能是计算机科学的一个分支，致力于研究、开发模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
"""

class TestDataMax(unittest.TestCase):

    def test_call_llm(self):
        result, status = DataMax.call_llm_with_bespokelabs(
            prompt=test_prompt,
            model_name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        print("LLM调用结果文本:", result)
        self.assertEqual(status, "success")
        self.assertIn("人工智能", result)

    def test_qa_generation(self):
        results = DataMax.qa_generator_with_bespokelabs(
            content=test_content,
            model_name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        print("QA生成结果:", results)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # Optionally check content
        self.assertIn("人工智能", results[0])

if __name__ == "__main__":
    unittest.main()
