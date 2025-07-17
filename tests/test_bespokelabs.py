import unittest
from datamax import DataMax

API_KEY = "sk-xxx"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "your-model-name"

test_prompt = "请简述人工智能的基本原理。"
test_content = "人工智能是计算机科学的一个分支，研究如何使计算机具备类似人的智能。"

class TestDataMax(unittest.TestCase):

    def test_call_llm(self):
        """Test basic LLM call."""
        text, status = DataMax.call_llm_with_bespokelabs(
            prompt=test_prompt,
            model_name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )

        print("LLM response:", text)
        self.assertEqual(status, "success")
        self.assertIn("人工智能", text)

    def test_qa_generation(self):
        """Test automatic QA generation."""
        qas = DataMax.qa_generator_with_bespokelabs(
            content=test_content,
            model_name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        print("QA pairs:", qas)
        self.assertIsInstance(qas, list)
        self.assertGreater(len(qas), 0)
        self.assertIn("question", qas[0])
        self.assertIn("answer", qas[0])

if __name__ == "__main__":
    unittest.main()
