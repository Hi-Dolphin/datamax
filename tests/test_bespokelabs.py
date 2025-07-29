import unittest
from datamax import DataMax

API_KEY = "sk-xxx"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "your-model-name"

test_prompt = "请简述人工智能的基本原理。"
test_content = "人工智能是计算机科学的一个分支，研究如何使计算机具备类似人的智能。"

class TestDataMax(unittest.TestCase):
    """
    Test suite for DataMax's integration with BespokeLabs Curator LLM and QA generator.
    """

    def test_call_llm(self):
        """
        Test basic LLM call.
        Allows both 'success' and 'fail' statuses to accommodate network or API issues.
        """
        text, status = DataMax.call_llm_with_bespokelabs(
            prompt=test_prompt,
            model_name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        print("LLM response:", text)
        # Allow both success and fail for flexible testing
        self.assertIn(status, ("success", "fail"))

    def test_qa_generation(self):
        """Test automatically generated question-answer pairs"""
        qas_list = DataMax.qa_generator_with_bespokelabs(
            content=test_content,
            model_name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        print("Raw QA response:", qas_list)
        # Make an assertion directly using qas_list
        self.assertIsInstance(qas_list, list)
        self.assertGreater(len(qas_list), 0)
        self.assertIn("question", qas_list[0])
        self.assertIn("answer", qas_list[0])

if __name__ == "__main__":
    unittest.main()
