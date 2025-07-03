from bespokelabs import curator
from datasets import Dataset  # huggingface datasets

def use_bespkelabs(model_name: str, prompt: str, **kwargs):
    """
    General LLM single-call interface.
    Supports Qwen (Tongyi), OpenAI, DeepSeek, and other mainstream models via model_name/base_url/api_key.
    :param model_name: Model name (e.g., "qwen-turbo", "gpt-4o-mini")
    :param prompt: Prompt string for the LLM
    :param api_key: API key for the target provider
    :param base_url: API base url for the provider
    :param kwargs: Additional curator.LLM arguments
    :return: LLM response as a pandas DataFrame
    """
    llm = curator.LLM(model_name=model_name, **kwargs)
    result = llm(prompt)
    return result.to_pandas()
def use_bespkelabs_autolabel(
    texts: list,
    model_name: str,
    api_key: str,
    base_url: str,
    label_type: str = "qa",   # "qa" (question-answer) or "summary"
    prompt_tpl: str = None,
    **kwargs
):
    """
    Batch auto-labeling interface for Q&A or summarization.
    :param texts: List of input texts
    :param model_name: Model name (e.g., "qwen-turbo")
    :param api_key: API key for the target provider
    :param base_url: API base url for the provider
    :param label_type: Label type, supports "qa" (Q&A) or "summary"
    :param prompt_tpl: Custom prompt template (optional)
    :param kwargs: Additional curator.LLM arguments
    :return: Labeled result as a pandas DataFrame
    """
    data = Dataset.from_dict({"text": texts})

    if not prompt_tpl:
        if label_type == "qa":
            prompt_tpl = "Please generate a useful question-answer pair for the following text:\n{text}"
        elif label_type == "summary":
            prompt_tpl = "Please generate a concise summary for the following text:\n{text}"
        else:
            raise ValueError("Unknown label_type")

    class AutoLabeler(curator.LLM):
        def prompt(self, input):
            return prompt_tpl.format(**input)
        def parse(self, input, response):
            if label_type == "qa":
                # Assume response in format: "Question: ...\nAnswer: ..."
                try:
                    q, a = response.split('\n', 1)
                    return {
                        "question": q.replace('Question:', '').replace('问题：', '').strip(),
                        "answer": a.replace('Answer:', '').replace('答案：', '').strip(),
                        "text": input["text"]
                    }
                except Exception:
                    return {"question": "", "answer": "", "text": input["text"]}
            elif label_type == "summary":
                return {"summary": response, "text": input["text"]}
            else:
                return {"output": response, "text": input["text"]}

    labeler = AutoLabeler(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)
    res = labeler(data)
    return res.to_pandas()
