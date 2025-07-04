import os

try:
    from bespokelabs import curator
except ImportError:
    curator = None
try:
    import dashscope  # Official SDK
except ImportError:
    dashscope = None

from datasets import Dataset  # huggingface datasets

def call_llm_with_bespokelabs(
    model_name: str,
    prompt: str,
    api_key: str = None,
    base_url: str = None,
    **kwargs
):
    """
    Universal LLM single inference call (OpenAI compatible protocol), supporting mainstream cloud vendors such as Qwen, OpenAI, DeepSeek, etc.
    """
    backend_params = {}
    if api_key: backend_params["api_key"] = api_key
    if base_url: backend_params["base_url"] = base_url

    llm = curator.LLM(
        model_name=model_name,
        backend_params=backend_params,
        **kwargs
    )
    result = llm(prompt)
    return result.to_pandas()

def qa_generator_with_bespokelabs(
    texts: list,
    model_name: str,
    api_key: str = None,
    base_url: str = None,
    label_type: str = "qa",   # "qa" or "summary"
    prompt_tpl: str = None,
    **kwargs
):
    """
    Batch automatic Q&A pairs or summary annotation.
    Support Qwen（dashscope）、OpenAI、DeepSeek compatible API。
    """
    # 1. compatible dashscope SDK
    if model_name.startswith("qwen"):
        if dashscope is None:
            raise ImportError("dashscope SDK is not installed. pip install dashscope")
        if not api_key:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        dashscope.api_key = api_key

        results = []
        if not prompt_tpl:
            if label_type == "qa":
                prompt_tpl = "请根据下文生成有用的问答对：\n{text}"
            elif label_type == "summary":
                prompt_tpl = "请为下文生成简明摘要：\n{text}"
            else:
                raise ValueError("Unknown label_type")
        for t in texts:
            p = prompt_tpl.format(text=t)
            response = dashscope.Generation.call(
                model=model_name,
                messages=[{"role": "user", "content": p}]
            )
            output = response["output"]["text"]
            if label_type == "qa":
                try:
                    q, a = output.split('\n', 1)
                    q = q.replace('Question:', '').replace('问题：', '').strip()
                    a = a.replace('Answer:', '').replace('答案：', '').strip()
                    results.append({"question": q, "answer": a, "text": t})
                except Exception:
                    results.append({"question": "", "answer": "", "text": t})
            elif label_type == "summary":
                results.append({"summary": output, "text": t})
            else:
                results.append({"output": output, "text": t})
        import pandas as pd
        return pd.DataFrame(results)
    # 2. compatible bespokelabs-curator（general OpenAI protocol backend）
    else:
        if curator is None:
            raise ImportError("bespokelabs SDK is not installed. pip install bespokelabs-curator")
        data = Dataset.from_dict({"text": texts})
        if not prompt_tpl:
            if label_type == "qa":
                prompt_tpl = "Please generate a useful question-answer pair for the following text:\n{text}"
            elif label_type == "summary":
                prompt_tpl = "Please generate a concise summary for the following text:\n{text}"
            else:
                raise ValueError("Unknown label_type")
        backend_params = {}
        if api_key: backend_params["api_key"] = api_key
        if base_url: backend_params["base_url"] = base_url

        class AutoLabeler(curator.LLM):
            def prompt(self, input):
                return prompt_tpl.format(**input)
            def parse(self, input, response):
                if label_type == "qa":
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

        labeler = AutoLabeler(
            model_name=model_name,
            backend_params=backend_params,
            **kwargs
        )
        res = labeler(data)
        return res.to_pandas()
