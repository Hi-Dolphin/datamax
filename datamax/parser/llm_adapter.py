import os
try:
    import dashscope
except ImportError:
    dashscope = None
try:
    from bespokelabs import curator
except ImportError:
    curator = None
def smart_llm_call(model_name, prompt, api_key=None, base_url=None, **kwargs):
    """
    auto select dashscope/curator/openai，platform universal calling，support dashscope/qwen/gpt/deepseek。
    """
    # Automatically determine whether or not Qwen model
    if (model_name.startswith("qwen") or (base_url and "dashscope" in base_url)):
        if dashscope is None:
            raise ImportError("请先 pip install dashscope")
        dashscope.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        response = dashscope.Generation.call(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["output"]["text"]
    elif curator is not None:
        # fallback to bespokelabs-curator (Can support gpt/openai/deepseek,etc)
        backend_params = {}
        if api_key: backend_params["api_key"] = api_key
        if base_url: backend_params["base_url"] = base_url
        llm = curator.LLM(
            model_name=model_name,
            backend_params=backend_params,
            **kwargs
        )
        return llm(prompt).to_pandas()
    else:
        raise RuntimeError("未检测到可用的 LLM 包，请安装 dashscope 或 bespokelabs-curator")

# 用例
if __name__ == "__main__":
    # Qwen dashscope
    print(smart_llm_call(
        model_name="qwen-turbo",
        prompt="写一首关于自动化标注的诗",
        api_key="sk-你的key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ))

    # openai gpt-3.5（用 curator）
    print(smart_llm_call(
        model_name="gpt-3.5-turbo",
        prompt="Write a poem about AI data labeling.",
        api_key="sk-your-openai-key",
        base_url="https://api.openai.com/v1"
    ))
