import tiktoken


class DashScopeClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DashScopeClient, cls).__new__(cls)
        return cls._instance

    def get_tokenizer(self, content):
        """
        Note: tiktoken only supports the following models with different token calculations
        A BPE word divider developed by tiktoken openai
        o200k_base corresponds to models: gpt-4o, GPT-4O-MINI
        cl100k_base models: GPT-4-Turbo, gpt-4, gpt-3.5-turbo...
        p50k_base corresponds to models text-davinci-002 and text-davinci-003
        r50k_base corresponds to model gpt2
        """
        encoding = tiktoken.get_encoding(encoding_name="cl100k_base")
        num_tokens = len(encoding.encode(content))
        return num_tokens
