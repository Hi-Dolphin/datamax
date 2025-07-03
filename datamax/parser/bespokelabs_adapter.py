from bespokelabs import curator

def use_bespkelabs(model_name: str, prompt: str, **kwargs) -> str:
    """
    Call bespokelabs-curator LLM and return the result as a string/DataFrame.
    :param model_name: Model name, e.g. "gpt-4o-mini"
    :param prompt: Prompt string for the model
    :param kwargs: Any extra arguments for curator.LLM
    :return: LLM result (as pandas DataFrame string by default)
    """
    llm = curator.LLM(model_name=model_name, **kwargs)
    result = llm(prompt)
    return result.to_pandas()
