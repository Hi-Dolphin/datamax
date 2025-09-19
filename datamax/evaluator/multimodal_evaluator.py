import re
import torch
import time
import math
import json
import dashscope
import base64
import mimetypes
from http import HTTPStatus
from openai import OpenAI
from loguru import logger
from pathlib import Path

logger.add(lambda msg: print(msg, end=""), format="{message}")

class MultimodalConsistencyEvaluator:
    """
    Evaluates the consistency between images and text using DashScope services.
    """
    def __init__(self, clip_model_name: str, vqa_model_name: str, dashscope_api_key: str, dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", device: str = None):
        """
        Initializes the multimodal consistency evaluator.

        Args:
            clip_model_name (str): The name of the DashScope multimodal embedding model.
            vqa_model_name (str): The name of the VQA model to use.
            dashscope_api_key (str): The API key for DashScope.
            dashscope_base_url (str, optional): The base URL for the DashScope OpenAI-compatible API.
            device (str, optional): The device to run PyTorch operations on ('cuda' or 'cpu'). Auto-detects if None.
        """
        if not torch:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch'.")
        
        dashscope.api_key = dashscope_api_key

        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_name = clip_model_name
        self.vqa_model_name = vqa_model_name
        self.dashscope_base_url = dashscope_base_url
        self.vqa_client = None
        logger.info(f"Evaluator initialized to run on device: '{self.device}'")

    def _load_vqa_client(self):
        """Initializes the OpenAI-compatible client for VQA tasks if not already present."""
        if self.vqa_client is None:
            self.vqa_client = OpenAI(
                api_key=dashscope.api_key,
                base_url=self.dashscope_base_url,
            )

    def _get_embeddings_with_sdk(self, input_data: list) -> list[list[float]] | None:
        """
        Helper function to get a list of embeddings using the DashScope SDK.

        Args:
            input_data (list): A list containing image paths and text content.

        Returns:
            list[list[float]] | None: A list of embedding vectors or None on failure.
        """
        resp = dashscope.MultiModalEmbedding.call(
            model=self.clip_model_name,
            input=input_data
        )

        if resp.status_code == HTTPStatus.OK and resp.output and resp.output.get('embeddings'):
            return [item['embedding'] for item in resp.output['embeddings']]
        else:
            logger.error(f"DashScope API SDK error: Status {resp.status_code}, Code {resp.code}, Message {resp.message}")
            return None

    def evaluate_clip_score(self, image_path: str, text: str, retries: int = 3, delay: int = 5) -> dict:
        """
        Calculates CLIPscore by computing the cosine similarity between image and text embeddings.

        Args:
            image_path (str): The local path to the image file.
            text (str): The text to compare with the image.
            retries (int, optional): Number of retry attempts in case of API failure. Defaults to 3.
            delay (int, optional): Delay in seconds between retries. Defaults to 5.

        Returns:
            dict: A dictionary containing the cosine similarity or an error message.
        """
        for attempt in range(retries):
            try:
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

                combined_input = [{'image': image_path}, {'text': text}]
                embeddings = self._get_embeddings_with_sdk(combined_input)

                if embeddings and len(embeddings) == 2:
                    image_embedding = embeddings[0]
                    text_embedding = embeddings[1]

                    img_tensor = torch.tensor(image_embedding, device=self.device)
                    txt_tensor = torch.tensor(text_embedding, device=self.device)
                    
                    cos_sim = torch.nn.functional.cosine_similarity(img_tensor, txt_tensor, dim=0).item()
                    
                    return {"cosine_similarity": cos_sim}
                else:
                    raise ValueError("Failed to retrieve a valid pair of embeddings from the API.")

            except Exception as e:
                logger.error(f"CLIP score evaluation failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return {"error": str(e), "cosine_similarity": 0.0}

        return {"error": "Failed to get embeddings after all retries", "cosine_similarity": 0.0}

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encodes a local image file to a Base64 data URL.

        Args:
            image_path (str): The path to the local image file.

        Returns:
            str: The data URL string.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"

    def evaluate_vqa_score(self, image_path: str, text_prompts: list[str]) -> list[float]:
        """
        Calculates VQAScores for an image against multiple text prompts.

        This function instructs a model via a system prompt to return a JSON 
        containing a calculated probability. It includes robust post-processing
        to handle cases where the model returns log-probabilities (negative values)
        instead of standard probabilities.

        Args:
            image_path (str): The local path or URL to the image.
            text_prompts (list[str]): A list of text descriptions to evaluate.

        Returns:
            list[float]: A list of VQAScore values, normalized to the [0.0, 1.0] range.
        """
        self._load_vqa_client()
        
        SYSTEM_PROMPT = """
        You are a specialized VQA assistant. Your sole function is to implement VQAScore, an image-text alignment metric, as defined in the provided research paper.
        Answer only if you are > t confident, since mistakes are penalized t/(1−t) points, while
        correct answers receive 1 point, and an answer of “I don’t know” receives 0 points. (t = 0.9)
        Your Task:
        Given an image and a text prompt (t), you must perform the following steps:
        Format the Question: Convert the input text t into the specific question format: "Does this figure show '{t}'? Please answer yes or no.".
        Calculate Probability: Compute the generative likelihood that the correct answer to the formatted question is the single word "Yes" or "yes" or "YES". This score is the VQAScore, defined as P({"Yes", "yes", "YES"}| image, question).
        Output Requirements:
        Your entire response must be a single JSON object and nothing else.
        The JSON object must contain a single key: "vqa_score".
        The value must be a float number between 0.0 and 1.0.
        Do not include any explanations, comments, or additional text.
        """

        vqa_scores = []
        image_url = image_path if image_path.startswith("http") else self.encode_image_to_base64(image_path)
        
        for text_prompt in text_prompts:
            user_question = f'Does this figure show "{text_prompt}"? Please answer yes or no.'
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_question},
                    ],
                },
            ]

            probability = 0.0
            response_text = ""
            try:
                completion = self.vqa_client.chat.completions.create(
                    model=self.vqa_model_name,
                    messages=messages
                )
                response_text = completion.choices[0].message.content

                json_match = re.search(r'```json\s*({.*?})\s*```|({.*?})', response_text, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in response", response_text, 0)

                json_str_to_parse = json_match.group(1) if json_match.group(1) else json_match.group(2)
                response_json = json.loads(json_str_to_parse)
                prob_value = response_json.get("vqa_score")

                if isinstance(prob_value, (int, float)):
                    if prob_value <= 0:
                        probability = math.exp(prob_value)
                    elif prob_value > 1:
                        probability = 1.0
                    else:
                        probability = float(prob_value)
                else:
                    logger.warning(f"Warning: 'vqa_score' was not a valid number for prompt '{text_prompt}'. Response: {response_text}")

            except json.JSONDecodeError:
                logger.error(f"Error: Could not decode JSON for prompt '{text_prompt}'. Response: {response_text}")
            except Exception as e:
                logger.error(f"An API call failed for prompt '{text_prompt}': {e}")
            
            vqa_scores.append(probability)

        return vqa_scores