import base64
import json
import math
import mimetypes
import re
import time
from pathlib import Path

from loguru import logger
from openai import OpenAI

logger.add(lambda msg: print(msg, end=""), format="{message}")


class MultimodalConsistencyEvaluator:
    """
    Evaluates the consistency between images and text using OpenAI services.
    """

    def __init__(
        self,
        clip_model_name: str,
        vqa_model_name: str,
        api_key: str,
        base_url: str | None = None,
        device: str | None = None,
    ):
        """
        Initializes the multimodal consistency evaluator.

        Args:
            clip_model_name (str): The OpenAI model used to judge image/text alignment.
            vqa_model_name (str): The OpenAI vision model used for VQA scoring.
            api_key (str): The API key for the OpenAI-compatible endpoint.
            base_url (str, optional): Custom base URL for OpenAI-compatible deployments.
            device (str, optional): Reserved for backward compatibility; no longer used for computation.
        """
        self.device = device or "cpu"
        self.clip_model_name = clip_model_name
        self.vqa_model_name = vqa_model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client: OpenAI | None = None
        logger.info("Evaluator initialized; OpenAI client will be created lazily.")

    def _ensure_client(self):
        """Initializes the OpenAI client if it has not been created yet."""
        if self.client is None:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)

    def _extract_message_text(self, message_content) -> str:
        """Normalizes message content into a plain string."""
        if isinstance(message_content, list):
            return "".join(
                part.get("text", "")
                for part in message_content
                if isinstance(part, dict)
            ).strip()
        return (message_content or "").strip()

    def _request_alignment_score(self, image_path: str, text: str) -> float:
        """Requests an OpenAI model to estimate image-text similarity as a cosine score."""
        self._ensure_client()
        image_reference = (
            image_path
            if image_path.startswith(("http://", "https://"))
            else self.encode_image_to_base64(image_path)
        )

        system_prompt = (
            "You evaluate how well an image matches a caption. Respond only with a JSON "
            "object containing a single key 'cosine_similarity' whose value is a number "
            "between 0.0 and 1.0."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_reference}},
                    {
                        "type": "text",
                        "text": (
                            "Return the CLIP-style cosine similarity between the provided "
                            f"image and caption: {text}"
                        ),
                    },
                ],
            },
        ]

        completion = self.client.chat.completions.create(
            model=self.clip_model_name,
            messages=messages,
        )

        response_text = self._extract_message_text(
            completion.choices[0].message.content
        )

        json_match = re.search(
            r"```json\s*({.*?})\s*```|({.*?})", response_text, re.DOTALL
        )
        if not json_match:
            raise json.JSONDecodeError(
                "No JSON object found in response", response_text, 0
            )

        json_str_to_parse = json_match.group(1) or json_match.group(2)
        response_json = json.loads(json_str_to_parse)
        similarity = response_json.get("cosine_similarity")

        if not isinstance(similarity, (int, float)):
            raise ValueError(
                "'cosine_similarity' was not a numeric value in the OpenAI response."
            )

        return max(0.0, min(1.0, float(similarity)))

    def evaluate_clip_score(
        self, image_path: str, text: str, retries: int = 3, delay: int = 5
    ) -> dict:
        """
        Estimates a CLIP-style cosine similarity by prompting an OpenAI vision model.

        Args:
            image_path (str): The local path or URL to the image file.
            text (str): The caption or description to compare with the image.
            retries (int, optional): Number of retry attempts in case of API failure. Defaults to 3.
            delay (int, optional): Delay in seconds between retries. Defaults to 5.

        Returns:
            dict: A dictionary containing the cosine similarity or an error message.
        """
        for attempt in range(retries):
            try:
                if not (
                    image_path.startswith(("http://", "https://"))
                    or Path(image_path).exists()
                ):
                    raise FileNotFoundError(
                        f"Image path '{image_path}' does not exist."
                    )

                cosine_similarity = self._request_alignment_score(image_path, text)
                return {"cosine_similarity": cosine_similarity}

            except Exception as e:
                logger.error(
                    f"CLIP score evaluation failed (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return {"error": str(e), "cosine_similarity": 0.0}

        return {
            "error": "Failed to estimate similarity after all retries",
            "cosine_similarity": 0.0,
        }

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
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"

    def evaluate_vqa_score(
        self, image_path: str, text_prompts: list[str]
    ) -> list[float]:
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
        self._ensure_client()

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
        image_url = (
            image_path
            if image_path.startswith("http")
            else self.encode_image_to_base64(image_path)
        )

        for text_prompt in text_prompts:
            user_question = (
                f'Does this figure show "{text_prompt}"? Please answer yes or no.'
            )

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
                completion = self.client.chat.completions.create(
                    model=self.vqa_model_name, messages=messages
                )
                response_text = self._extract_message_text(
                    completion.choices[0].message.content
                )

                json_match = re.search(
                    r"```json\s*({.*?})\s*```|({.*?})", response_text, re.DOTALL
                )
                if not json_match:
                    raise json.JSONDecodeError(
                        "No JSON object found in response", response_text, 0
                    )

                json_str_to_parse = (
                    json_match.group(1) if json_match.group(1) else json_match.group(2)
                )
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
                    logger.warning(
                        f"Warning: 'vqa_score' was not a valid number for prompt '{text_prompt}'. Response: {response_text}"
                    )

            except json.JSONDecodeError:
                logger.error(
                    f"Error: Could not decode JSON for prompt '{text_prompt}'. Response: {response_text}"
                )
            except Exception as e:
                logger.error(f"An API call failed for prompt '{text_prompt}': {e}")

            vqa_scores.append(probability)

        return vqa_scores
