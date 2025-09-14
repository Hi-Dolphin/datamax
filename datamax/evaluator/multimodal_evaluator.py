import torch
import t2v_metrics
import time
from loguru import logger
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class MultimodalConsistencyEvaluator:
    """
    Evaluates the consistency between images and text using models like CLIP and VQA.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", vqa_model_name: str = 'clip-flant5-xxl', device: str = None):
        """
        Initializes the multimodal consistency evaluator.

        Args:
            clip_model_name (str): The name of the CLIP model to use.
            vqa_model_name (str): The name of the VQA model to use.
            device (str): The device to run the models on ('cuda' or 'cpu'). Auto-detects if None.
        """
        if not torch:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch torchvision'.")

        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_name = clip_model_name
        self.vqa_model_name = vqa_model_name

        self._clip_model = None
        self._clip_processor = None
        self._clip_tokenizer = None # Add tokenizer for truncation
        self._vqa_scorer = None

    def _load_clip(self):
        if not self._clip_model:
            if not CLIPProcessor or not CLIPModel:
                 raise ImportError("Transformers is not installed. Please run 'pip install transformers'.")
            logger.info(f"Loading CLIP model '{self.clip_model_name}' to {self.device}...")
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name) # Load tokenizer

    def _load_vqa(self):
        if not self._vqa_scorer:
            if not t2v_metrics:
                raise ImportError("t2v-metrics is not installed. Please run 'pip install t2v-metrics'.")
            logger.info(f"Loading VQAScore model '{self.vqa_model_name}'...")
            self._vqa_scorer = t2v_metrics.VQAScore(model=self.vqa_model_name)

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncates text to a maximum token length."""
        tokens = self._clip_tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return self._clip_tokenizer.decode(tokens, skip_special_tokens=True)

    def evaluate_clip_score(self, image_path: str, text: str, retries: int = 3, delay: int = 5) -> dict:
        """
        Calculates CLIPScore, representing the similarity between an image and a text description.

        Args:
            image_path (str): The path to the image file.
            text (str): The text description.
            retries (int): Number of retries for network errors.
            delay (int): Delay between retries in seconds.

        Returns:
            dict: A dictionary containing the raw logits score and cosine similarity.
        """
        self._load_clip()
        
        # Truncate text to max length of CLIP model (77)
        truncated_text = self._truncate_text(text, 77)
        if len(text) != len(truncated_text):
            logger.warning(f"Input text truncated for CLIP evaluation. Original length: {len(text)}, Truncated length: {len(truncated_text)}")
            text = truncated_text

        for attempt in range(retries):
            try:
                image = Image.open(image_path)
                inputs = self._clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)

                with torch.no_grad():
                    outputs = self._clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    clip_score = logits_per_image.item()

                    # Also calculate cosine similarity
                    image_features = self._clip_model.get_image_features(inputs['pixel_values'])
                    text_features = self._clip_model.get_text_features(inputs['input_ids'])
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    cosine_similarity = (image_features @ text_features.T).item()

                return {"clip_score_logits": clip_score, "cosine_similarity": cosine_similarity}

            except FileNotFoundError:
                logger.error(f"Image file not found at {image_path}")
                return {"error": "File not found"}
            except Exception as e:
                logger.error(f"An error occurred during CLIP scoring (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return {"error": str(e)}

    def evaluate_vqa_score(self, image_path: str, captions: list[str], retries: int = 3, delay: int = 5) -> list[float]:
        """
        Calculates VQAScore, which measures how likely a VQA model would answer 'yes' to a
        question asking if the caption describes the image.

        Args:
            image_path (str): The path to the image file.
            captions (list[str]): A list of captions to evaluate against the image.
            retries (int): Number of retries for network errors.
            delay (int): Delay between retries in seconds.

        Returns:
            list[float]: A list of VQA scores, one for each caption.
        """
        self._load_vqa()
        for attempt in range(retries):
            try:
                image = Image.open(image_path)
                scores = self._vqa_scorer.score(pil_images=[image], texts=captions)
                return scores
            except FileNotFoundError:
                logger.error(f"Image file not found at {image_path}")
                return []
            except Exception as e:
                logger.error(f"An error occurred during VQA scoring (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return []