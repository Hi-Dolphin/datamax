import numpy as np
import sacrebleu
from bert_score import score as bert_scorer
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer


class TextQualityEvaluator:
    """
    Evaluates the quality of text using various metrics like BERTScore, ROUGE, BLEU, and Self-CIDEr.
    """

    def __init__(self, lang: str = "zh"):
        """
        Initializes the text quality evaluator.

        Args:
            lang (str): The language of the text to be evaluated (e.g., "en", "zh").
        """
        self.lang = lang
        if rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        else:
            self.rouge_scorer = None

    def evaluate_bertscore(self, candidates: list[str], references: list[str]) -> dict:
        """
        Calculates BERTScore between candidate and reference sentences.

        Args:
            candidates (list[str]): List of generated sentences.
            references (list[str]): List of reference sentences.

        Returns:
            dict: A dictionary containing precision, recall, and F1 scores.
        """
        if not bert_scorer:
            raise ImportError(
                "BERT-Score is not installed. Please run 'pip install bert-score'."
            )

        P, R, F1 = bert_scorer(
            cands=candidates, refs=references, lang=self.lang, verbose=True
        )
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }

    def evaluate_rouge(self, candidate: str, reference: str) -> dict:
        """
        Calculates ROUGE scores for a candidate sentence against a reference.

        Args:
            candidate (str): The generated sentence.
            reference (str): The reference sentence.

        Returns:
            dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        if not self.rouge_scorer:
            raise ImportError(
                "ROUGE-Score is not installed. Please run 'pip install rouge-score'."
            )

        scores = self.rouge_scorer.score(reference, candidate)
        return {
            key: {
                "precision": value.precision,
                "recall": value.recall,
                "f1": value.fmeasure,
            }
            for key, value in scores.items()
        }

    def evaluate_bleu(self, candidate: str, references: list[str]) -> float:
        """
        Calculates BLEU score for a candidate sentence against one or more references.

        Args:
            candidate (str): The generated sentence.
            references (list[str]): A list of reference sentences.

        Returns:
            float: The BLEU score.
        """
        if not sacrebleu:
            raise ImportError(
                "sacrebleu is not installed. Please run 'pip install sacrebleu'."
            )

        bleu = sacrebleu.corpus_bleu([candidate], [references])
        return bleu.score

    def calculate_self_cider_diversity(self, captions: list[str]) -> float:
        """
        Calculates the semantic diversity of a set of captions using the Self-CIDEr metric.
        A higher score indicates greater semantic diversity.

        Args:
            captions (list[str]): A list of captions to evaluate.

        Returns:
            float: The Self-CIDEr diversity score.
        """
        if not Cider:
            raise ImportError(
                "pycocoevalcap is not installed. Please run 'pip install pycocoevalcap'."
            )

        if not captions or len(captions) < 2:
            return 0.0

        num_captions = len(captions)
        cider_scorer = Cider()

        similarity_matrix = np.zeros((num_captions, num_captions))

        for i in range(num_captions):
            for j in range(num_captions):
                res = {0: [captions[j]]}
                gts = {0: [captions[i]]}
                _, individual_scores = cider_scorer.compute_score(gts, res)
                similarity_matrix[i, j] = individual_scores[0]

        eigenvalues, _ = np.linalg.eigh(similarity_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigenvalues = np.sqrt(eigenvalues)

        if np.sum(sqrt_eigenvalues) == 0:
            return 0.0

        ratio = np.max(sqrt_eigenvalues) / np.sum(sqrt_eigenvalues)
        ratio = max(ratio, 1e-9)

        diversity_score = (
            -np.log(ratio) / np.log(num_captions) if num_captions > 1 else 0.0
        )

        return diversity_score
