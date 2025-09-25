"""
Evaluate generated text quality with BERTScore.
Requires bert-score installed.
"""
from datamax.evaluator import TextQualityEvaluator


def main():
    cand = ["智能体在航运场景中的应用包括……"]
    refs = ["航运场景中，智能体主要应用于……"]
    e = TextQualityEvaluator(lang="zh")
    scores = e.evaluate_bertscore(cand, refs)
    print(scores)


if __name__ == "__main__":
    main()

