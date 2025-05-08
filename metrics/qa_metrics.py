import os
import sys

from datasets import load_metric, list_metrics
from metrics.base_metric import BaseMetric
import re
import string
import evaluate
from metrics.custom_em_metric import compute_em, compute_em_with_tolerance

sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip


def show_all_metrics():
    metrics_list = list_metrics()
    for metric in metrics_list:
        print(metric)


def show_detail_metric(metric_name):
    metric = load_metric(metric_name)
    print(metric)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class QAMetric(BaseMetric):

    def __init__(self, **kwargs):
        self.rouge = evaluate.load('rouge')

    def prepsocess(self, references, predictions):
        '''
        Preprocess predictions and references
        '''
        processed_predictions = []
        processed_references = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference = references[i]
            # normalize prediction and reference
            prediction = normalize_answer(prediction)
            reference = normalize_answer(reference)
            # add prediction and reference to processed list
            processed_predictions.append(prediction)
            processed_references.append(reference)
        predictions = processed_predictions
        references = processed_references
        return references, predictions

    def compute(self, references, predictions):
        '''
        Support Mtrics: EM, ROUGE-L
        '''
        metric_scores = {}
        references, predictions = self.prepsocess(references, predictions)

        sys.setrecursionlimit(8735 * 2080 + 10)
        # calculate F1,EM, ROUGE-L, SacreBLEU, Meteor
        em_score = compute_em(references=references, predictions=predictions)
        em_score_with_error_2 = compute_em_with_tolerance(
            references=references, predictions=predictions, error_range=2)
        em_score_with_error_5 = compute_em_with_tolerance(
            references=references, predictions=predictions, error_range=5)
        em_score_with_error_10 = compute_em_with_tolerance(
            references=references, predictions=predictions, error_range=10)
        rouge_score = self.rouge.compute(
            references=references, predictions=predictions)

        metric_scores = {
            'EM': round(em_score*100, 2),
            'EM_with_error_2': round(em_score_with_error_2*100, 2),
            'EM_with_error_5': round(em_score_with_error_5*100, 2),
            'EM_with_error_10': round(em_score_with_error_10*100, 2),
            'ROUGE-L': round(rouge_score['rougeL']*100, 2),
        }

        return metric_scores
