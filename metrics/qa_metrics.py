""" Evaluation script for RAG models."""
import os
import sys

from datasets import load_metric,list_metrics
from metrics.base_metric import BaseMetric
import re
import string
from collections import Counter
import evaluate

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

def word_level_f1_score(references,predictions):
    '''
    Word Level F1 Score
    '''
    num_same_words = 0
    num_pred_words = 0
    num_ref_words = 0
    for reference, prediction in zip(references, predictions):
        prediction_words = prediction.split()
        reference_words = reference.split()
        common = Counter(prediction_words) & Counter(reference_words)
        num_same = sum(common.values())
        num_pred_words += len(prediction_words)
        num_ref_words += len(reference_words)
        num_same_words += num_same
    if num_same_words==0:
        return 0.0
    else:
        precision = 1.0 * num_same_words / num_pred_words
        recall = 1.0 * num_same_words / num_ref_words
        f1 = (2 * precision * recall) / (precision + recall)
    return f1


class QAMetric(BaseMetric):

    def __init__(self, **kwargs):
        self.em = evaluate.load('exact_match')
        self.rouge = evaluate.load('rouge')
        self.sacrebleu = evaluate.load('sacrebleu')
        self.bertscore = evaluate.load('bertscore')
        print(f'Successfully loaded F1, EM, ROUGE, SacreBLEU,BERTScore')
        self.count_blank=True
        # self.meteor = evaluate.load('meteor')
    
    def prepsocess(self,references,predictions):
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
            if len(prediction) == 0:
                if self.count_blank:
                    prediction = '#'
                    processed_predictions.append(prediction)
                    processed_references.append(reference)
            else:
                processed_predictions.append(prediction)
                processed_references.append(reference)
        predictions = processed_predictions
        references = processed_references
        return references,predictions


    def compute(self,references,predictions):
        '''
        Support Mtrics: F1, EM, ROUGE-L, SacreBLEU, Meteor
        '''
        metric_scores = {}
        references,predictions = self.prepsocess(references,predictions)

        sys.setrecursionlimit(8735 * 2080 + 10)
        # calculate F1,EM, ROUGE-L, SacreBLEU, Meteor
        f1_score = word_level_f1_score(references=references, predictions=predictions)
        em_score = self.em.compute(references=references, predictions=predictions)
        rouge_score = self.rouge.compute(references=references, predictions=predictions)
        sacrebleu_score = self.sacrebleu.compute(
            references=references, predictions=predictions)

        metric_scores = {
            'F1': round(f1_score*100, 2),
            'EM': round(em_score['exact_match']*100, 2),
            'ROUGE-L': round(rouge_score['rougeL']*100, 2),
            'SacreBLEU': round(sacrebleu_score['score'], 2),
        }

        return metric_scores

