import re
from typing import List
from decimal import Decimal, ROUND_HALF_UP


def normalize_number(value: str) -> Decimal:
    """Convert the string to Decimal, supporting percentages."""
    if value.endswith('%'):
        value = value.strip('%')
        decimal_value = Decimal(value) / Decimal('100')
        return decimal_value.quantize(Decimal('1.0000'), rounding=ROUND_HALF_UP)
    return Decimal(value)


def get_decimal_precision(values: List[str]) -> int:
    """Get the smallest number of decimal places in a set of references (without percentages)"""
    precisions = []
    for val in values:
        if val.endswith('%'):
            continue
        if '.' in val:
            precisions.append(len(val.split('.')[-1]))
        else:
            precisions.append(0)
    return min(precisions) if precisions else 0


def round_decimal(value: Decimal, precision: int) -> str:
    """Rounded to specified decimal places"""
    rounding_format = f'1.{"0" * precision}'
    return str(value.quantize(Decimal(rounding_format), rounding=ROUND_HALF_UP))


def is_number(val: str) -> bool:
    """Determine if it is in the form of a number or percentage"""
    val = val.strip()
    return bool(re.match(r'^-?\d+(\.\d+)?%?$', val))


def compute_em(references: List[str], predictions: List[str]) -> float:
    """Evaluate overall EM values and consider inconsistencies in the number of predicted outcomes"""
    total_score = 0.0
    total_count = 0

    for pred, ref in zip(predictions, references):
        ref_answers = [x.strip() for x in ref.split(',')]
        pred_answers = [x.strip() for x in pred.split(',')]

        match_score = 0.0
        weight = 1.0 / len(ref_answers)

        for i, r in enumerate(ref_answers):
            if i >= len(pred_answers):
                continue
            p = pred_answers[i]
            if is_number(r):
                try:
                    if r.endswith('%'):
                        norm_r = normalize_number(r)
                        norm_p = normalize_number(p)
                        if norm_r == norm_p:
                            match_score += weight
                    else:
                        ref_vals = [x for x in ref_answers if is_number(
                            x) and not x.endswith('%')]
                        precision = get_decimal_precision(ref_vals)
                        norm_r = round_decimal(
                            normalize_number(r), precision)
                        norm_p = round_decimal(
                            normalize_number(p), precision)
                        if norm_r == norm_p:
                            match_score += weight
                except:
                    continue
            else:
                if r == p:
                    match_score += weight

        total_score += match_score
        total_count += 1

    return total_score / total_count if total_count else 0.0


def compute_em_with_tolerance(references: List[str], predictions: List[str], error_range: float) -> float:
    """Evaluation of EM values, numerical categories within the margin of error, in percent (e.g., 5 for 5%)"""
    total_score = 0.0
    total_count = 0

    for pred, ref in zip(predictions, references):
        ref_answers = [x.strip() for x in ref.split(',')]
        pred_answers = [x.strip() for x in pred.split(',')]

        match_score = 0.0
        weight = 1.0 / len(ref_answers)

        for i, r in enumerate(ref_answers):
            if i >= len(pred_answers):
                continue
            p = pred_answers[i]

            if is_number(r):
                try:
                    val_r = normalize_number(r)
                    val_p = normalize_number(p)

                    if val_r == Decimal('0'):
                        if val_p == val_r:
                            match_score += weight
                    else:
                        error = abs(val_r - val_p) / abs(val_r)
                        if error <= error_range / 100:
                            match_score += weight
                except:
                    continue
            else:
                if r == p:
                    match_score += weight

        total_score += match_score
        total_count += 1

    return total_score / total_count if total_count else 0.0
