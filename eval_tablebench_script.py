from utils.file_util import read_json_file, write_json_to_file, iter_file_from_dir
from metrics.qa_metrics import QAMetric
import pandas as pd
import os

"""
Evaluation Metric Instruction

This module defines the evaluation metrics used for different types of tasks in the system:

- Fact Checking:
  Uses Exact Match (EM) as the evaluation metric to assess whether the predicted statement exactly matches the reference.

- Numerical Reasoning:
  Also evaluated using Exact Match (EM), focusing on the correctness of numerical outputs.

- Data Analysis:
  Subdivided into specific categories with tailored metrics:
    - Correlation Analysis, Trend Forecasting, and Statistical Analysis:
      Evaluated using EM_with_error_10, allowing a 10% margin of error in numerical predictions.
    - Impact Analysis:
      Assessed with strict Exact Match (EM) to ensure precise identification of influential factors.
    - Others:
      Evaluated using ROUGE-L, suitable for more open-ended or textual responses in data-related tasks.

- Visualization:
  Assessed using Pass@1, which measures the success rate of producing a correct result in the first attempt.

These metric assignments help ensure fair and task-appropriate evaluation across various capabilities.
"""


def merge_parsed_results_to_one_sim_file(parsed_results_dir, eval_results_dir):
    overall_sim_inference_results_path = f'{eval_results_dir}/overall_sim_inference_results.json'
    overall_results = []
    for file_path in iter_file_from_dir(parsed_results_dir):
        exp_results = read_json_file(file_path)
        if not isinstance(exp_results, list):
            exp_results = [exp_results]
        for result in exp_results:
            result.pop('instruction')
            result.pop('prediction')
            overall_results.append(result)
    write_json_to_file(
        overall_sim_inference_results_path, overall_results)
    print(
        f'Save overall results to {overall_sim_inference_results_path}')
    return overall_sim_inference_results_path


def build_categoried_llm_inference_results(candidate_eval_file_paths, eval_models):
    """
    The `categoried_llm_inference_results` structure is formatted as follows:

    {
        "model_name/instruction_type": {
            "merged_type_1": [result_1, result_2, ...],
            "merged_type_2": [result_1, result_2, ...],
            ...
        }
    }

    - The top-level key combines the model name and instruction type, typically separated by a slash (e.g., "GPT-4/DP").
    - Each value is a dictionary that maps a merged task type (e.g., "FactChecking_NumericalReasoning_Aggregation", "FactChecking_MatchBased") to a list of inference results.
    - Each result in the list represents the model's output on an individual instance of that task type.

    This structure is used to organize and analyze inference results by model configuration and task category.
    """
    candidate_eval_results = []
    for file_path in candidate_eval_file_paths:
        data = read_json_file(file_path)
        candidate_eval_results.extend(data)
    categoried_llm_inference_results = {}
    for result in candidate_eval_results:
        model_name = result['model_name']
        if len(eval_models) != 0 and model_name not in eval_models:
            continue
        type = result['qtype']
        subtype = result['qsubtype']
        merged_type = f'{type}_{subtype}'
        instruction_type = result['instruction_type']
        llm_reasoning_method = f'{model_name}/{instruction_type}'
        if llm_reasoning_method not in categoried_llm_inference_results:
            categoried_llm_inference_results[llm_reasoning_method] = {}
        if merged_type not in categoried_llm_inference_results[llm_reasoning_method]:
            categoried_llm_inference_results[llm_reasoning_method][merged_type] = [
            ]
        categoried_llm_inference_results[llm_reasoning_method][merged_type].append(
            result)
    return categoried_llm_inference_results


def eval_by_subtype(categoried_llm_inference_results, metric_eval_engine):
    """
    The `llm_eval_subtype_results` structure is formatted as follows:

    {
        "model_name/instruction_type": {
            "merged_type_1": {
                "metric_1": score,
                "metric_2": score,
                ...
            },
            "merged_type_2": {
                "metric_1": score,
                "metric_2": score,
                ...
            },
            ...
        }
    }

    Description:
    - The top-level key is a string combining the model name and instruction type (e.g., "GPT-4/DP").
    - Each value is a dictionary mapping a merged task type (e.g., "FactChecking_NumericalReasoning_Aggregation", "FactChecking_MatchBased") to its evaluation results.
    - For each task type, metrics (e.g., "EM", "ROUGE-L", "Pass@1") are mapped to their corresponding evaluation scores (usually floats).
    """

    llm_eval_subtype_results_path = f'{EVAL_RESULT_DIR}/llm_eval_subtype_results.json'
    llm_eval_subtype_results = {}
    # evlaute by subtype
    for llm_reasoning_method, result in categoried_llm_inference_results.items():
        print(f'Processing {llm_reasoning_method}...')
        for merged_type, results in result.items():
            # Calculate the metric on visualization
            if merged_type == 'Visualization_ChartGeneration':
                metric_scores = {
                }
                total = len(results)
                metric_scores['total'] = total
                # Code Excution Result
                ecr_1_acc = None
                ecr_1s = [result["parsed_result"].get('ecr_1', None)
                          for result in results]
                ecr_1_acc = ecr_1s.count(True) / total
                metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
                # Code Pass@1 Result
                pass_1_acc = None
                pass_1s = [result["parsed_result"].get('parsed_prediction', None)
                           for result in results]
                pass_1_acc = pass_1s.count(True) / total
                metric_scores['Pass@1'] = round(pass_1_acc * 100, 2)
            else:
                predictions = [result["parsed_result"]
                               ['parsed_prediction'] for result in results]
                references = [result['answer'] for result in results]
                metric_scores = metric_eval_engine.compute(
                    references=references, predictions=predictions)
                total = len(predictions)
                metric_scores['total'] = total
                if llm_reasoning_method.split('/')[1] == 'PoT':
                    ecr_1_acc = None
                    ecr_1s = [result["parsed_result"].get('ecr_1', None)
                              for result in results]
                    ecr_1_acc = ecr_1s.count(True) / total
                    metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
            # Calculate the ratio of successfully parsed results
            parse_1s = [result["parsed_result"].get(
                'Parse@1', None) for result in results]
            metric_scores['Parse@1'] = round(
                parse_1s.count(True) / total * 100, 2)
            if llm_reasoning_method not in llm_eval_subtype_results:
                llm_eval_subtype_results[llm_reasoning_method] = {}
            llm_eval_subtype_results[llm_reasoning_method][merged_type] = metric_scores
    write_json_to_file(llm_eval_subtype_results_path, llm_eval_subtype_results)

    return llm_eval_subtype_results


def eval_by_type(categoried_llm_inference_results, metric_eval_engine):
    """
    The `llm_eval_type_results` structure is formatted as follows::

    {
        "model_name/instruction_type": {
            "FactChecking": {
                "metric_1": score,
                "metric_2": score,
                ...
            },
            "NumericalReasoning": {
                "metric_1": score,
                "metric_2": score,
                ...
            },
            "DataAnalysis": {
                "metric_1": score,
                "metric_2": score,
                ...
            },
            "Visualization": {
                "metric_1": score,
                "metric_2": score,
                ...
            },
            "Overall": {
                "metric_1": score,
                "metric_2": score,
                ...
            }
        }
    }

    Description:
    - The top-level key is a string combining the model name and instruction type (e.g., "GPT4/DP").
    - Each value is a dictionary mapping an individual task type (e.g., "FactChecking", "DataAnalysis") to its evaluation results.
    - For each task type, the dictionary contains metric names (such as "EM", "ROUGE-L", "Pass@1") mapped to their respective scores (typically floats).

    This structure provides a fine-grained view of evaluation scores per task type for each model and instruction configuration.
    """
    llm_eval_type_results_path = f'{EVAL_RESULT_DIR}/llm_eval_type_results.json'
    llm_eval_type_results = {}
    # evlaute by subtype
    for llm_reasoning_method, result in categoried_llm_inference_results.items():
        llm_eval_type_result = {}
        qtype_dict = {}
        print(f'Processing {llm_reasoning_method}...')
        # categoried single exp results by qtype
        for merged_type, results in result.items():
            qtype = merged_type.split('_')[0]
            if qtype not in qtype_dict:
                qtype_dict[qtype] = []
            qtype_dict[qtype].extend(results)
        # calculate the metric for each qtype
        # == FactChecking ==
        fc_results = qtype_dict['FactChecking']
        fc_predictions = [fc_result["parsed_result"]
                          ['parsed_prediction'] for fc_result in fc_results]
        fc_references = [fc_result['answer'] for fc_result in fc_results]
        fc_metric_scores = metric_eval_engine.compute(
            references=fc_references, predictions=fc_predictions)
        # Calculate the ratio of successfully parsed results
        fc_parse_1s = [fc_result["parsed_result"].get(
            'Parse@1', None) for fc_result in fc_results]
        fc_metric_scores['Parse@1'] = round(
            fc_parse_1s.count(True) / len(fc_results) * 100, 2)
        fc_metric_scores['total'] = len(fc_results)
        # In Case Instruction type is PoT
        if llm_reasoning_method.split('/')[1] == 'PoT':
            # Calculate the ratio of successfully excution results
            fc_ecr_1s = [fc_result["parsed_result"].get(
                'ecr_1', None) for fc_result in fc_results]
            fc_metric_scores['ECR@1'] = round(
                fc_ecr_1s.count(True) / len(fc_results) * 100, 2)
        llm_eval_type_result['FactChecking'] = fc_metric_scores

        # == NumericalReasoning ==
        nr_results = qtype_dict['NumericalReasoning']
        nr_predictions = [nr_result["parsed_result"]
                          ['parsed_prediction'] for nr_result in nr_results]
        nr_references = [nr_result['answer'] for nr_result in nr_results]
        nr_metric_scores = metric_eval_engine.compute(
            references=nr_references, predictions=nr_predictions)
        # Calculate the ratio of successfully parsed results
        nr_parse_1s = [nr_result["parsed_result"].get(
            'Parse@1', None) for nr_result in nr_results]
        nr_metric_scores['Parse@1'] = round(
            nr_parse_1s.count(True) / len(nr_results) * 100, 2)
        nr_metric_scores['total'] = len(nr_results)
        # In Case Instruction type is PoT
        if llm_reasoning_method.split('/')[1] == 'PoT':
            # Calculate the ratio of successfully excution results
            nr_ecr_1s = [nr_result["parsed_result"].get(
                'ecr_1', None) for nr_result in nr_results]
            nr_metric_scores['ECR@1'] = round(
                nr_ecr_1s.count(True) / len(nr_results) * 100, 2)
        llm_eval_type_result['NumericalReasoning'] = nr_metric_scores

        # == DataAnalysis ==
        da_results = qtype_dict['DataAnalysis']
        da_predictions = [da_result["parsed_result"]
                          ['parsed_prediction'] for da_result in da_results]
        da_references = [da_result['answer'] for da_result in da_results]
        da_metric_scores = metric_eval_engine.compute(
            references=da_references, predictions=da_predictions)

        # Due to the differences in sub-tasks of data analysis, a mixed metric needs to be calculated.
        da_em_part_results = []
        da_em_with_error_10_part_results = []
        da_rouge_part_results = []
        for da_result in da_results:
            da_subtype = da_result['qsubtype']
            if da_subtype == 'CorrelationAnalysis' or da_subtype == 'TrendForecasting' or da_subtype == 'StatisticalAnalysis':
                da_em_with_error_10_part_results.append(da_result)
            elif da_subtype == 'ImpactAnalysis':
                da_em_part_results.append(da_result)
            else:
                da_rouge_part_results.append(da_result)

        da_em_total = len(da_em_part_results)
        da_em_references = [da_result['answer']
                            for da_result in da_em_part_results]
        da_em_predictions = [da_result['parsed_result']
                             ['parsed_prediction'] for da_result in da_em_part_results]
        da_em = metric_eval_engine.compute(
            references=da_em_references, predictions=da_em_predictions)['EM']

        da_em_with_error_10_total = len(da_em_with_error_10_part_results)
        da_em_with_error_10_references = [
            da_result['answer'] for da_result in da_em_with_error_10_part_results]
        da_em_with_error_10_predictions = [
            da_result['parsed_result']['parsed_prediction'] for da_result in da_em_with_error_10_part_results]
        da_em_with_error_10 = metric_eval_engine.compute(
            references=da_em_with_error_10_references, predictions=da_em_with_error_10_predictions)['EM_with_error_10']

        da_rouge_total = len(da_rouge_part_results)
        da_rouge_references = [da_result['answer']
                               for da_result in da_rouge_part_results]
        da_rouge_predictions = [da_result['parsed_result']
                                ['parsed_prediction'] for da_result in da_rouge_part_results]
        da_rouge = metric_eval_engine.compute(
            references=da_rouge_references, predictions=da_rouge_predictions)['ROUGE-L']

        da_mix_metric = (da_em * da_em_total + da_em_with_error_10 * da_em_with_error_10_total + da_rouge * da_rouge_total) / (
            da_em_total + da_em_with_error_10_total + da_rouge_total)

        da_metric_scores['Mix_Metric'] = round(da_mix_metric, 2)

        # Calculate the ratio of successfully parsed results
        da_parse_1s = [da_result["parsed_result"].get(
            'Parse@1', None) for da_result in da_results]
        da_metric_scores['Parse@1'] = round(
            da_parse_1s.count(True) / len(da_results) * 100, 2)
        da_metric_scores['total'] = len(da_results)
        # In Case Instruction type is PoT
        if llm_reasoning_method.split('/')[1] == 'PoT':
            # Calculate the ratio of successfully excution results
            da_ecr_1s = [da_result["parsed_result"].get(
                'ecr_1', None) for da_result in da_results]
            da_metric_scores['ECR@1'] = round(
                da_ecr_1s.count(True) / len(da_results) * 100, 2)
        llm_eval_type_result['DataAnalysis'] = da_metric_scores

        # == Visualization ==
        vis_results = qtype_dict['Visualization']
        vis_metric_scores = {}
        vis_metric_scores['total'] = len(vis_results)
        # Code Excution Ratio
        ecr_1_acc = None
        ecr_1s = [vis_result["parsed_result"].get('ecr_1', None)
                  for vis_result in vis_results]
        ecr_1_acc = ecr_1s.count(True) / len(vis_results)
        vis_metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
        # Code Pass@1 Result
        pass_1_acc = None
        pass_1s = [vis_result["parsed_result"].get('parsed_prediction', None)
                   for vis_result in vis_results]
        pass_1_acc = pass_1s.count(True) / len(vis_results)
        vis_metric_scores['Pass@1'] = round(pass_1_acc * 100, 2)
        llm_eval_type_result['Visualization'] = vis_metric_scores

        # == Overall ==
        overall_em_results = fc_results + nr_results + da_em_part_results
        overall_em_with_error_10_results = da_em_with_error_10_part_results
        overall_rouge_results = da_rouge_part_results
        overall_pass_1_results = vis_results

        overall_em_predictions = [result["parsed_result"]
                                  ['parsed_prediction'] for result in overall_em_results]
        overall_em_references = [result['answer']
                                 for result in overall_em_results]
        overall_em_score = metric_eval_engine.compute(
            references=overall_em_references, predictions=overall_em_predictions)['EM']
        overall_em_total = len(overall_em_results)

        overall_em_with_error_10_predictions = [result["parsed_result"]
                                                ['parsed_prediction'] for result in overall_em_with_error_10_results]
        overall_em_with_error_10_references = [result['answer']
                                               for result in overall_em_with_error_10_results]
        overall_em_with_error_10_score = metric_eval_engine.compute(
            references=overall_em_with_error_10_references, predictions=overall_em_with_error_10_predictions)['EM_with_error_10']
        overall_em_with_error_10_total = len(overall_em_with_error_10_results)

        overall_rouge_predictions = [result["parsed_result"]
                                     ['parsed_prediction'] for result in overall_rouge_results]
        overall_rouge_references = [result['answer']
                                    for result in overall_rouge_results]
        overall_rouge_score = metric_eval_engine.compute(
            references=overall_rouge_references, predictions=overall_rouge_predictions)['ROUGE-L']
        overall_rouge_total = len(overall_rouge_results)

        overall_pass_1_predictions = [result["parsed_result"]
                                      ['parsed_prediction'] for result in overall_pass_1_results]
        overall_pass_1_total = len(overall_pass_1_results)
        overall_pass_1_score = overall_pass_1_predictions.count(
            True) / overall_pass_1_total

        overall_score = (overall_em_score * overall_em_total + overall_em_with_error_10_score * overall_em_with_error_10_total + overall_rouge_score * overall_rouge_total + overall_pass_1_score * overall_pass_1_total) / (
            overall_em_total + overall_em_with_error_10_total + overall_rouge_total + overall_pass_1_total)
        llm_eval_type_result['Overall'] = {
            "Mix_Metric": round(overall_score, 2)}

        # Calculate the ratio of successfully parsed results
        overall_results = fc_results + nr_results + da_results + vis_results
        overall_parse_1s = [result["parsed_result"].get(
            'Parse@1', None) for result in overall_results]
        llm_eval_type_result['Overall']["Parse@1"] = round(
            overall_parse_1s.count(True) / len(overall_results) * 100, 2)
        # In Case Instruction type is PoT
        if llm_reasoning_method.split('/')[1] == 'PoT':
            # Calculate the ratio of successfully excution results
            overall_ecr_1s = [result["parsed_result"].get(
                'ecr_1', None) for result in overall_results]
            llm_eval_type_result['Overall']['ECR@1'] = round(
                overall_ecr_1s.count(True) / len(overall_results) * 100, 2)

        # save the llm_eval_type_result
        llm_eval_type_results[llm_reasoning_method] = llm_eval_type_result

    write_json_to_file(llm_eval_type_results_path, llm_eval_type_results)
    return llm_eval_type_results


def save_subtype_results_to_csv(llm_eval_subtype_results, llm_eval_subtype_results_csv_path):
    # Save the llm_eval_subtype_results to csv
    llm_eval_subtype_csv_results = []
    for llm_reasoning_method, result in llm_eval_subtype_results.items():
        csv_result = {
            'model_name': llm_reasoning_method.split('/')[0],
            'instruction_type': llm_reasoning_method.split('/')[1]
        }
        for merged_type, metric_scores in result.items():
            qtype = merged_type.split('_')[0]
            subtype = merged_type.split('_')[1]
            total = metric_scores['total']
            if qtype == 'Visualization':
                merge_type_score = metric_scores['Pass@1']
            elif qtype == 'NumericalReasoning' or qtype == 'FactChecking':
                merge_type_score = metric_scores['EM']
            elif qtype == 'DataAnalysis':
                if subtype == 'CorrelationAnalysis' or subtype == 'TrendForecasting' or subtype == 'StatisticalAnalysis':
                    merge_type_score = metric_scores['EM_with_error_10']
                elif subtype == 'ImpactAnalysis':
                    merge_type_score = metric_scores['EM']
                else:
                    merge_type_score = metric_scores['ROUGE-L']
            csv_result[merged_type] = merge_type_score
        # calculate the overall results
        llm_eval_subtype_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_subtype_csv_results)
    llm_eval_df.to_csv(
        llm_eval_subtype_results_csv_path, index=False, sep='\t')


def save_type_results_to_csv(llm_eval_type_results, llm_eval_type_results_csv_path):
    # Save the llm_eval_type_results to csv
    llm_eval_type_csv_results = []
    for llm_reasoning_method, result in llm_eval_type_results.items():
        csv_result = {
            'model_name': llm_reasoning_method.split('/')[0],
            'instruction_type': llm_reasoning_method.split('/')[1]
        }
        for qtype, metric_scores in result.items():
            if qtype == 'Visualization':
                merge_type_score = metric_scores['Pass@1']
            elif qtype == 'NumericalReasoning' or qtype == 'FactChecking':
                merge_type_score = metric_scores['EM']
            elif qtype == 'DataAnalysis' or qtype == 'Overall':
                merge_type_score = metric_scores['Mix_Metric']
            csv_result[qtype] = merge_type_score
        # calculate the overall results
        llm_eval_type_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_type_csv_results)
    llm_eval_df.to_csv(
        llm_eval_type_results_csv_path, index=False, sep='\t')


if __name__ == '__main__':

    # ==== Global settings ====
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = 'eval_examples'

    PARSED_RUSULT_DIR = f'{PROJECT_ROOT_DIR}/{EXP_DIR}/parsed_results'
    EVAL_RESULT_DIR = f'{PROJECT_ROOT_DIR}/{EXP_DIR}/evaluation_results'

    # Init metric evaluation engine
    metric_eval_engine = QAMetric()

    # Merge parsed results to one file
    overall_sim_inference_results_path = merge_parsed_results_to_one_sim_file(
        PARSED_RUSULT_DIR, EVAL_RESULT_DIR)

    # Support add multiple inference result files
    candidate_eval_file_paths = [
        overall_sim_inference_results_path
    ]

    # Support eval targeted models,if eval_models is empty, all models will be eval
    eval_models = [
        # 'o3-mini-2025-01-31',
    ]

    # Load the candidate evaluation results
    categoried_llm_inference_results = build_categoried_llm_inference_results(
        candidate_eval_file_paths, eval_models)

    # Evaluate by subtype
    print('==== Evaluate by subtype ====')
    llm_eval_subtype_results = eval_by_subtype(
        categoried_llm_inference_results, metric_eval_engine)
    # sve_subtype_results to csv
    llm_eval_subtype_results_csv_path = f'{EVAL_RESULT_DIR}/llm_eval_subtype_results.csv'
    save_subtype_results_to_csv(
        llm_eval_subtype_results, llm_eval_subtype_results_csv_path)

    # Evaluate by type
    print('==== Evaluate by type ====')
    llm_eval_type_results = eval_by_type(
        categoried_llm_inference_results, metric_eval_engine)
    # save_type_results to csv
    llm_eval_type_results_csv_path = f'{EVAL_RESULT_DIR}/llm_eval_type_results.csv'
    save_type_results_to_csv(
        llm_eval_type_results, llm_eval_type_results_csv_path)
