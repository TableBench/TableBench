from utils.file_util import read_json_file, write_json_to_file, iter_file_from_dir
import os
from metrics.qa_metrics import QAMetric
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
global PARSED_RUSULT_DIR
global EVAL_RESULT_DIR
global eval_models



def merge_parsed_results_to_one_file():
    overall_sim_inference_results_path = f'{EVAL_RESULT_DIR}/overall_sim_inference_results.json'
    overall_results = []
    for dir in os.listdir(PARSED_RUSULT_DIR):
        for file_path in iter_file_from_dir(f'{PARSED_RUSULT_DIR}/{dir}'):
            file_name = file_path.split('/')[-1]
            model_name = file_name.split('_')[0]
            if len(eval_models) != 0:
                if model_name not in eval_models:
                    continue
            if 'TableBench_PoT' in file_name:
                prompt_type = 'PoT'
            elif 'TableBench_SCoT' in file_name:
                prompt_type = 'SCoT'
            elif 'TableBench_TCoT' in file_name:
                prompt_type = 'TCoT'
            elif 'TableBench_DP' in file_name:
                prompt_type = 'DP'
            exp_results = read_json_file(file_path)
            if not isinstance(exp_results, list):
                exp_results = [exp_results]
            simple_results = []
            for result in exp_results:
                new_result = {
                    "id": result['id'],
                    'prompt_type': prompt_type,
                    'model_name': model_name,
                    "qtype": result['qtype'],
                    "qsubtype": result['qsubtype'],
                    "question": result['question'],
                    "answer": result['answer'],
                    "parsed_result": result['parsed_result']
                }
                sim_result = {
                    "id": result['id'],
                    'prompt_type': prompt_type,
                    'model_name': model_name,
                    "qtype": result['qtype'],
                    "qsubtype": result['qsubtype'],
                    "question": result['question'],
                    "answer": result['answer'],
                    "prediction": result['prediction'],
                    "parsed_result": result['parsed_result']
                }
                simple_results.append(sim_result)
                overall_results.append(new_result)
            write_json_to_file(
                f'{EVAL_RESULT_DIR}/sim_results/{file_name}', simple_results, is_json_line=True)
    write_json_to_file(
        overall_sim_inference_results_path, overall_results)
    print(
        f'Save overall results to {overall_sim_inference_results_path}')


def build_categoried_llm_inference_results():
    '''
    categoried_llm_inference_results format is:
    {
        "model_name/prompt_type": { "merged_type": [result1, result2, ...] }
    }
    '''
    overall_sim_inference_results_path = f'{EVAL_RESULT_DIR}/overall_sim_inference_results.json'
    overall_sim_inference_results = read_json_file(
        overall_sim_inference_results_path)
    categoried_llm_inference_results = {}
    for result in overall_sim_inference_results:
        model_name = result['model_name']
        type = result['qtype']
        subtype = result['qsubtype']
        merged_type = f'{type}_{subtype}'
        prompt_type = result['prompt_type']
        key = f'{model_name}/{prompt_type}'
        if key not in categoried_llm_inference_results:
            categoried_llm_inference_results[key] = {}
        if merged_type not in categoried_llm_inference_results[key]:
            categoried_llm_inference_results[key][merged_type] = []
        categoried_llm_inference_results[key][merged_type].append(result)
    return categoried_llm_inference_results


def eval_by_subtype(categoried_llm_inference_results, qa_metric, metric_name='ROUGE-L'):
    llm_eval_subtype_results_path = f'{EVAL_RESULT_DIR}/llm_eval_subtype_results.json'
    llm_eval_subtype_results_csv_path = f'{EVAL_RESULT_DIR}/llm_eval_subtype_results.csv'
    llm_eval_subtype_results = {}
    for key, result in categoried_llm_inference_results.items():
        print(f'Processing {key}...')
        for merge_type, results in result.items():
            if merge_type == 'Visualization_ChartGeneration':
                metric_scores = {
                    'F1': 0,
                    'EM': 0,
                    'ROUGE-L': 0,
                    'SacreBLEU': 0,
                }
                total = len(results)
                metric_scores['total'] = total
                ecr_1_acc = None
                ecr_1s = [result["parsed_result"].get('ecr_1', None)
                          for result in results]
                ecr_1_acc = ecr_1s.count(True) / total
                metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
                pass_1_acc = None
                parsed_prediction_results = []
                for result in results:
                    parsed_prediction = result["parsed_result"]['parsed_prediction']
                    if parsed_prediction == 'True':
                        parsed_prediction_results.append(True)
                    elif parsed_prediction == 'False':
                        parsed_prediction_results.append(False)
                    else:
                        parsed_prediction_results.append('None')
                pass_1_acc = parsed_prediction_results.count(True) / total
                metric_scores['Pass@1'] = round(pass_1_acc*100, 2)
            else:
                predictions = [result["parsed_result"]
                               ['parsed_prediction'] for result in results]
                references = [result['answer'] for result in results]
                metric_scores = qa_metric.compute(predictions, references)
                total = len(predictions)
                metric_scores['total'] = total
                if key.split('/')[1] == 'PoT':
                    ecr_1_acc = None
                    ecr_1s = [result["parsed_result"].get('ecr_1', None)
                              for result in results]
                    ecr_1_acc = ecr_1s.count(True) / total
                    metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
            parse_1s = [result["parsed_result"].get(
                'Parse@1', None) for result in results]
            metric_scores['Parse@1'] = round(
                parse_1s.count(True) / total * 100, 2)
            if key not in llm_eval_subtype_results:
                llm_eval_subtype_results[key] = {}
            llm_eval_subtype_results[key][merge_type] = metric_scores
    write_json_to_file(llm_eval_subtype_results_path, llm_eval_subtype_results)

    llm_eval_subtype_csv_results = []
    for key, result in llm_eval_subtype_results.items():
        csv_result = {
            'model_name': key.split('/')[0],
            'prompt_type': key.split('/')[1]
        }
        for merge_type, metric_scores in result.items():
            if merge_type == 'Visualization_ChartGeneration':
                csv_result[merge_type] = metric_scores['Pass@1']
            else:
                csv_result[merge_type] = metric_scores[metric_name]
        llm_eval_subtype_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_subtype_csv_results)
    llm_eval_df.to_csv(llm_eval_subtype_results_csv_path,
                       index=False, sep='\t')

def eval_by_type(categoried_llm_inference_results, qa_metric, metric_name='ROUGE-L'):
    llm_eval_type_results_path = f'{EVAL_RESULT_DIR}/llm_eval_type_results.json'
    llm_eval_type_results_csv_path = f'{EVAL_RESULT_DIR}/llm_eval_type_results.csv'
    llm_eval_type_results = {}
    for key, result in categoried_llm_inference_results.items():
        type_dict = {}
        print(f'Processing {key}...')
        for merge_type, results in result.items():
            type = merge_type.split('_')[0]
            if type not in type_dict:
                type_dict[type] = []
            type_dict[type].extend(results)
        for type, results in type_dict.items():
            if type == 'Visualization':
                metric_scores = {
                    'F1': 0,
                    'EM': 0,
                    'ROUGE-L': 0,
                    'SacreBLEU': 0,
                }
                total = len(results)
                metric_scores['total'] = total
                ecr_1_acc = None
                ecr_1s = [result["parsed_result"].get('ecr_1', None)
                          for result in results]
                ecr_1_acc = ecr_1s.count(True) / total
                metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
                pass_1_acc = None
                parsed_prediction_results = []
                for result in results:
                    parsed_prediction = result["parsed_result"]['parsed_prediction']
                    if parsed_prediction == 'True':
                        parsed_prediction_results.append(True)
                    elif parsed_prediction == 'False':
                        parsed_prediction_results.append(False)
                    else:
                        parsed_prediction_results.append('None')
                pass_1_acc = parsed_prediction_results.count(True) / total
                metric_scores['Pass@1'] = round(pass_1_acc*100, 2)
            else:
                predictions = [result["parsed_result"]['parsed_prediction']
                               for result in results]
                references = [result['answer'] for result in results]
                metric_scores = qa_metric.compute(predictions, references)
                total = len(predictions)
                metric_scores['total'] = total
                if key.split('/')[1] == 'PoT':
                    ecr_1_acc = None
                    ecr_1s = [result["parsed_result"].get('ecr_1', None)
                              for result in results]
                    ecr_1_acc = ecr_1s.count(True) / total
                    metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
            parse_1s = [result["parsed_result"].get(
                'Parse@1', None) for result in results]
            metric_scores['Parse@1'] = round(
                parse_1s.count(True) / total * 100, 2)
            if key not in llm_eval_type_results:
                llm_eval_type_results[key] = {}
            llm_eval_type_results[key][type] = metric_scores
    write_json_to_file(llm_eval_type_results_path, llm_eval_type_results)

    llm_eval_type_csv_results = []
    for key, result in llm_eval_type_results.items():
        csv_result = {
            'model_name': key.split('/')[0],
            'prompt_type': key.split('/')[1]
        }
        for type, metric_scores in result.items():
            if type == 'Visualization':
                csv_result[type] = metric_scores['Pass@1']
            else:
                csv_result[type] = metric_scores[metric_name]
        llm_eval_type_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_type_csv_results)
    llm_eval_df.to_csv(llm_eval_type_results_csv_path, index=False, sep='\t')

def eval_by_overall(categoried_llm_inference_results, qa_metric, metric_name='ROUGE-L'):
    llm_eval_overall_results_path = f'{EVAL_RESULT_DIR}/llm_eval_overall_results.json'
    llm_eval_overall_results_csv_path = f'{EVAL_RESULT_DIR}/llm_eval_overall_results.csv'
    llm_eval_overall_results = {}
    for key, result in categoried_llm_inference_results.items():
        print(f'Processing {key}...')
        overall_results = []
        overall_wov_results = []
        overall_wv_results = []
        for merge_type, results in result.items():
            if merge_type == 'Visualization_ChartGeneration':
                overall_wv_results.extend(results)
            else:
                overall_wov_results.extend(results)
            overall_results.extend(results)
        metric_scores = {}
        total = len(overall_results)
        metric_scores['total'] = total

        wov_total = len(overall_wov_results)
        predictions = [result["parsed_result"]['parsed_prediction']
                       for result in overall_wov_results]
        references = [result['answer'] for result in overall_wov_results]
        wv_metric_scores = qa_metric.compute(predictions, references)
        rouge_l = wv_metric_scores['ROUGE-L']

        wv_total = len(overall_wv_results)
        parsed_predictions = [result["parsed_result"]['parsed_prediction']
                              for result in overall_wv_results]
        parsed_prediction_results = []
        for parsed_prediction in parsed_predictions:
            if parsed_prediction == 'True':
                parsed_prediction_results.append(True)
            elif parsed_prediction == 'False':
                parsed_prediction_results.append(False)
            else:
                parsed_prediction_results.append('None')
        if wv_total == 0:
            pass_1_acc = 0
        else:
            pass_1_acc = parsed_prediction_results.count(True) / wv_total * 100

        mix_metric = (rouge_l*wov_total + pass_1_acc*wv_total) / total
        metric_scores['MIX_Metric'] = round(mix_metric, 2)

        if key.split('/')[1] == 'PoT':
            ecr_1_acc = None
            ecr_1s = [result["parsed_result"].get('ecr_1', None)
                      for result in overall_results]
            ecr_1_acc = ecr_1s.count(True) / total
            metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)

        parse_1s = [result["parsed_result"].get(
            'Parse@1', None) for result in overall_results]
        metric_scores['Parse@1'] = round(
            parse_1s.count(True) / total * 100, 2)
        llm_eval_overall_results[key] = metric_scores
    write_json_to_file(llm_eval_overall_results_path, llm_eval_overall_results)

    llm_eval_overall_csv_results = []
    for key, metric_scores in llm_eval_overall_results.items():
        csv_result = {
            'model_name': key.split('/')[0],
            'prompt_type': key.split('/')[1]
        }
        csv_result['overall'] = metric_scores['MIX_Metric']
        csv_result['Parse@1'] = metric_scores['Parse@1']
        llm_eval_overall_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_overall_csv_results)
    llm_eval_df.to_csv(llm_eval_overall_results_csv_path,
                       index=False, sep='\t')


if __name__ == '__main__':
    exp_version = '20240730_hf'
    PARSED_RUSULT_DIR = f'{PROJ_ROOT}/experiment_results/{exp_version}/parsed_results'
    EVAL_RESULT_DIR = f'{PROJ_ROOT}/experiment_results/{exp_version}/evaluation_results'
    metric_name = 'ROUGE-L'
    eval_models = []

    qa_metric = QAMetric()

    merge_parsed_results_to_one_file()
    categoried_llm_inference_results = build_categoried_llm_inference_results()

    print('-'*10, 'Eval by subtype', '-'*10)
    eval_by_subtype(categoried_llm_inference_results, qa_metric, metric_name)
    print('-'*10, 'Eval by type', '-'*10)
    eval_by_type(categoried_llm_inference_results, qa_metric, metric_name)
    print('-'*10, 'Eval by overall', '-'*10)
    eval_by_overall(categoried_llm_inference_results, qa_metric, metric_name)
