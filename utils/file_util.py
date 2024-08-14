# -*- coding: UTF-8 -*-
import os
import json
import pickle
import itertools
import hashlib
import os
import pandas as pd


def iter_file_from_dir(folder_path, ext=''):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(ext):
            yield file_path


def walk_file_from_dir(folder_path, ext=''):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(ext):
                yield os.path.join(root, file)


def iter_line_from_file(file_path, func=None):
    with open(file_path, 'r') as f:
        for line in f:
            if func is not None:
                yield func(line.strip())
            else:
                yield line.strip()


def read_json_file(path, filter_func=None):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                if filter_func is not None:
                    json_data = list(filter(filter_func, json_data))
                return json_data
            except Exception as e:
                f.seek(0)
                lines = f.readlines()
                json_list = [json.loads(line.strip(
                )) for line in lines if filter_func is None or filter_func(json.loads(line.strip()))]
                return json_list
    else:
        return None


def write_json_to_file(path: str, data: dict, is_json_line: bool = False) -> None:
    valid_path(path)
    with open(path, 'w', encoding='utf-8') as f:
        if is_json_line:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))


def save_as_csv(path: str, data: list, sep: str = '\t'):
    valid_path(path)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding='utf-8', sep=sep)


def save_variable_to_bin_file(path, var):
    with open(path, 'wb') as f:
        pickle.dump(var, f)


def load_variable_from_bin_file(path):
    with open(path, 'rb') as f:
        var = pickle.load(f)
    return var


def batch_iterator(iterator, batch_size):
    iterator = iter(iterator)
    while True:
        res = tuple(itertools.islice(iterator, batch_size))
        if not res:
            break
        yield res


def concat_iterators(*iterators):
    return itertools.chain(*iterators)


def valid_path(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)