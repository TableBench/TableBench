from hashlib import md5
from datetime import datetime
import os
import pprint
from json.encoder import JSONEncoder
import json


def generate_four_digit_code(string):
    md5_hash = md5(string.encode()).hexdigest()
    four_digit_code = md5_hash[:4]

    return four_digit_code


def generate_md5_hash(string):
    md5_hash = md5(string.encode()).hexdigest()
    return md5_hash


def get_date_suffix():
    return datetime.now().strftime('%m%d')


def get_env_vars(var_name):
    value = os.environ.get(var_name)
    if var_name in os.environ:
        value = os.environ[var_name]
    else:
        value = None
    return value


def pprint_dict(dict):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict)


def get_enum_from_val(Enum, val):
    members = Enum.__members__
    for member in members:
        member_val = members[member].value
        if member_val == val:
            return members[member]
    return None


class DictObjEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


def dict_obj_to_json(obj):
    return json.dumps(obj, cls=DictObjEncoder)
