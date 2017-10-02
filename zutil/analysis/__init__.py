
from get_case_dict import get_case_dict
from get_case_dict import case_dict
from get_case_dict import validation_dict

from data import init as data_init
from data import data_dir
from data import ref_data_dir

import os


def is_remote():
    return data.remote_data


def get_ref(filename):
    return os.path.join(data.ref_data_dir, filename)


def get_data(filename):
    return os.path.join(data.data_dir, filename)
