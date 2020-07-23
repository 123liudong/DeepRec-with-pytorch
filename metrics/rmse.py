import numpy as np


def gen_rmse(predict_dict, target_dict):
    '''
    cal rmse
    :param predict_dict:
    :param target_dict: be like parameter predict_dict
    :return: float value
    '''
    item_nb = 0
    for uid in predict_dict.keys():
        item_nb +=