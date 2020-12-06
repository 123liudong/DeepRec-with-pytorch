from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, roc_auc_score
import numpy as np


def get_metrics(predict_vec, target_vec):
    predict_vec, target_vec = np.asarray(predict_vec, dtype=np.float), np.asarray(target_vec, dtype=np.int)
    value_rmse = mse(predict_vec, target_vec, squared=False)
    value_mae = mae(predict_vec, target_vec)
    value_auc = roc_auc_score(target_vec, predict_vec)
    return value_rmse, value_mae, value_auc