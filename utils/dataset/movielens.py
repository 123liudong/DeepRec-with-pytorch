import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ML1m(Dataset):
    '''
    加载movielens 1m的数据
    '''
    def __init__(self, data_path, sep='::', header=None):
        '''
        :param data_path: 评分文件
        :param sep: 切割符
        :param header: 是否有标题
        '''
        # 只需要用户、项目和评分，不需要时间戳
        data = pd.read_csv(data_path, sep=sep, header=header).to_numpy()[:, :3]
        # 对用户和项目的Id进行处理，因为所有的id都是从1开始，所有这里减去
        self.features = data[:, :2].astype(np.long)-1
        self.targets = self.__process_score(data[:, -1]).astype(np.float)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return self.features.shape[0]

    def __process_score(self, score):
        '''
        分数小于等于3的就是0，大于3的就是1
        :param score:
        :return:
        '''
        score[score<=3] = 0
        score[score>3] = 1
        return score


class ML20m(ML1m):
    '''
    加载movielens 20m的数据
    '''
    def __init__(self, data_path, sep=',', header='infer'):
        super().__init__(data_path, sep=sep, header=header)




# ml1m = ML1m('E:\liudong\\feature_cross\借鉴的代码\pytorch-fm-master\data\\ml-1m\\ratings.dat')
# ml20m = ML20m('C:\\Users\\dongliu\\Downloads\\ml-25m\\ml-25m\\ratings.csv')
# print(len(ml20m))