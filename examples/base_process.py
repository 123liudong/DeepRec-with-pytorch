from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

from models.LR import LR
from utils.dataset.movielens import ML1m, ML20m


def train(model, dataloader, opt, criterion, device, log_interval=100):
    '''
    训练一次模型
    :param model: 模型
    :param dataloader: 数据加载器
    :param opt: 优化器
    :param criterion: 损失函数
    :param device: 运行设备
    :param log_interval: 日志打印间隔
    :return:
    '''
    model.train()
    total_loss = 0
    tk = tqdm(dataloader, smoothing=0, mininterval=1.0)
    for i, (features, target) in enumerate(tk):
        features, target = features.to(device), target.to(device)
        out = model(features)
        loss = criterion(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.cpu().item()
        if (i+1) % log_interval == 0:
            tk.set_postfix(loss=total_loss/log_interval)
            total_loss = 0


def test(model, dataloader, device):
    '''
    简单统计
    :param model:
    :param dataloader:
    :param device:
    :return:
    '''
    pass


def choose_model(model_name, dataset):
    '''
    根据模型名称制造模型
    :param model_name: 模型名称简写
    :return:
    '''
    if model_name == 'lr':
        model = LR(feature_dims=dataset.feature_dims)
    return model


def choose_dataset(dataset_name, dataset_path):
    if dataset_name == 'ML1m':
        dataset = ML1m(dataset_path)
    elif dataset_name == 'ML20m':
        dataset = ML20m(dataset_path)
    return dataset


def main(model_name, dataset_name, dataset_path, epoches, batch_size,
         lr, device, save_path):
    '''
    :param model_name:
    :param dataset_name: 数据集的名称
    :param dataset_path: 数据集的路径
    :param epoches:
    :param batch_size:
    :param lr:
    :param device:
    :param save_path:
    :return:
    '''
    device = torch.device(device)
    # 加载数据集
    dataset = choose_dataset(dataset_name, dataset_path)
    # 划分测试集、验证集、训练集
    train_length, valid_length = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    dataloader_train = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    dataloader_valid = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)
    # 定义模型相关内容
    model = choose_model(model_name=model_name, dataset=dataset).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = torch.nn.BCELoss()
    for e in range(epoches):
        train(model, dataloader_train, opt, criterion=loss_f, device=device, log_interval=100)