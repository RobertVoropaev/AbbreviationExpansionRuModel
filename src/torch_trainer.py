import os
import sys

import pandas as pd
import numpy as np
import random

from matplotlib import pyplot as plt

from tqdm import tqdm

from pandarallel import pandarallel

import pymorphy2
import nltk
import pickle

from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences

SEED = 1
def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True
init_random_seed(SEED)

def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))

class PytorchLearner:
    def __init__(self, 
                 model, 
                 loss_function, 
                 optimizer = torch.optim.Adam,
                 train_dataset,
                 val_dataset,
                 batch_size: int = 64,
                 shuffle_train: bool = True,
                 lr: float = 1e-4
                 device: str = None,
                 ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)

        self.model = model
        self.model.to(self.device)

        self.optimizer = optimizer(model.parameters(), lr=lr)
        
        self.loss_function = loss_function

        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_train,
        )

        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )
        
        self.liveplot = PlotLosses(groups={"log loss": ["train", "val"]})
        
    def fit_epoch(self, steps_per_epoch = None):
        self.model.train()
        loss_sum = 0
        batch_n = 0
        
        for batch_i, (batch_x, batch_y) in enumerate(self.train_dataloader):
            if steps_per_epoch is not None:
                if batch_i > steps_per_epoch:
                    break
                    
            batch_x = copy_data_to_device(batch_x, device)
            batch_y = copy_data_to_device(batch_y, device)
            
            pred = self.model(batch_x)
            loss = self.loss_function(pred, batch_y)

            model.zero_grad() 
            loss.backward() 
            optimizer.step()

            loss_sum += float(loss)
            batch_n += 1
            
        return loss_sum / batch_n
    
    def eval(self, steps_per_epoch = None):
        self.model.eval()
        loss_sum = 0
        batch_n = 0
        
        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(self.val_dataloader):
                if steps_per_epoch is not None:
                    if batch_i > steps_per_epoch:
                        break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = self.model(batch_x)
                loss = self.loss_function(pred, batch_y)

                loss_sum += float(loss)
                batch_n += 1
                
        return loss_sum / batch_n
                
    def fit(self, 
            epoch: int = 10, 
            step_per_epoch_train: int = 100,
            step_per_epoch_val: int = 100, 
            early_stopping_patience: int = 10):
        
    self.best_val_loss = float('inf')
    self.best_epoch_i = 0
    self.best_model = copy.deepcopy(self.model)
    for epoch_i in range(epoch):
        try:
            train_loss = self.fit_epoch(step_per_epoch_train)
            val_loss = self.eval(step_per_epoch_val)

            if val_loss < best_val_loss:
                self.best_epoch_i = epoch_i
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(model)
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    early_stopping_patience))
                break
                
            self.liveplot.update({'train': loss_sum / batch_n, 'val': val_loss_sum / val_batch_n})
            self.liveplot.draw()
            
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
            
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break