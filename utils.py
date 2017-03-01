import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
def add_gaussian_noise(X_train, mean, std):
    if std == 0:
        return X_train
    else:
        noise = np.random.normal(loc=mean,scale=std,size = X_train.size())
    X_train.add_(torch.Tensor(noise))

    clipped_X = torch.clamp(X_train, min = 0., max = 1.)
    return clipped_X

def class_report(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    test_err = []
    for i in range(10):
        temp = 1 -  C[i,i] / sum(C[i,:])
        test_err.append(temp)
    return C, test_err


def add_label_noise(y_train, ratio= .9):
    n_labels = y_train.size()[0]
    n_to_rand = int(n_labels * ratio)
    rand_labels = np.random.randint(0,10,n_to_rand)
    idx_to_rand = np.random.choice(range(n_labels), n_to_rand)
    y_train_data = y_train.numpy().copy()
    y_train_data[idx_to_rand] = rand_labels
    return torch.LongTensor(y_train_data)

def permute_label(y_train, ratio = .9):
    perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])

    n_labels = y_train.size()[0]
    n_to_rand = int(n_labels * ratio)
    idx_to_rand = np.random.choice(n_labels, n_to_rand, replace=False)
    y_train_data = y_train.numpy().copy()
    noise = perm[y_train_data]
    y_train_data[idx_to_rand] = noise[idx_to_rand]
    return torch.LongTensor(y_train_data)

def random_corrupt_label(y_train, p = .9):
    n_labels = y_train.size()[0]
    y_train_data = y_train.numpy().copy()
    for i in range(n_labels):
        temp = [0,1,2,3,4,5,6,7,8,9]
        temp.remove(int(y_train_data[i]))
        y_train_data[i] = y_train_data[i] if np.random.rand()>p else np.random.choice(temp,1)

    return torch.LongTensor(y_train_data)