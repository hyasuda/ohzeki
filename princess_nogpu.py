import numpy as np
import matplotlib.pyplot as plt

import chainer.optimizers as Opt
import chainer.functions as F
import chainer.links as L

from chainer import Variable, Chain, config

def data_divide(Dtrain, D, xdata, tdata):
    index = np.random.permutation(range(D))
    xtrain = xdata[index[0:Dtrain],:]
    ttrain = tdata[index[0:Dtrain]]
    xtest = xdata[index[Dtrain:D], :]
    ttest = tdata[index[Dtrain:D]]
    return xtrain, xtest, ttrain, ttest

def plot_result2(result1, result2, title, xlabel, ylabel, ymin = 0.0, ymax = 1.0):
    Tall = len(result1)
    plt.figure(figsize=(8,6))
    plt.plot(range(Tall), result1)
    plt.plot(range(Tall), result2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, Tall])
    plt.ylim([ymin, ymax])
    plt.show()

def learning_classification(model, optNN, data, result, T=50):
    for time in range(T):
        config.train = True
        optNN.target.zerograds()
        ytrain = model(data[0])
        loss_train = F.softmax_cross_entropy(ytrain, data[2])
        acc_train = F.accuracy(ytrain, data[2])
        loss_train.backward()
        optNN.update()
        # バッチ規格化をはじめ, いくつかのものはそのまま利用してしまうと, 
        # 訓練データを利用する際とテストデータを利用する際において違う結果を招くことがある.
        config.train = False
        ytest = model(data[1])
        loss_test = F.softmax_cross_entropy(ytest, data[3])
        acc_test = F.accuracy(ytest, data[3])
        result[0].append(loss_train.data)
        result[1].append(loss_test.data)
        result[2].append(acc_train.data)
        result[3].append(acc_test.data)

# 回帰の関数の定義
def learning_regression(model, optNN, data, result, T=10):
    for time in range(T):
        config.train = True
        optNN.target.cleargrads()
        ytrain = model(data[0])
        loss_train = F.mean_squared_error(ytrain, data[2])
        loss_train.backward()
        optNN.update()
        
        config.train = False
        ytest = model(data[1])
        loss_test = F.mean_squared_error(ytest, data[3])
        result[0].append(loss_train.data)
        result[1].append(loss_test.data)