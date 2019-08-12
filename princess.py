import numpy as np
import matplotlib.pyplot as plt

import chainer.optimizaers as Opt
import chainer.functions as F
import chainer.links as L

from chainer import Variable, Chain, config, cuda

def data_divide(Dtrain, D, xdata, tdata)
    index = np.random.permutation(range(D))
    xtrain = xdata[index[0:Dtrain], :]
    ttrain = tdata[index[0:Dtrain]]
    xtest = xdata[index[Dtrain:D], :]
    ttest = tdata[index[Dtrain]]
    return xtrain, xtest, ttrain, ttest

def plot_result2(result1, result2, title, xlabel, ylabel, ymin=0.0, ymax=1.0):
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

def learning_classification(model, optNN, data, result, T=10):
    for time in range(T):
        config.train = True
        optNN.target.cleargrads()
        ytrain = model(data[0])
        loss_train = F.softmax_cross_entropy(ytrain, data[2])
        acc_train = F.accuracy(ytrain, data[2])
        loss_train.backward()
        optNN.update()

        config.train = False
        ytest = model(data[1])
        loss_test = F.softmax_cross_entropy(ytest, data[3])
        acc_test = F.accuracy(ytest, data[3])
        result[0].append(cuda.to_cpu(loss_train.data))
        result[1].append(cuda.to_cpu(loss_test.data))
        result[2].append(cuda.to_cpu(acc_train.data))
        result[3].append(cuda.to_cpu(acc_test.data))