import numpy as np
import pandas as pd
import pickle
import os

# Functions
def softmax(x, axis=None):
    """
    Stable softmax function
    :param x: input data
    :param axis: the dimension to sum up.
    :return: softmax result
    """
    exps = np.exp(x - np.max(x))
    ax_sum = np.expand_dims(np.sum(exps, axis=axis), axis)
    return exps / ax_sum

def cross_entropy(p, y):
    m = y.shape[0]
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss

# Derivatives
def d_cross_entropy(x, y):
    """
    source: https://deepnotes.io/softmax-crossentropy
    :param x: input x
    :param y: label y
    :return: derivative of cross entropy loss
    """
    m = y.shape[0]
    grad = softmax(x, axis=1)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad

def d_abs(x):
    mask = (x >= 0) * 1.0
    mask2 = (x < 0) * -1.0
    return mask + mask2

# Optimizer
class GD_Optimizer(object):
    def __init__(self, params, data_x, label_y, lr=0.01, l1_weight=1e-4):
        self.lr = lr
        self.params = params
        self.data_x = data_x
        self.label_y = label_y
        self.l1_weight = l1_weight
        self.grad_ce = np.zeros(shape=(self.data_x.shape[0], self.params.shape[1])) # TODO: need to check syntax
        self.grad_params = np.zeros(shape=self.params.shape)    # gradient of 'w' w.r.t y=wx+b
        self.grad_l1 = np.zeros(shape=self.params.shape)

    def backward(self):
        # grad_ce shape: num_of_records * num_of_classes
        self.grad_ce += d_cross_entropy(np.matmul(self.data_x, self.params), self.label_y)
        # avg_grad_ce shape: 1 * num_of_classes
        avg_grad_ce = np.asmatrix(np.average(self.grad_ce, axis=0))
        # avg_grad_wx shape must be 1 * num_of_features
        avg_grad_wx = np.asmatrix(np.average(self.data_x, axis=0))

        self.grad_params += np.matmul(np.transpose(avg_grad_wx), avg_grad_ce)   # need to zero grad after each update
        self.grad_l1 += d_abs(self.params) * self.l1_weight

    def zero_grad(self):
        self.grad_ce = np.zeros(shape=(self.data_x.shape[0], self.params.shape[1]))
        self.grad_l1 = np.zeros(shape=self.params.shape)
        self.grad_params = np.zeros(shape=self.params.shape)

    def step(self):
        assert(self.grad_params.shape == self.grad_l1.shape)
        grad = self.grad_params + self.grad_l1
        self.params -= self.lr * grad

def save_model(model, path='Model.pkl'):
    pickle.dump(model, open(path, 'wb'))

def load_model(path='Model.pkl'):
    return pickle.load(open(path, 'rb'))

def save_weight_bias(weight, bias, columns, num_class, path='output/weight_bias.csv'):
    """
    save weights and bias into csv file
    :param weight: weight list
    :param bias: bias list
    :param columns: column names list for csv header
    :param num_class: number of classes
    :return: None
    """
    indexes = []
    for i in range(num_class):
        s = 'class ' + str(i + 1)
        indexes.append(s)
    df = pd.DataFrame(weight, columns=columns, index=indexes)
    df['Bias'] = bias
    # print(df)
    # print(df.mean(axis=0))
    df.loc['Average'] = df.mean(axis=0)
    df.to_csv(path, sep=',', encoding='utf-8')

def save_report_to_csv(data, index, column, path, sep=','):
    df = None
    if os.path.exists(path) is True:
        df = pd.read_csv(path, sep=sep, index_col=0)
    if df is None:
        df = pd.DataFrame(data, index=index, columns=column)
    else:
        tmp_df = pd.DataFrame(data, index=index, columns=column)
        df = df.join(tmp_df)
    df.to_csv(path, sep=sep, encoding='utf-8')