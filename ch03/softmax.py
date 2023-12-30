import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def softmax_dash(x):
    c = np.max(x)
    exp_x = np.exp(x - c) # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

x = np.array([1010, 1000, 990])
print(softmax(x)) # [ nan  nan  nan] オーバーフロー
print(softmax_dash(x)) # [  9.99954600e-01   4.53978686e-05   2.06106005e-09]