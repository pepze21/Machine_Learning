import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Funciton

def machine(x, w, b):
    y_hat = (w*x) + b # x : np array -> y_hat : np array
    return y_hat

def gradient(x, y, w, b):
    y_hat = machine(x, w, b)
    # Metric = sigma ((y - y_hat)**2) / n
    # Metric = np.mean((y - y_hat)**2)
    dw = np.mean((y - y_hat) * (-2 * x)) # dw = round Metric / round w
    db = np.mean((y - y_hat) * (-2)) # db = round Metric / round b
    return dw, db

def learning(x, y, w, b, step):
    dw, db = gradient(x, y, w, b)
    uw = w - step * dw
    ub = b - step * db
    return uw, ub

## Variables

w = 2
b = 3
step = 0.05 # hyperparameter


url = 'https://raw.githubusercontent.com/rusita-ai/pyData/master/testData.csv'
DATA = pd.read_csv(url)
DATA.head()
DATA.info()
plt.scatter(DATA.inputs, DATA.outputs, s=0.5) # s : size of markers
plt.show()

# learn 1500 times by Gradient descent
for i in range(0, 1500):
    uw, ub = learning(DATA.inputs, DATA.outputs, w, b ,step)
    w = uw
    b= ub

print('learned_w is ', '%.3f' % w)
print('learned_b is ', '%.3f' % b)

X = np.linspace(0, 1, 100)
Y = (w * X) + b
plt.scatter(DATA.inputs, DATA.outputs, s=0.5)
plt.plot(X, Y, '-r', linewidth=1.5)
plt.show()

