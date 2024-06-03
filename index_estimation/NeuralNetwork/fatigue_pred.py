import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Dense, Dropout
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.metrics import r2_score
from statistics import mean

import config
import json

""" ----------------- data processing --------------------"""
# configの読込み

data_cfg = config.data_processing
pred_cfg = config.fatigue_pred
explonatory_variable = pred_cfg["explonatory_variable"]
objective_variable = pred_cfg["objective_variable"]

build_model = pred_cfg["build_model"]
model_fit = pred_cfg["model_fit"]


columns = copy.copy(explonatory_variable)
columns.extend(objective_variable)
data = pd.read_csv(data_cfg["input"], index_col=None, usecols=columns)


# X = data.loc[:, explonatory_variable]
# y = data.loc[:, objective_variable]


# 正規化
def standardlize(x, columns):
    x_standardlized = []
    for i in x:        
        data = x.loc[:, i]
        max_value = data.describe()[7]
        min_value = data.describe()[3]

        standard = []

        for i in data:
            standardization = (i - min_value) / (max_value - min_value)
            standard.append(standardization)

        x_standardlized.append(standard)

    # 転置しDataframeに戻す
    x_standardlized_t = []
    for i in range(len(x_standardlized[0])):
        tmp = []
        for v in x_standardlized:
            tmp.append(v[i])
        x_standardlized_t.append(tmp)

    x = pd.DataFrame(x_standardlized_t, columns=columns)
    return x


from sklearn.model_selection import LeaveOneGroupOut

X = data.loc[:, explonatory_variable]
X = standardlize(X, explonatory_variable)

y = data.loc[:, objective_variable]

y = np_utils.to_categorical(y)
y = pd.DataFrame(y)

from model_2 import classify_model

model = classify_model(activation='relu', optimizer='adamax')

groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, ]

logo = LeaveOneGroupOut()
result = {
    'loss': [],
    'accuracy': []
    }

loss = result["loss"]
accuracy = result["accuracy"]


for train, test in logo.split(X, y, groups):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]

    print(f'{train}|{test}')
    
    """ --------------- learning ---------------------"""

    train_history = model.fit(X_train, y_train, batch_size=model_fit["batch_size"], epochs=model_fit["epochs"], verbose=0)

    plt.plot(train_history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.xlim(0, model_fit["epochs"])
    # plt.ylim(-0.01, 1)
    # plt.show()

    pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test, verbose=0)
    r2 = r2_score(y_test, pred)
    print(f'loss: {score[0]}, accuracy: {score[1]}')
    print(y_test)
    print('=========================')
    max_value = np.max(pred, axis=1)
    res = []
    for i in range(len(pred)):
        b = np.where(pred[i]==max_value[i], 1.0, 0.0)
        b = b.tolist()
        res.append(b)
    print(pd.DataFrame(res))
    print(pred)
    loss.append(score[0])
    accuracy.append(score[1])
    

print(f'loss: {mean(loss)}, accuracy: {mean(accuracy)}')
# [ 0  1  2  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]|[3 4 5]
#     0    1    2    3
# 3  0.0  1.0  0.0  0.0
# 4  0.0  1.0  0.0  0.0
# 5  0.0  1.0  0.0  0.0
# =========================
#      0    1    2    3
# 0  0.0  1.0  0.0  0.0
# 1  0.0  1.0  0.0  0.0
# 2  0.0  1.0  0.0  0.0