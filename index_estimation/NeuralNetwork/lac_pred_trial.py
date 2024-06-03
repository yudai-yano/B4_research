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
from sklearn.metrics import r2_score
from statistics import mean

import config
import json

""" ----------------- data processing --------------------"""
# configの読込み

data_cfg = config.data_processing
pred_cfg = config.lac_pred
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

data = standardlize(data, columns)

X = data.loc[:, explonatory_variable]
y = data.loc[:, objective_variable]

import create_model
model = create_model.lac_create_model(activation='relu', optimizer='adamax')

groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]

logo = LeaveOneGroupOut()

result = {
    'mse': [],
    'mae': [],
    'r2_score': []
}

mse = result["mse"]
mae = result["mae"]
r2_s = result["r2_score"]

'''
for train, test in logo.split(X, y, groups):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]

    print(f'{train}|{test}')
    '''
""" --------------- learning ---------------------"""

train_history = model.fit(X, y, batch_size=model_fit["batch_size"], epochs=model_fit["epochs"], verbose=0)

plt.plot(train_history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.xlim(0, model_fit["epochs"])
# plt.ylim(-0.01, 1)
# plt.show()

pred = model.predict(X)
r2 = r2_score(y, pred)
score = model.evaluate(X, y, verbose=0)
print(f'mse: {score[0]}, mae: {score[1]}, r2: {r2}')
print(y)
print(pred)
mse.append(score[0])
mae.append(score[1])
r2_s.append(r2)
   
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 18 19 20 21 22 23 24 25 26]|[15 16 17]

# print(result)
print(f'mse: {mean(mse)}, mae: {mean(mae)}, r2_score: {mean(r2_s)}')