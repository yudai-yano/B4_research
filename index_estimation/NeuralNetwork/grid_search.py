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
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

import config
import json

""" ----------------- data processing --------------------"""
# configの読込み

data_cfg = config.data_processing
pred_cfg = config.emg_pred
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

X_train = X.iloc[pred_cfg["train_row"], ]
X_test = X.iloc[pred_cfg["test_row"], ]
y_train = y.iloc[pred_cfg["train_row"], ]
y_test = y.iloc[pred_cfg["test_row"]]

print(X_train)
from model_2 import classify_model

# グリッドサーチ対象のハイパーパラメータを準備
activation = ['relu', 'sigmoid']
optimizer = ['adam', 'adadelta', 'adamax', 'adagrid']
nb_epoch = [1000, 1500, 2000, 2500]
batch_size = [1, 2, 4, 8]

param_grid = dict(activation=activation, optimizer=optimizer, nb_epoch=nb_epoch, batch_size=batch_size)
model = KerasRegressor(build_fn=classify_model, verbose=0)


# グリッドサーチの実行
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)


print(grid_result.best_params_)

""" --------------- learning ---------------------"""

# train_history = model.fit(X_train, y_train, batch_size=model_fit["batch_size"], epochs=model_fit["epochs"], validation_split=model_fit["validation_split"], verbose=0)

# plt.plot(train_history.history['loss'])
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.xlim(0, model_fit["epochs"])
# # plt.ylim(-0.01, 1)
# plt.show()

# pred = model.predict(X_test)
# r2 = r2_score(y_test, pred)
# score = model.evaluate(X_test, y_test, verbose=0)
# print(f'mse: {score[0]}, mae: {score[1]}, r2: {r2}')

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 18 19 20 21 22 23 24 25 26]|[15 16 17]