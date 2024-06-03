import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential,model_from_json
from keras.models import save_model
from keras.layers import Dense, Dropout
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.utils.vis_utils import plot_model
from sklearn.metrics import r2_score
from statistics import mean
from keras import initializers
import time

import config
import json

""" ----------------- data processing --------------------"""
# configの読込み
subject = rf"C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\result_kousatu_RIR3_nonmax"
subject2 = "result_kousatu_RIR3_nonmax"

data_cfg = config.data_processing
pred_cfg = config.lac_pred
input_data = pred_cfg["columns"]
explonatory_variable = pred_cfg["explonatory_variable"]
objective_variable = pred_cfg["objective_variable"]

index_remaining = []

for index in input_data:
    if not index in explonatory_variable:
        if not index in objective_variable:
            index_remaining.append(index)

build_model = pred_cfg["build_model"]
model_fit = pred_cfg["model_fit"]


columns = copy.copy(explonatory_variable)
columns.extend(objective_variable)
data_input = pd.read_csv(data_cfg["input"], index_col=None, usecols=input_data)


# X = data.loc[:, explonatory_variable]
# y = data.loc[:, objective_variable]


#本家
# 正規化
def standardlize(x, columns):
    x_standardlized = []
    for i in columns:
        data = x.loc[:, i]
        max_value = data.describe()[7]
        #min_value = data.describe()[3]

        standard = []

        for i in data:
            standardization = i / max_value
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
'''
def standardlize(x, columns):
    x_standardlized = []
    for i in x:        
        data = x.loc[:, i]
        average_value = data.describe()[1]
        standard_value = data.describe()[2]

        standard = []

        for i in data:
            standardization = (i - average_value) / standard_value
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
'''

from sklearn.model_selection import LeaveOneGroupOut

data = standardlize(data_input, columns)

X = data.loc[:, explonatory_variable]
if not index_remaining == []:
    for index in index_remaining:
        X[index] = data_input.loc[:,index]
y = data.loc[:, objective_variable]

import create_model

x_columns = X.columns.tolist()
y_columns = y.columns.tolist()

#test_list = [19, 49, 42, 73]
test_list = [12,13,14]
x_test = pd.DataFrame(columns=x_columns)
y_test = pd.DataFrame(columns=y_columns)
j = 0

for i in test_list:
    x_test.loc[j] = X.iloc[i,:]
    y_test.loc[j] = y.iloc[i,:]
    X = X.drop(i, axis=0)
    X = X.loc[:, ~X.isnull().all()]
    y = y.drop(i, axis=0)
    y = y.loc[:, ~y.isnull().all()]
    j += 1

logo = LeaveOneGroupOut()

model = create_model.lac_create_model_charenge(activation='relu', optimizer='adamax')

processing_time = []

'''-----トレーニングの実行-----'''
train_result = model.fit(X, y, batch_size=model_fit["batch_size"], epochs=model_fit["epochs"], verbose=0)
'''-----モデルの保存-----'''
open(subject + "_model.json","w").write(model.to_json())
'''-----重みの保存-----'''
model.save_weights(subject + "_weight.hdf5")

# 保存したモデル構造の読み込み
model2 = model_from_json(open(subject + "_model.json", 'r').read())

# 保存した学習済みの重みを読み込み
model2.load_weights(subject + "_weight.hdf5")

# 検証用データをモデルに入力し、出力（予測値）を取り出す
predict_y = model2.predict(x_test)
np.savetxt(rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\NN_estimate_{subject2}.csv',predict_y,delimiter=',')
y_test.to_csv(rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\NN_standard_{subject2}.csv')

#print(x_test)
#print(y_test)