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
from keras import initializers
import time

import config
import json

""" ----------------- data processing --------------------"""
# configの読込み

subject = "result_kousakennsyou"

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
#model = create_model.lac_create_model(activation='relu', optimizer='adamax')

groups = []
for i in range(len(X)):
    groups.append(i)



#groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
#groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
#groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]
#groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]


logo = LeaveOneGroupOut()

processing_time = []

result = {
    'mse': [],
    'mae': [],
    #'r2_score': []
}

estimate = {
    '1st_set': [],
    #'2nd_set': [],
    #'3rd_set': []
}

standard_data = {
    '1st_set': [],
    #'2nd_set': [],
    #'3rd_set': []
}

mse = result["mse"]
mae = result["mae"]
#r2_s = result["r2_score"]

E_1 = estimate["1st_set"]
#E_2 = estimate["2nd_set"]
#E_3 = estimate["3rd_set"]

S_1 = standard_data["1st_set"]
#S_2 = standard_data["2nd_set"]
#S_3 = standard_data["3rd_set"]

for train, test in logo.split(X, y, groups):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]
    
    model = create_model.lac_create_model(X_train.shape[1],activation='relu', optimizer='adamax')
    
    print(f'{train}|{test}')
    
    start = time.time()
    """ --------------- learning ---------------------"""

    train_history = model.fit(X_train, y_train, batch_size=model_fit["batch_size"], epochs=model_fit["epochs"], verbose=0)

    end = time.time()
    
    process_time = (end - start) * 10 ** 3
    processing_time.append(process_time)
    
    plt.plot(train_history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.xlim(0, model_fit["epochs"])
    # plt.ylim(-0.01, 1)
    # plt.show()

    pred = model.predict(X_test)
    #r2 = r2_score(y_test, pred)
    score = model.evaluate(X_test, y_test, verbose=0)
    #print(f'mse: {score[0]}, mae: {score[1]}, r2: {r2}')
    print(f'mse: {score[0]}, mae: {score[1]}')
    print(y_test)
    print(pred)
    mse.append(score[0])
    mae.append(score[1])
    #r2_s.append(r2)
    E_1.append(pred[0][0])
    #E_2.append(pred[1][0])
    #E_3.append(pred[2][0])
    S_1.append(y_test.iloc[0][0])
    #S_2.append(y_test.iloc[1][0])
    #S_3.append(y_test.iloc[2][0])
    model.reset_states()
   
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 18 19 20 21 22 23 24 25 26]|[15 16 17]

all_variations = []
residual_variation = []
S_ave = sum(S_1)/len(S_1)
for i in range(len(E_1)):
    e_s = (E_1[i] - S_1[i])**2
    s_a = (S_1[i] - S_ave)**2
    residual_variation.append(e_s)
    all_variations.append(s_a)

sum_residual_variation = sum(residual_variation)
sum_all_variations = sum(all_variations)
R2 = 1 - (sum_residual_variation/sum_all_variations)

# print(result)
#print(f'mse: {mean(mse)}, mae: {mean(mae)}, r2_score: {mean(r2_s)}')
print(f'mse: {mean(mse)}, mae: {mean(mae)}, R^2: {R2}')

#print("ALL result",f'mse: {mse}, mae: {mae}, r2_score: {r2_s}')
print("ALL result",f'mse: {mse}, mae: {mae}')

result = pd.DataFrame(result)
estimate = pd.DataFrame(estimate)
standard_data = pd.DataFrame(standard_data)
ave_time = sum(processing_time)/len(processing_time)
#result.to_csv(rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\NN_result_{subject}.csv')
estimate.to_csv(rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\NN_estimate_{subject}.csv')
standard_data.to_csv(rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\NN_standard_{subject}.csv')
print(ave_time)