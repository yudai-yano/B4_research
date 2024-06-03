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

from model_2 import create_model_emg
model = create_model_emg(activation='PReLU', optimizer='adamax')

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
    print(f'mse: {score[0]}, mae: {score[1]}')
    r2 = r2_score(y_test, pred)
    print(f'r2: {r2}')
    print(y_test)
    print('================')
    print(pred)
    mse.append(score[0])
    mae.append(score[1])
    r2_s.append(r2)
   
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 18 19 20 21 22 23 24 25 26]|[15 16 17]

# print(result)
print(f'mse: {mean(mse)}, mae: {mean(mae)}, r2_score: {mean(r2_s)}')

# [ 0  1  2  3  4  5  6  7  8  9 10 11 15 16 17 18 19 20 21 22 23 24 25 26]|[12 13 14]

#     rectus_femoris  vastus_lateralis  vastus_medialis  biceps_femoris
# 12        0.608367          0.859776         0.568058        0.426599
# 13        0.665598          0.476261         0.598139        0.404653
# 14        0.616557          0.651332         0.723506        0.401460
# ================
# 12        0.5647693         0.46591502       0.55059254      0.5191043
# 13        0.56190056        0.46858534       0.55565405      0.51007915
# 14        0.6204234         0.48594552       0.5878005       0.5545813

# [ 0  1  2  3  4  5  6  7  8  9 10 11 15 16 17 18 19 20 21 22 23 24 25 26]|[12 13 14]
# mse: 0.0009892683010548353, mae: 0.028721055015921593
#     rectus_femoris
# 12        0.608367
# 13        0.665598
# 14        0.616557
# ================
# [[0.59747654]
#  [0.62511235]
#  [0.6513442 ]]

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 21 22 23 24 25 26]|[18 19 20]
# mse: 0.0016387969953939319, mae: 0.03172573447227478
#     vastus_lateralis
# 18          0.289787
# 19          0.525754
# 20          0.399735
# ================
# [[0.3175526]
#  [0.461444 ]
#  [0.4028355]]

# [ 0  1  2  3  4  5  6  7  8  9 10 11 15 16 17 18 19 20 21 22 23 24 25 26]|[12 13 14]
# mse: 0.0021844524890184402, mae: 0.03658837080001831
#     vastus_medialis
# 12         0.568058
# 13         0.598139
# 14         0.723506
# ================
# [[0.53400034]
#  [0.5958349 ]
#  [0.65010244]]

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 18 19 20 21 22 23 24 25 26]|[15 16 17]
# 1/1 [==============================] - 0s 13ms/step
# mse: 0.0027211138512939215, mae: 0.05170176550745964
#     biceps_femoris
# 15        0.397829
# 16        0.428275
# 17        0.499421
# ================
# [[0.44739816]
#  [0.47275767]
#  [0.56047523]]