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

subject = rf"C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result\result_final_1"

# 保存したモデル構造の読み込み
model2 = model_from_json(open(subject + "_model.json", 'r').read())

# 保存した学習済みの重みを読み込み
model2.load_weights(subject + "_weight.hdf5")

l = model2.layers[8]

display = model2.get_weight()

print(display)