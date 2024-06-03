import pandas as pd
import PySimpleGUI as sg
import datetime
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import scipy.stats as stats


def move_ave(vec, window_len):
    #strictly it has problem.
    return_array = np.convolve(
        vec, (np.ones(window_len))/window_len, mode='same')
    return return_array


def weighted_move_ave(vec, window_len):
    weight = np.arange(window_len) + 1
    weight = weight[::-1]
    convolved_array = np.convolve(vec, weight, mode='same') / weight.sum()
    return convolved_array


def exp_move_ave(vec, alpha):
    #pandas で関数があるらしい。
    return_array = np.zeros_like(vec)
    return_array[0] = vec[0] * alpha
    for i in range(len(vec) - 1):
        return_array[i+1] = vec[i + 1] * alpha + return_array[i] * (1 - alpha)
    return return_array


def move_med(vec, window_len):
    return_array = np.zeros_like(vec)
    idx = np.arange(len(vec))
    for i in range(len(vec)):
        kernel = np.where(
            (idx >= (idx[i] - window_len)) & (idx <= (idx[i] + window_len)), True, False)
        return_array[i] = np.median(vec[kernel])
    return return_array


def threshold_median_isoutliers(vec, thres=0.15):
    #not beautiful
    temp_vec = vec.copy()
    temp_vec = temp_vec / thres
    isoutlier = np.where((temp_vec > 1) | (temp_vec < -1), -1, 1)
    isoutlier[-1], isoutlier[0] = 1, 1
    return isoutlier


def qqplot(dist):
    plt.hist(dist, bins=350, rwidth=0.7)
    plt.xlim(-0.05, 0.05)
    plt.show()
    stats.probplot(dist, dist="norm", plot=plt)
    plt.ylim(-0.2, 0.2)
    plt.show()
