#動かしているのがpython3.9、matplotlibがあるのが3.7の可能性あり
from matplotlib import pyplot as plt
from random import randint
import math
from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def sine_wave(samplerate, frequency):
    x = list(range(samplerate))
    y = [math.sin(i*2*math.pi/samplerate*frequency) for i in x]
    return x, y

def add_noise(values):
    noise = [randint(-100, 100) * 0.01 for _ in values]
    y = [w + n for (w, n) in zip(values, noise)]
    return y

def filtered_kalman(data, param = 0.00001):
    dic = {}
    for i in range(len(data.T)):
        values = data.iloc[:,i]
        
        #kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                          #transition_covariance=0.0001 * np.eye(2)) # np.eyeは単位行列

        n_dim_obs = 1                  # 観測値の次元数
        n_dim_trend = 2                # トレンドの次元数（状態の次元数）
        n_dim_state = n_dim_trend

        F = np.array([
            [2, -1],
            [1, 0]
        ], dtype=float)

        G = np.array([
            [1],
            [0]
        ], dtype=float)

        H = np.array([
            [1, 0]
        ], dtype=float)
        param = param
        Q = np.eye(1) * param
        Q = G.dot(Q).dot(G.T)
        state_mean = np.zeros(n_dim_state)              # 状態の平均値ベクトルの初期値
        state_cov = np.ones((n_dim_state, n_dim_state)) # 状態の分散共分散行列の初期値
        kf = KalmanFilter(
        n_dim_obs=n_dim_obs,
        n_dim_state=n_dim_state,
        initial_state_mean=state_mean,
        initial_state_covariance=state_cov,
        transition_matrices=F,
        # システムノイズの共分散分散行列
        transition_covariance=Q,
        observation_matrices=H,
        # 観測ノイズの共分散分散行列(観測値が一次元の場合はスカラ)
        observation_covariance=1.0)    
        state_means, state_covs = kf.smooth(values)
        ovsevation_means_predicted = state_means.dot(H.T)
        #smoothed = kf.em(values).smooth(values)[0]
        #filtered = kf.em(values).filter(values)[0]
        #return smoothed, filtered
        dic[f'{i}'] = [ovsevation_means_predicted]
    
    
    for i in range(len(data.T)):
        kalman_data= pd.DataFrame(dic[f'{i}'][0].flatten())
        if i == 0:
            kalman_all = kalman_data
        else:
            kalman_all = pd.concat([kalman_all, kalman_data], axis = 1)

    head = data.columns.copy()
    kalman_all.columns = head
    print(kalman_all)
    #return ovsevation_means_predicted
    return kalman_all
def filtered_lowpass(values):
    res = [0 for _ in values]
    k = 1.3
    i = 0
    for a in values:
        try:
            b = res[i - 1]
        except:
            b = 0
        res[i] = k * a + (1-k) * b
        i += 1
    return res

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\sk122\OneDrive - 東京理科大学\4M\引継ぎ資料\実験データ\日体大\M5\20211014_094138.bag.csv_angle_velocity.csv', usecols = ['FROM right_waist TO right_knee'])
    #x, sine_y = sine_wave(500, 5)
    #noised_y = add_noise(sine_y)
    print(df)
    x = range(len(df))
    y1 = df
    #y1 = list(df.iloc[:,2])
    #smoothed, filtered = filtered_kalman(y1)
    kalman_all = filtered_kalman(y1)
    #lowpass_y = filtered_lowpass(y1)
    plt.figure(figsize=(8, 8), dpi=80)
    #plt.plot(x, sine_y, label='Original')
    plt.plot(x, y1, label='Original')
    #plt.plot(x, lowpass_y, label='FIR LPF')
    plt.plot(x, kalman_all, label='Kalman')
    # plt.plot(x, filtered[:, 0], label='Filtered')
    plt.ylim(-10,10)
    plt.legend()
    plt.show()

