import glob
from posixpath import basename
import pandas as pd
import PySimpleGUI as sg
import datetime
from pandas.core.series import Series
from scipy import signal
import numpy as np
import correct_func as correct
import os
import os.path
import glob
from natsort import natsorted

# change file (or folder) path accordingly, subject info（in this case, foler info）
subject = 'data'

folder = rf'C:\Users\sk122\inheriting\test\res\{subject}\csvdata\*.csv'
files = natsorted(glob.glob(folder))
for i, file in enumerate(files):
    filename = os.path.splitext(os.path.basename(file))[0]
    print(filename)
    
    os.makedirs(rf'C:\Users\sk122\inheriting\test\res\{subject}\{filename}', exist_ok=True)

    df = pd.read_csv(file, index_col=57, na_values=[0, -1])
    df_copy = df.copy()
    df_copy.interpolate(limit_direction='both', limit_area='inside',
                            inplace=True, method='linear')

    header_keypoints = [
        'head',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_waist',
        'right_knee',
        'right_foot',
        'left_waist',
        'left_knee',
        'left_foot',
        'right_eye',
        'left_eye',
        'right_ear',
        'left_ear'
    ]
    separator = ('_x', '_y', '_z')
    header_new = []
    for keypoint in header_keypoints:
        for sep in separator:
            header_new.append(keypoint + sep)
    header_new.append('framecount')
    header_new.append('timedelta[ms]')
    header_new.append('timestamp[ms]')

    # Filter
    # setting parameter
    n = len(df)
    Hz = 30
    dt = 1 / Hz
    fn = 1 / (2 * dt) # nyquist frequency
    fp = 2.7  # 通過域端周波数[Hz]
    fs = 3.7  # 阻止域端周波数[Hz]
    gpass = 1  # 通過域最大損失量[dB]
    gstop = 40  # 阻止域最大減衰量[dB]

    # normalize
    Wp = fp / fn
    Ws = fs / fn

    # setting LPF
    # Buttorworth filter
    N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
    b, a = signal.butter(N, Wn, btype='low')


    df_for_save = pd.DataFrame(index=df.index)
    for i in range(18):
        x = df_copy.iloc[:, 3 * i]
        y = df_copy.iloc[:, 3 * i + 1]
        z = df_copy.iloc[:, 3 * i + 2]
        x.fillna(0, inplace=True)
        y.fillna(0, inplace=True)
        z.fillna(0, inplace=True)

        for j in range(3):
            data = df_copy.iloc[:, 3 * i + j]
            data.fillna(0, inplace=True)
            series = data.values # 列ごとに切り出し

            # move median range
            move_median = correct.move_med(series, 15)
            diff = series - move_median
            
            # median outliers filter : parameter > threshold
            isout = correct.threshold_median_isoutliers(diff, 0.1)

            x = np.where(isout > 0, x, np.nan)
            y = np.where(isout > 0, y, np.nan)
            z = np.where(isout > 0, z, np.nan)

        x = Series(x, index=df_copy.index)
        y = Series(y, index=df_copy.index)
        z = Series(z, index=df_copy.index)
        
        x.interpolate(limit_direction='both', limit_area='inside', inplace=True, method='linear')
        y.interpolate(limit_direction='both', limit_area='inside', inplace=True, method='linear')
        z.interpolate(limit_direction='both', limit_area='inside', inplace=True, method='linear')
    
        x = signal.filtfilt(b, a, x.values)
        y = signal.filtfilt(b, a, y.values)
        z = signal.filtfilt(b, a, z.values)

        x = Series(x, index=df_copy.index,
                        name=df.columns[i * 3])
        y = Series(y, index=df_copy.index,
                        name=df.columns[i * 3 + 1])
        z = Series(z, index=df_copy.index,
                        name=df.columns[i * 3 + 2])

        df_for_save = pd.concat([df_for_save, x, y, z], axis=1)    
        # print(df_for_save)
        # print('-------------')
        # print(y)
        # print('-------------')
        # print(z)
        # print('-------------')
    df_for_save['framecount'] = df_copy.iloc[:, 54].values
    df_for_save['timedelta[ms]'] = df_copy.iloc[:, 55].values
    df_for_save['timestamp[ms]'] = df_copy.iloc[:, 56].values
    df_for_save.columns = header_new
    print(df_for_save)
    df_for_save.to_csv(rf'C:\Users\sk122\inheriting\test\res\{subject}\{filename}\coor_correct_data.csv')