import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

subject = 'saito_aoto'
setnum = '70RM_1'
# path = r"C:\Users\sk723\OneDrive - 東京理科大学\4M\EMG\M1_N\saito_aota\saito_70RM_2\set_4.csv"
path = rf"C:\Users\yota0\Desktop\kamiya\卒業論文\データ\筋電データ\筋電データ\saito_aota\emg\70RM_1.CSV"
outpath = rf"C:\Users\yota0\Desktop\kamiya\program_copy\out\Average_frequency\{subject}\{setnum}"
os.makedirs(outpath, exist_ok=True)

data = pd.read_csv(path)
rate = 1000

data_split = [
    [
        [1000, 2287],
        [3040, 4184],
        [5456, 6718],
        [7577, 8735],
        [10012, 11310],
        [12517, 14014],
        [15001, 16397],
        [17397, 19050],
        [20749, 22435],
        [24283, 26001]
    ],
    [
        [1128, 3008],
        [3409, 5007],
        [5343, 7003],
        [7297, 9046],
        [9571, 11389],
        [12359, 14052],
        [14784, 16486],
        [17378, 19099],
        [20713, 22508],
        [24140, 26013]
    ],
    [
        [1205, 3017],
        [3344, 5008],
        [5290, 7049],
        [7348, 9085],
        [9615, 11340],
        [12224, 14039],
        [14639, 16393],
        [17224, 19197],
        [20206, 22496],
        [24005, 26023]
    ],
    [
        [2420, 3352],
        [4435, 5380], 
        [6469, 7487],
        [8515, 9623], 
        [11018, 12234],
        [13567, 14640],
        [16005, 17103],
        [18500, 20490],
        [22021, 24027],
        [25362, 28330]
    ]
]






freqency_mean = []

for j in range(len(data_split)):
    fq_mean = []
    for k in range(len(data_split[0])):
        emg_1 = data.iloc[data_split[j][k][0]:data_split[j][k][1], j+1]
        N = len(emg_1)

        t = [i * 0.001 for i in range(len(emg_1))]
        fft_1 = np.fft.fft(emg_1)

        fft_1 = np.abs(fft_1)[0:N//2]

        freq = np.fft.fftfreq(N, d=1/rate)[:N//2]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t, emg_1)
        plt.subplot(1, 2, 2)
        plt.plot(freq, fft_1*2.0/N)
        plt.savefig(rf'{outpath}\part_{j}_rep_{k}.png')
        # plt.show()

        freq_df = pd.DataFrame([freq[1:int(N/2)], fft_1[1:int(N/2)]], index=['freqency [Hz]', 'amplitude [rtHz]'])
        freq_df = freq_df.T
        freq_power = []
        power_sum = []
        amplitude = []
        for i in range(len(freq_df)):
            freqency = freq_df.iloc[i, 0]
            amp = freq_df.iloc[i, 1]
            power = amp ** 2
            freq_power.append(freqency * power)
            power_sum.append(power)
            amplitude.append(amp)

        mean_freq = sum(freq_power) / sum(power_sum)
        fq_mean.append(mean_freq)
    print(fq_mean)
    freqency_mean.append(fq_mean)

df = pd.DataFrame(freqency_mean, columns=[i+1 for i in range(10)], index=['Rectus femoris', 'Vastus lateralis', 'Vastus medialis', 'Biceps femoris'])
df = df.T
df.to_csv(rf'{outpath}\freq_mean.csv')