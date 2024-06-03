import numpy as np
from scipy import fftpack
import pandas as pd
import matplotlib.pyplot as plt
 
# フーリエ変換をする関数
def calc_fft(data, samplerate):
    spectrum = fftpack.fft(data)                                     # 信号のフーリエ変換
    amp = np.sqrt((spectrum.real ** 2) + (spectrum.imag ** 2))       # 振幅成分
    amp = amp / (len(data) / 2)                                      # 振幅成分の正規化（辻褄合わせ）
    phase = np.arctan2(spectrum.imag, spectrum.real)                 # 位相を計算
    phase = np.degrees(phase)                                        # 位相をラジアンから度に変換
    freq = np.linspace(0, samplerate, len(data))                     # 周波数軸を作成
    return spectrum, amp, phase, freq
 
# csvから列方向に順次フーリエ変換を行い保存する関数
def csv_fft(in_file, out_file):
    df = pd.read_csv(in_file, encoding='SHIFT-JIS')                  # ファイル読み込み
    dt = df.T.iloc[0,1]                                              # 時間刻み
 
    # データフレームを初期化
    df_amp = pd.DataFrame()
    df_phase = pd.DataFrame()
    df_fft = pd.DataFrame()
 
    # 列方向に順次フーリエ変換（DFT）をするコード
    for i in range(len(df.T)-1):
        data = df.T.iloc[i+1]                                        # フーリエ変換するデータ列を抽出
        spectrum, amp, phase, freq = calc_fft(data.values, 1/dt)     # フーリエ変換をする関数を実行
        df_amp[df.columns[i+1] + '_amp'] = pd.Series(amp)            # 列名と共にデータフレームに振幅計算結果を追加
        df_phase[df.columns[i+1] + '_phase[deg]'] = pd.Series(phase) # 列名と共にデータフレームに位相計算結果を追加
 
    df_fft['freq[Hz]'] = pd.Series(freq)                             # 周波数軸を作成
    df_fft = df_fft.join(df_amp).join(df_phase)                      # 周波数・振幅・位相のデータフレームを結合
    df_fft = df_fft.iloc[range(int(len(df)/2) + 1),:]                # ナイキスト周波数でデータを切り捨て
    df_fft.to_csv(out_file)                                          # フーリエ変換の結果をcsvに保存
 
    return df, df_fft
 
# 関数を実行してcsvファイルをフーリエ変換するだけの関数を実行
df, df_fft = csv_fft(in_file='signals.csv', out_file='fft.csv')