import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y
 
# ハイパスフィルタ
def highpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "high")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y
 
# バンドパスフィルタ
def bandpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y
 
# バンドストップフィルタ
def bandstop(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "bandstop")       #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y


# csvから列方向に順次フィルタ処理を行い保存する関数
def csv_filter(df, fp_lp, fs_lp, type):
    #df = pd.read_csv(in_file, encoding='SHIFT-JIS')                  # ファイル読み込み 
    dt = 1/60 #df.T.iloc[0,1]                                              # 時間刻み
    # データフレームを初期化
    df_filter = pd.DataFrame()
    #df_filter[df.columns[0]] = df.T.iloc[0]
 
    # ローパスの設定-----------------------------------------------------------------------------
    fp_lp = fp_lp
    fs_lp = fs_lp
    #fp_lp = 0.7                                                     # 通過域端周波数[Hz]
    #fs_lp = 2.0                                                     # 阻止域端周波数[Hz]
 
    # ハイパスの設定-----------------------------------------------------------------------------
    fp_hp = 25                                                       # 通過域端周波数[Hz]
    fs_hp = 10                                                       # 阻止域端周波数[Hz]
 
    # バンドパスの設定---------------------------------------------------------------------------
    fp_bp = np.array([15, 25])                                       # 通過域端周波数[Hz]※ベクトル
    fs_bp = np.array([5, 50])                                        # 阻止域端周波数[Hz]※ベクトル
 
    # バンドストップの設定---------------------------------------------------------------------------
    fp_bs = np.array([15, 25])                                       # 通過域端周波数[Hz]※ベクトル
    fs_bs = np.array([5, 50])                                        # 阻止域端周波数[Hz]※ベクトル
 
    gpass = 3                                                        # 通過域端最大損失[dB]
    gstop = 40                                                       # 阻止域端最小損失[dB]
 
    # 列方向に順次フィルタ処理をするコード
    for i in range(len(df.T)):
        print(df.T)
        data = df.T.iloc[i]                       # フィルタ処理するデータ列を抽出
        # フィルタ処理の種類を文字列で読み取って適切な関数を選択する
        if type == 'lp':
            # ローパスフィルタを実行
            print('wave=', i, ':Lowpass.')
            data = lowpass(x=data, samplerate=1 / dt,
                           fp=fp_lp, fs=fs_lp,
                           gpass=gpass, gstop=gstop)
        elif type == 'hp':
            # ハイパスフィルタを実行
            print('wave=', i, ':Highpass.')
            data = highpass(x=data, samplerate=1 / dt,
                           fp=fp_hp, fs=fs_hp,
                           gpass=gpass, gstop=gstop)
        elif type == 'bp':
            # バンドパスフィルタを実行
            print('wave=', i, ':Bandpass.')
            data = bandpass(x=data, samplerate=1/dt,
                            fp=fp_bp, fs=fs_bp,
                            gpass=gpass, gstop=gstop)
        elif type == 'bs':
            # バンドストップフィルタを実行
            print('wave=', i, ':Bandstop.')
            data = bandstop(x=data, samplerate=1/dt,
                            fp=fp_bs, fs=fs_bs,
                            gpass=gpass, gstop=gstop)
        else:
            # 文字列が当てはまらない時はパス（動作テストでフィルタかけたくない時はNoneとか書いて実行するとよい）
            pass
 
        data = pd.Series(data)                                       # dataをPandasシリーズデータへ変換
        df_filter[df.columns[i] + '_filter'] = data              # 保存用にデータフレームへdataを追加
 
    #df_filter.to_csv(out_file)                                       # フィルタ処理の結果をcsvに保存
 
    return df_filter

if __name__ == "__main__":
    csv_filter
