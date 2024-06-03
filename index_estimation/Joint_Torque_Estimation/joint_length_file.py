from multiprocessing.managers import Namespace
import pandas as pd
import numpy as np
import os
import glob
import statistics
from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path
import sort

def calculate_median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)

    # リストの要素数が奇数の場合
    if n % 2 != 0:
        median = sorted_lst[n // 2]
    else:
        # リストの要素数が偶数の場合は中央の2つの値の平均を取る
        middle1 = sorted_lst[(n // 2) - 1]
        middle2 = sorted_lst[n // 2]
        median = (middle1 + middle2) / 2

    return median

# Subject information (in this case, folder name)
root = Tk()
root.withdraw()

folderpath = askdirectory()
#filepath = rf'C:\\Users\\yota0\\Documents\\Yota\\githubrepos\\poseestimate_mediapipe\\poseestimate_mediapipe\\out\\modelbased'

file_path = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\base\LENGTH_WORKSET.csv'
length_data = pd.read_csv(file_path, usecols=[1,2,3])

for file_name in glob.glob(rf'{folderpath}\*'):
    filepath = os.path.split(os.path.basename(file_name))[-1]
    filename = filepath.replace("_correct_modelbased.csv", "")

    if not filepath.endswith(".csv"):
        pass
    else:
        # 取り出すことのできる関節長さ
        head_neck = []
        shoulder_waist = []
        shoulder_elbow = []
        elbow_wrist = []
        waist_knee = []
        knee_ankle = []

        # print(sum(os.path.isfile(os.path.join(d, name)) for name in os.listdir(d)))

        Ld = {}

        # エクセルデータの読み込み
        ord_df = pd.read_csv(f'{file_name}',header=1)
        # データ列の切り出し
        #TODO1データの作りが違うので、修正必須
        df = ord_df.loc[:, 'NOSE_x':'RIGHT_FOOT_INDEX_z']
        iterables = [
            ["head", "left_eye_inner", "left_eye", "left_eye_outer",
                "right_eye_inner", "right_eye", "right_eye_outer",
                "left_ear", "right_ear",
                "left_mouth", "right_mouth",
                "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
                "left_pinky", "right_pinky",
                "left_index", "right_index",
                "left_thumb", "right_thumb",
                "left_waist", "right_waist",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle",
                "left_heel", "right_heel",
                "left_foot", "right_foot"
                ],
            ['x', 'y', 'z']]
        
        #iterables = [
            #["head", "neck",
                #"right_shoulder", "right_elbow", "right_wrist",
                #"left_shoulder", "left_elbow", "left_wrist",
                #"right_waist", "right_knee", "right_foot",
                #"left_waist", "left_knee", "left_foot",
                #"right_eye", "left_eye",
                #"right_ear", "left_ear"
                #],
            #['x', 'y', 'z']]
        df.columns = pd.MultiIndex.from_product(iterables)
        
        #いらないかも
        #L_shoulder = df.loc[:, ('left_shoulder', slice(None))].droplevel(0, axis=1)
        #R_shoulder = df.loc[:, ('right_shoulder', slice(None))].droplevel(0, axis=1)
        #L_mouth = df.loc[:, ('left_mouth', slice(None))].droplevel(0, axis=1)
        #R_mouth = df.loc[:, ('right_mouth', slice(None))].droplevel(0, axis=1)
        
        x_neck = df.loc[:, [('left_shoulder','x'), ('right_shoulder','x'), ('left_mouth','x'), ('right_mouth','x')]].mean(axis=1)
        y_neck = df.loc[:, [('left_shoulder','y'), ('right_shoulder','y'), ('left_mouth','y'), ('right_mouth','y')]].mean(axis=1)
        z_neck = df.loc[:, [('left_shoulder','z'), ('right_shoulder','z')]].mean(axis=1)
        
        
            
        #new_cols = [("neck", dim) for dim in ['x' ,'y' ,'x']]
        #df.columns = df.columns.append(new_cols)
        
        #new_cols = [("neck", dim)]
        #df.columns = pd.MultiIndex.from_tuples(list(df.columns) + list(new_cols))
        
        df['neck', 'x'] = x_neck
        df['neck', 'y'] = y_neck
        df['neck', 'z'] = z_neck
        
        #df.columns = df.columns.append(new_cols)
        
        # 首の位置を取得
        neck = df.loc[:, ('neck', slice(None))].droplevel(0, axis=1)
        # 右腰の位置を取得
        right_foot = df.loc[:, ('right_foot', slice(None))].droplevel(0, axis=1)
        # 左腰の位置を取得
        left_foot = df.loc[:, ('left_foot', slice(None))].droplevel(0, axis=1)

        # 結果データを格納するための入れ物
        dfs = {}

        # パーツ間毎に処理
        target_pair = [('neck', 'right_shoulder'),
                        ('right_shoulder', 'right_elbow'),
                        ('right_elbow', 'right_wrist'),
                        ('neck', 'left_shoulder'),
                        ('left_shoulder', 'left_elbow'),
                        ('left_elbow', 'left_wrist'),
                        ('neck', 'right_waist'),
                        ('right_shoulder','right_waist'),#ここ長さ取り出し
                        ('left_shoulder', 'left_waist'),#ここ長さ取り出し
                        ('right_waist', 'right_knee'),
                        ('right_knee', 'right_foot'),
                        ('neck', 'left_waist'),
                        ('left_waist', 'left_knee'),
                        ('left_knee', 'left_foot'),
                        ('neck', 'head'),
                        ('head', 'right_eye'),
                        ('right_eye', 'right_ear'),
                        ('head', 'left_eye'),
                        ('left_eye', 'left_ear')]
        
        # 長さを一旦入れる
        length = []
        # 上記に定義したペア毎にループ
        for start_part, end_part in target_pair:
            # 始点座標
            sp = df.loc[:, (start_part, slice(None))].droplevel(0, axis=1)
            
            # 終点座標
            ep = df.loc[:, (end_part, slice(None))].droplevel(0, axis=1)

            # ベクトルを求める
            vec = ep - sp
            
            theta = []

            # 極座標変換
            r = vec.apply(np.linalg.norm, axis=1)
            theta = (vec.y / r).apply(np.arccos)
            theta = theta.fillna(0)
            phi = vec.apply(lambda d: np.arctan2(d.z, d.x), axis=1)
            
            # 新たな長さを取得
            r_list = []
            for i in range(len(r)):
                if not r.iloc[i] == 0:
                    r_list.append(r.iloc[i])
            new_r = calculate_median(r_list)
            # 一旦格納
            length.append(new_r)
            
            # 関節間の長さ(部分長用)
            if start_part=='neck' and end_part=='head':
                head_neck.append(new_r)
            elif (start_part=='right_shoulder' and end_part =='right_waist') or (start_part=='left_shoulder' and end_part =='left_waist'):
                shoulder_waist.append(new_r)
            elif (start_part=='right_shoulder' and end_part =='right_elbow') or (start_part=='left_shoulder' and end_part =='left_elbow'):
                shoulder_elbow.append(new_r)
            elif (start_part=='right_elbow' and end_part =='right_wrist') or (start_part=='left_elbow' and end_part =='left_wrist'):
                elbow_wrist.append(new_r)
            elif (start_part=='right_waist' and end_part =='right_knee') or (start_part=='left_waist' and end_part =='left_knee'):
                waist_knee.append(new_r)
            elif (start_part=='right_knee' and end_part =='right_foot') or (start_part=='left_knee' and end_part =='left_foot'):
                knee_ankle.append(new_r)
        # 辞書に格納
        Ld[Namespace] = length

        keys = list(Ld.keys())
        # フォルダ内の関節長さ中央値の平均値
        ave = np.mean(([Ld[key] for key in keys]), axis = 0)
        # 阿江部分長用のcsv
        Head_Neck = pd.DataFrame([statistics.mean(head_neck)])
        Sholder_Waist = pd.DataFrame([statistics.mean(shoulder_waist)])
        Shoulder_Elbow = pd.DataFrame([statistics.mean(shoulder_elbow)])
        Elbow_Wrist = pd.DataFrame([statistics.mean(elbow_wrist)])
        Waist_Knee = pd.DataFrame([statistics.mean(waist_knee)])
        Knee_Ankle = pd.DataFrame([statistics.mean(knee_ankle)])
        joint_length = pd.concat([Head_Neck,Sholder_Waist,Shoulder_Elbow,
                                Elbow_Wrist,Waist_Knee,Knee_Ankle], axis = 1)
        joint_length.columns = ['頭-首','肩-腰','肩-肘','肘-手首','腰-膝','膝-足首']
        # os.makedirs(f'Data/Output/length/', exist_ok = True)
        joint_length.to_csv(rf'C:\Users\yota0\Desktop\kamiya\program_copy\out\length_csvdeta\{filename}_length.csv', index=False, encoding="shift jis")
        # os.makedirs(f'Data/Output/length_correction', exist_ok = True)

        # 補正(上のと同じ部分多い)
        # エクセルデータの読み込み
        #ord_df = pd.read_csv(f'{filepath}',header=1)
        
        # データ列の切り出し
        #TODO1
        #df = ord_df.loc[:, 'NOSE_x':'RIGHT_FOOT_INDEX_z']
        #iterables = [
            #["head", "left_eye_inier", "left_eye", "left_eye_outer",
                #"right_eye_inier", "right_eye", "right_eye_outer",
                #"left_ear", "right_ear",
                #"left_mouth", "right_mouth",
                #"left_shoulder", "right_shoulder",
                #"left_elbow", "right_elbow",
                #"left_wrist", "right_wrist",
                #"left_pinky", "right_pinky",
                #"left_index", "right_index",
                #"left_thumb", "right_thumb",
                #"left_waist", "right_waist",
                #"left_knee", "right_knee",
                #"left_ankle", "right_ankle"
                #"left_heel", "right_heel",
                #"left_foot", "right_foot",
                #],
            #['x', 'y', 'z']]
        
        #iterables = [
            #["head", "neck",
                #"right_shoulder", "right_elbow", "right_wrist",
                #"left_shoulder", "left_elbow", "left_wrist",
                #"right_waist", "right_knee", "right_foot",
                #"left_waist", "left_knee", "left_foot",
                #"right_eye", "left_eye",
                #"right_ear", "left_ear"
                #],
            #['x', 'y', 'z']]
            
        #df.columns = pd.MultiIndex.from_product(iterables)
        
        #いらないかも
        #L_shoulder = df.loc[:, ('left_shoulder', slice(None))].droplevel(0, axis=1)
        #R_shoulder = df.loc[:, ('right_shoulder', slice(None))].droplevel(0, axis=1)
        #L_mouth = df.loc[:, ('left_mouth', slice(None))].droplevel(0, axis=1)
        #R_mouth = df.loc[:, ('right_mouth', slice(None))].droplevel(0, axis=1)
        
        #x_neck = df.loc[:, [('left_shoulder','x'), ('right_shoulder','x'), ('left_mouth','x'), ('right_mouth','x')]].mean()
        #y_neck = df.loc[:, [('left_shoulder','y'), ('right_shoulder','y'), ('left_mouth','y'), ('right_mouth','y')]].mean()
        #z_neck = df.loc[:, [('left_shoulder','z'), ('right_shoulder','z')]].mean()
        
        #new_cols = pd.MultiIndex.from_product([[["neck"], ['x', 'y', 'z']], df.columns])
        
        #df.columns = pd.MultiIndex.from_tuples(list(df.columns) + list(new_cols))
        
        #df['neck', 'x'] = x_neck
        #df['neck', 'y'] = y_neck
        #df['neck', 'z'] = z_neck
        
        # 首の位置を取得
        #neck = df.loc[:, ('neck', slice(None))].droplevel(0, axis=1)
        # 右腰の位置を取得
        #right_foot = df.loc[:, ('right_foot', slice(None))].droplevel(0, axis=1)
        # 左腰の位置を取得
        #left_foot = df.loc[:, ('left_foot', slice(None))].droplevel(0, axis=1)

        # 結果データを格納するための入れ物
        dfs = {}

        # パーツ間毎に処理
        target_pair = [('neck', 'right_shoulder'),
                        ('right_shoulder', 'right_elbow'),
                        ('right_elbow', 'right_wrist'),
                        ('neck', 'left_shoulder'),
                        ('left_shoulder', 'left_elbow'),
                        ('left_elbow', 'left_wrist'),
                        ('neck', 'right_waist'),
                        ('right_shoulder','right_waist'),#ここ長さ取り出し
                        ('left_shoulder', 'left_waist'),#ここ長さ取り出し
                        ('right_waist', 'right_knee'),
                        ('right_knee', 'right_foot'),
                        ('neck', 'left_waist'),
                        ('left_waist', 'left_knee'),
                        ('left_knee', 'left_foot'),
                        ('neck', 'head'),
                        ('head', 'right_eye'),
                        ('right_eye', 'right_ear'),
                        ('head', 'left_eye'),
                        ('left_eye', 'left_ear')]

        # 上記に定義したペア毎にループ
        for (start_part, end_part), new_r in zip(target_pair,ave):
            # 始点座標
            sp = df.loc[:, (start_part, slice(None))].droplevel(0, axis=1)
            # 終点座標
            ep = df.loc[:, (end_part, slice(None))].droplevel(0, axis=1)
            # ベクトルを求める
            vec = ep - sp
            # 極座標変換
            r = vec.apply(np.linalg.norm, axis=1)
            theta = (vec.y / r).apply(np.arccos)
            theta = theta.fillna(0)
            phi = vec.apply(lambda d: np.arctan2(d.z, d.x), axis=1)
            # 新たな長さを取得
            new_r = new_r
            # XYZ座標変換
            #x = new_r * theta.apply(np.sin) * phi.apply(np.sin)
            #y = new_r * (vec.y / r)
            x = new_r * theta.apply(np.sin) * phi.apply(np.cos)
            y = new_r * theta.apply(np.cos)
            z = new_r * theta.apply(np.sin) * phi.apply(np.sin)
            #z = new_r * theta.apply(np.sin) * phi.apply(np.cos)

            # 結果を格納
            dfs[(start_part, end_part)] = pd.DataFrame({'x': x, 'y': y, 'z': z})
            
        # 各パーツの位置を算出
        # 頭の位置
        head = neck + dfs[('neck', 'head')]
        # 右肩の位置
        right_shoulder = neck + dfs[('neck', 'right_shoulder')]
        # 右肘の位置
        right_elbow = right_shoulder + dfs[('right_shoulder', 'right_elbow')]
        # 右手の位置
        right_wrist = right_elbow + dfs[('right_elbow', 'right_wrist')]
        # 左肩の位置
        left_shoulder = neck + dfs[('neck', 'left_shoulder')]
        # 左肘の位置
        left_elbow = left_shoulder + dfs[('left_shoulder', 'left_elbow')]
        # 左手の位置
        left_wrist = left_elbow + dfs[('left_elbow', 'left_wrist')]
        # 右腰の位置
        right_waist = right_shoulder + dfs[('neck', 'right_waist')]
        # 右膝の位置
        right_knee = right_waist + dfs[('right_waist', 'right_knee')]
        # 右足の位置
        right_foot = right_knee + dfs[('right_knee', 'right_foot')]
        # 左腰の位置
        left_waist = neck + dfs[('neck', 'left_waist')]
        # 左膝の位置
        left_knee = left_waist + dfs[('left_waist', 'left_knee')]
        # 左足の位置
        left_foot = left_knee + dfs[('left_knee', 'left_foot')]
        # 右目の位置
        right_eye = head + dfs[('head', 'right_eye')]
        # 右耳の位置
        right_ear = right_eye + dfs[('right_eye', 'right_ear')]
        # 左目の位置
        left_eye = head + dfs[('head', 'left_eye')]
        # 左耳の位置
        left_ear = left_eye + dfs[('left_eye', 'left_ear')]


        # 各パーツの位置データのカラム名を修正
        # 現在、各パーツの位置のデータのカラム名は全て[x, y, z] となっている
        # 最終的にこれらのデータを結合してエクセル化するが、結合前にカラムを元の名前に戻しておく
        # （例）頭の位置データ [x, y, z] -> [head_x, head_y, head_z]
        head.rename(columns={'x': 'head_x', 'y': 'head_y',
                    'z': 'head_z'}, inplace=True)
        neck.rename(columns={'x': 'neck_x', 'y': 'neck_y',
                    'z': 'neck_z'}, inplace=True)
        right_shoulder.rename(columns={'x': 'right_shoulder_x',
                                'y': 'right_shoulder_y', 'z': 'right_shoulder_z'}, inplace=True)
        right_elbow.rename(columns={
                            'x': 'right_elbow_x', 'y': 'right_elbow_y', 'z': 'right_elbow_z'}, inplace=True)
        right_wrist.rename(columns={
                            'x': 'right_wrist_x', 'y': 'right_wrist_y', 'z': 'right_wrist_z'}, inplace=True)
        left_shoulder.rename(columns={'x': 'left_shoulder_x',
                                'y': 'left_shoulder_y', 'z': 'left_shoulder_z'}, inplace=True)
        left_elbow.rename(columns={'x': 'left_elbow_x',
                            'y': 'left_elbow_y', 'z': 'left_elbow_z'}, inplace=True)
        left_wrist.rename(columns={'x': 'left_wrist_x',
                            'y': 'left_wrist_y', 'z': 'left_wrist_z'}, inplace=True)
        right_waist.rename(columns={
                            'x': 'right_waist_x', 'y': 'right_waist_y', 'z': 'right_waist_z'}, inplace=True)
        right_knee.rename(columns={'x': 'right_knee_x',
                            'y': 'right_knee_y', 'z': 'right_knee_z'}, inplace=True)
        right_foot.rename(columns={'x': 'right_foot_x',
                            'y': 'right_foot_y', 'z': 'right_foot_z'}, inplace=True)
        left_waist.rename(columns={'x': 'left_waist_x',
                            'y': 'left_waist_y', 'z': 'left_waist_z'}, inplace=True)
        left_knee.rename(columns={'x': 'left_knee_x',
                            'y': 'left_knee_y', 'z': 'left_knee_z'}, inplace=True)
        left_foot.rename(columns={'x': 'left_foot_x',
                            'y': 'left_foot_y', 'z': 'left_foot_z'}, inplace=True)
        right_eye.rename(columns={'x': 'right_eye_x',
                            'y': 'right_eye_y', 'z': 'right_eye_z'}, inplace=True)
        right_ear.rename(columns={'x': 'right_ear_x',
                            'y': 'right_ear_y', 'z': 'right_ear_z'}, inplace=True)
        left_eye.rename(columns={'x': 'left_eye_x',
                        'y': 'left_eye_y', 'z': 'left_eye_z'}, inplace=True)
        left_ear.rename(columns={'x': 'left_ear_x',
                        'y': 'left_ear_y', 'z': 'left_ear_z'}, inplace=True)

        # 結果を結合
        # 上記で作成した位置データを再度結合して、元のデータフレーム（エクセル）と同等の並びのデータフレームを生成
        # count列、framecount列, timedelta[ms]列, timestamp[ms]列はもとのデータフレームのものを使用する
        new_df = pd.concat([ord_df[['count']], head, neck,
                            right_shoulder, right_elbow, right_wrist,
                            left_shoulder, left_elbow, left_wrist,
                            right_waist, right_knee, right_foot,
                            left_waist, left_knee, left_foot,
                            right_eye, right_ear, left_eye, left_ear,
                            ord_df[['framecount', 'timestamp', 'backendtime']]], axis=1)

        # 結果を表示
        # print(new_df)
        # 結果出力
        # データフレームをエクセルデータの書き出す
        # new_df.to_csv(f'Data/Output/length_correction/{n}/{filename}', index=False)
        new_df.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta\{filename}_length_correction.csv', index=False)
        pd.set_option('display.max_columns', None)

        # ここから角度補正
        filepath = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta\{filename}_length_correction.csv'
        dfo = pd.read_csv(f'{filepath}')
        print(dfo)
        print('--------')
        count_df = dfo.loc[:, 'count']
        #ヘッダー取り出し
        dfoheader = list(dfo.columns.values)
        # データ列の切り出し
        df = dfo.loc[:, 'head_x':'left_ear_z']
        header_list = df.columns.values
        header_new = ",".join(header_list)
        header_new = header_new.replace('_x', '')
        header_new = header_new.replace('_y', '')
        header_new = header_new.replace('_z', '')
        header_new = header_new.split(",")
        header_new = list(header_new)

        df.columns = header_new
        df_count = dfo.loc[:, 'count']
        other = dfo.loc[:, 'right_eye_x':'backendtime']
        # 座標軸を全部反対にする
        df = df *-1

        def vec(data, start_part, end_part):
            sp = data.loc[:, start_part]
            sp.columns = ['x', 'y', 'z']
            ep = data.loc[:, end_part]
            ep.columns = ['x', 'y', 'z']
            vec = ep - sp
            return vec

        rw = df.loc[:, 'right_waist']
        rw.columns = ['x', 'y', 'z']
        lw = df.loc[:, 'left_waist']
        lw.columns = ['x', 'y', 'z']
        mid_waist = (rw + lw)/2
        mid_waist.columns = ['mid_waist' + '_' + i  for i in ['x', 'y', 'z']]
        m = np.array(mid_waist)
        d = pd.DataFrame(m)
        mid_waist.columns = ['x', 'y', 'z']

        '''
        a = (mid_waist.iloc[:, 2]>-0.025) & (mid_waist.iloc[:, 1]<-0.1)
        mid_waist.iloc[~a] = np.nan
        print(mid_waist.iloc[~2])
        print(mid_waist)
        L = pd.concat([mid_waist, rw], axis = 1)
        '''
        def simple_col(data, p):
            simple = data.loc[:, p]
            simple.columns = ['x', 'y', 'z']
            return simple
            

        def Motion_Range(data):
            # 相対座標系XYZ
            XYZ = np.array([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]] * len(data))
            # 0. Head()
            # 基準軸uvw Neck中心
            # v1: Neck→Head軸
            # w1: LeftShoulder→RightShoulderとv1軸の外積
            # u1: v1軸とw1軸の外積
            NtoHead1_1 = np.array(vec(data, 'neck', 'head'))
            NtoHead1_1_copy = np.copy(NtoHead1_1)
            NtoHead   = np.array(simple_col(data, 'head') - simple_col(data, 'neck'))
            len_NtoHead = np.linalg.norm(NtoHead, ord=2, axis=1).reshape(len(data), 1)
            MHtoN1   = np.array(simple_col(data, 'neck') - mid_waist)
            len_MHtoN1 = np.linalg.norm(MHtoN1, ord=2, axis=1).reshape(len(data), 1)
            v1 = MHtoN1 / len_MHtoN1
            #v1 = NtoHead / len_NtoHead
            LStoRS1 = np.array(vec(data, 'left_shoulder', 'right_shoulder'))
            len_LStoRS1 = np.linalg.norm(LStoRS1, ord=2, axis=1).reshape(len(data), 1)
            LStoRS1_norm = LStoRS1 / len_LStoRS1
            w1 = np.cross(LStoRS1_norm, v1, axis=1)
            u1 = np.cross(v1, w1, axis=1)
            uvw1 = np.array([u1, v1, w1]).transpose(1, 2, 0)
            
            #for文で書き変えしかないかも
            j = 1
            start_list = uvw1[0]
            for i in range(len(data)):
                matrix_factor = uvw1[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw1[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw1[i - j + k + 2] = start_list
                
            # 回転行列
            RotMatrix1 = np.linalg.solve(uvw1, XYZ)
            # 相対座標計算(回転後)
            NtoHead1_2 = np.matmul(RotMatrix1, NtoHead1_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) -50<β<+50
            #Head_beta1 = np.arctan2(NtoHead1_2[:, 0], NtoHead1_2[:, 1])
            Head_beta1 = np.arctan2(NtoHead1_2[:, 0], NtoHead1_2[:, 1])
            # 回転角γ(w軸回り) -50<γ<+60
            Head_gamma1 = np.arctan2(-NtoHead1_2[:, 1], NtoHead1_2[:, 2])
            #Head_gamma1 = np.arctan2(NtoHead1_2[:, 2], NtoHead1_2[:, 1])

            # print(Head_beta1)
            HEAD = pd.DataFrame(Head_beta1)
            HEAD2 = pd.DataFrame(Head_gamma1)
            # print(HEAD)
            HEAD3 = pd.concat([HEAD,HEAD2], axis = 1)
            # print(HEAD3)
            # os.makedirs(rf'..\result\{subject}\head', exist_ok=True)
            HEAD3.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta\{filename}_HEAD.csv')
            

            # 可動域内にあるか判定
            Head_ToF1 = (Head_beta1>(-50*np.pi/180)) & (Head_beta1<(50*np.pi/180)) & (Head_gamma1>(-50*np.pi/180)) & (Head_gamma1<(60*np.pi/180)) 
            # 可動域にない座標点をnanで置き換え
            NtoHead1_1[~Head_ToF1] = np.nan
            # pandasで線形補間
            NtoHead1_1_df = pd.DataFrame(NtoHead1_1).interpolate(axis=0)
            # 補正後の出力
            NtoHead1_1_corrected = NtoHead1_1_df.to_numpy()
            # nanを置き換え
            NtoHead1_1_corrected = np.nan_to_num(NtoHead1_1_corrected, nan=NtoHead1_1_copy)        
            # 相対座標(回転前)データに格納
            head = simple_col(data,'head')
            head = pd.DataFrame(NtoHead1_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'neck')
            #ls = pd.DataFrame(NtoLS1_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'neck')
            
            # 1. Shoulder(1, 4)
            # 基準軸uvw Neck中心
            # v1: MidHip→Neck軸
            # w1: LeftShoulder→RightShoulderとv1軸の外積
            # u1: v1軸とw1軸の外積
            NtoRS1_1 = np.array(vec(data, 'neck', 'right_shoulder'))
            NtoLS1_1 = np.array(vec(data, 'neck', 'left_shoulder'))
            NtoRS1_1_copy = np.copy(NtoRS1_1)
            NtoLS1_1_copy = np.copy(NtoLS1_1)
            MHtoN1   = np.array(simple_col(data, 'neck') - mid_waist)
            len_MHtoN1 = np.linalg.norm(MHtoN1, ord=2, axis=1).reshape(len(data), 1)
            v1 = MHtoN1 / len_MHtoN1
            LStoRS1 = np.array(vec(data, 'left_shoulder', 'right_shoulder'))
            len_LStoRS1 = np.linalg.norm(LStoRS1, ord=2, axis=1).reshape(len(data), 1)
            LStoRS1_norm = LStoRS1 / len_LStoRS1
            w1 = np.cross(LStoRS1_norm, v1, axis=1)
            u1 = np.cross(v1, w1, axis=1)
            uvw1 = np.array([u1, v1, w1]).transpose(1, 2, 0)
            
            j = 1
            start_list = uvw1[0]
            for i in range(len(data)):
                matrix_factor = uvw1[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw1[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw1[i - j + k + 2] = start_list
            
            # 回転行列
            RotMatrix1 = np.linalg.solve(uvw1, XYZ)
            # 相対座標計算(回転後)
            NtoRS1_2 = np.matmul(RotMatrix1, NtoRS1_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            NtoLS1_2 = np.matmul(RotMatrix1, NtoLS1_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) -20<β<+20
            RS_beta1 = np.arctan2(-NtoRS1_2[:, 0], NtoRS1_2[:, 2])
            LS_beta1 = np.arctan2(-NtoLS1_2[:, 0], -NtoLS1_2[:, 2])
            # 回転角γ(w軸回り) -10<γ<+20
            RS_gamma1 = np.arctan2(NtoRS1_2[:, 1], NtoRS1_2[:, 2])
            LS_gamma1 = np.arctan2(NtoLS1_2[:, 1], -NtoLS1_2[:, 2])
            # 可動域内にあるか判定
            RS_ToF1 = (RS_beta1>(-20*np.pi/180)) & (RS_beta1<(20*np.pi/180)) & (RS_gamma1>(-10*np.pi/180)) & (RS_gamma1<(20*np.pi/180))
            LS_ToF1 = (LS_beta1>(-20*np.pi/180)) & (LS_beta1<(20*np.pi/180)) & (LS_gamma1>(-10*np.pi/180)) & (LS_gamma1<(20*np.pi/180))
            # 可動域にない座標点をnanで置き換え
            NtoRS1_1[~RS_ToF1] = np.nan
            NtoLS1_1[~LS_ToF1] = np.nan
            # pandasで線形補間
            NtoRS1_1_df = pd.DataFrame(NtoRS1_1).interpolate(axis=0)
            NtoLS1_1_df = pd.DataFrame(NtoLS1_1).interpolate(axis=0)
            # 補正後の出力
            NtoRS1_1_corrected = NtoRS1_1_df.to_numpy()
            NtoLS1_1_corrected = NtoLS1_1_df.to_numpy()
            # nanを置き換え
            NtoRS1_1_corrected = np.nan_to_num(NtoRS1_1_corrected, nan=NtoRS1_1_copy)
            NtoLS1_1_corrected = np.nan_to_num(NtoLS1_1_corrected, nan=NtoLS1_1_copy)
            # 相対座標(回転前)データに格納
            rs = simple_col(data,'right_shoulder')
            ls = simple_col(data,'left_shoulder')
            rs = pd.DataFrame(NtoRS1_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'neck')
            ls = pd.DataFrame(NtoLS1_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'neck')
            # print(rs)
            
            # 2. Elbow(2, 5)
            # 基準軸uvw Shoulder中心
            # v2: MidHip→Neck軸
            # w2_1, w2_2: Neck→RLShoulderとv軸の外積
            # u2_1, u2_2: v軸とw1, w2軸の外積
            RStoRE2_1 = np.array(vec(data, 'right_shoulder', 'right_elbow'))
            LStoLE2_1 = np.array(vec(data, 'left_shoulder', 'left_elbow'))
            RStoRE2_1_copy = np.copy(RStoRE2_1)
            LStoLE2_1_copy = np.copy(LStoLE2_1)
            v2 = v1
            NtoRS2 = np.array(vec(data, 'neck', 'right_shoulder'))
            NtoLS2 = np.array(vec(data, 'neck', 'left_shoulder'))
            len_NtoRS2 = np.linalg.norm(NtoRS2, ord=2, axis=1).reshape(len(data), 1)
            len_NtoLS2 = np.linalg.norm(NtoLS2, ord=2, axis=1).reshape(len(data), 1)
            NtoRS2_norm = NtoRS2 / len_NtoRS2
            NtoLS2_norm = NtoLS2 / len_NtoLS2
            w2_1 = np.cross(NtoRS2_norm, v2, axis=1)
            w2_2 = np.cross(NtoLS2_norm, v2, axis=1)
            u2_1 = np.cross(v2, w2_1, axis=1)
            u2_2 = np.cross(v2, w2_2, axis=1)
            uvw2_1 = np.array([u2_1, v2, w2_1]).transpose(1, 2, 0)
            uvw2_2 = np.array([u2_2, v2, w2_2]).transpose(1, 2, 0)
            
            j = 1
            start_list = uvw2_1[0]
            for i in range(len(data)):
                matrix_factor = uvw2_1[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw2_1[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw2_1[i - j + k + 2] = start_list
                
            j = 1
            start_list = uvw2_2[0]
            for i in range(len(data)):
                matrix_factor = uvw2_2[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw2_2[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw2_2[i - j + k + 2] = start_list
            
            # 回転行列
            RotMatrix2_1 = np.linalg.solve(uvw2_1, XYZ)
            RotMatrix2_2 = np.linalg.solve(uvw2_2, XYZ)
            # 相対座標計算(回転後)
            RStoRE2_2 = np.matmul(RotMatrix2_1, RStoRE2_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            LStoLE2_2 = np.matmul(RotMatrix2_2, LStoLE2_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) -30<β<+170
            RE_beta2 = np.arctan2(-RStoRE2_2[:, 0], RStoRE2_2[:, 2])
            LE_beta2 = np.arctan2(-LStoLE2_2[:, 0], LStoLE2_2[:, 2]) * (-1)
            
            # 回転角γ(w軸回り) +-180
            RE_gamma2 = np.arctan2(RStoRE2_2[:, 1], RStoRE2_2[:, 2])
            LE_gamma2 = np.arctan2(LStoLE2_2[:, 1], LStoLE2_2[:, 2])
            # 可動域内にあるか判定
            RE_ToF2 = ((RE_beta2>(-30*np.pi/180)) & (RE_beta2<(170*np.pi/180)) |
                        (RE_beta2>(-100*np.pi/180)) & (RE_beta2<(-30*np.pi/180)) & (RE_gamma2>(-100*np.pi/180)) & (RE_gamma2<(0*np.pi/180)))
            LE_ToF2 = ((LE_beta2>(-30*np.pi/180)) & (LE_beta2<(170*np.pi/180)) |
                        (LE_beta2>(-100*np.pi/180)) & (LE_beta2<(-30*np.pi/180)) & (LE_gamma2>(-100*np.pi/180)) & (LE_gamma2<(0*np.pi/180)))
            # 可動域にない座標点をnanで置き換え
            RStoRE2_1[~RE_ToF2] = np.nan
            LStoLE2_1[~LE_ToF2] = np.nan
            # pandasで線形補間
            RStoRE2_1_df = pd.DataFrame(RStoRE2_1).interpolate(axis=0)
            LStoLE2_1_df = pd.DataFrame(LStoLE2_1).interpolate(axis=0)
            # 補正後の出力
            RStoRE2_1_corrected = RStoRE2_1_df.to_numpy()
            LStoLE2_1_corrected = LStoLE2_1_df.to_numpy()
            # nanを置き換え
            RStoRE2_1_corrected = np.nan_to_num(RStoRE2_1_corrected, nan=RStoRE2_1_copy)
            LStoLE2_1_corrected = np.nan_to_num(LStoLE2_1_corrected, nan=LStoLE2_1_copy)
            # 相対座標(回転前)データに格納
            re = simple_col(data, 'right_elbow')
            le = simple_col(data, 'left_elbow')
            re = pd.DataFrame(RStoRE2_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'right_shoulder')
            le = pd.DataFrame(LStoLE2_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'left_shoulder')
            
            # 3. Wrist(3, 6)
            # 基準軸uvw Elbow中心
            # u3_1, u3_2: Shoulder→Elbow
            # w3_1, w3_2: u3とMidHip→Neck軸との外積
            # v3_1, v3_2: w3とu3の外積
            REtoRW3_1 = np.array(vec(data, 'right_elbow', 'right_wrist'))
            LEtoLW3_1 = np.array(vec(data, 'left_elbow', 'left_wrist'))
            REtoRW3_1_copy = np.copy(REtoRW3_1)
            LEtoLW3_1_copy = np.copy(LEtoLW3_1)
            RStoRE3 = np.array(vec(data, 'right_shoulder', 'right_elbow'))
            LStoLE3 = np.array(vec(data, 'left_shoulder', 'left_elbow'))
            len_RStoRE3 = np.linalg.norm(RStoRE3, ord=2, axis=1).reshape(len(data), 1)
            len_LStoLE3 = np.linalg.norm(LStoLE3, ord=2, axis=1).reshape(len(data), 1)
            u3_1 = RStoRE3 / len_RStoRE3
            u3_2 = LStoLE3 / len_LStoLE3
            w3_1 = np.cross(u3_1, v1, axis=1)
            w3_2 = np.cross(u3_2, v1, axis=1)
            v3_1 = np.cross(w3_1, u3_1, axis=1)
            v3_2 = np.cross(w3_2, u3_2, axis=1)
            uvw3_1 = np.array([u3_1, v3_1, w3_1]).transpose(1, 2, 0)
            uvw3_2 = np.array([u3_2, v3_2, w3_2]).transpose(1, 2, 0)
            
            j = 1
            start_list = uvw3_1[0]
            for i in range(len(data)):
                matrix_factor = uvw3_1[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw3_1[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw3_1[i - j + k + 2] = start_list
                
            j = 1
            start_list = uvw3_2[0]
            for i in range(len(data)):
                matrix_factor = uvw3_2[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw3_2[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw3_2[i - j + k + 2] = start_list
            
            # 回転行列
            RotMatrix3_1 = np.linalg.solve(uvw3_1, XYZ)
            RotMatrix3_2 = np.linalg.solve(uvw3_2, XYZ)    
            # 相対座標計算(回転後)
            REtoRW3_2 = np.matmul(RotMatrix3_1, REtoRW3_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            LEtoLW3_2 = np.matmul(RotMatrix3_2, LEtoLW3_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) 0<β<+180
            RW_beta3 = np.arctan2(REtoRW3_2[:, 0], REtoRW3_2[:, 2])
            LW_beta3 = np.arctan2(-LEtoLW3_2[:, 0], LEtoLW3_2[:, 2]) * (-1)
            # 回転角γ(w軸回り) +-180
            # RW_gamma3 = np.arctan2(REtoRW3_2[:, 1], REtoRW3_2[:, 2])
            # LW_gamma3 = np.arctan2(LEtoLW3_2[:, 1], LEtoLW3_2[:, 2])        
            # 可動域内にあるか判定
            RW_ToF3 = (RW_beta3>(0*np.pi/180)) & (RW_beta3<(170*np.pi/180))
            LW_ToF3 = (LW_beta3>(0*np.pi/180)) & (LW_beta3<(170*np.pi/180))
            # 可動域にない座標点をnanで置き換え
            REtoRW3_1[~RW_ToF3] = np.nan
            LEtoLW3_1[~LW_ToF3] = np.nan
            # pandasで線形補間
            REtoRW3_1_df = pd.DataFrame(REtoRW3_1).interpolate(axis=0)
            LEtoLW3_1_df = pd.DataFrame(LEtoLW3_1).interpolate(axis=0)
            # 補正後の出力
            REtoRW3_1_corrected = REtoRW3_1_df.to_numpy()
            LEtoLW3_1_corrected = LEtoLW3_1_df.to_numpy()
            # nanを置き換え
            REtoRW3_1_corrected = np.nan_to_num(REtoRW3_1_corrected, nan=REtoRW3_1_copy)
            LEtoLW3_1_corrected = np.nan_to_num(LEtoLW3_1_corrected, nan=LEtoLW3_1_copy)
            # 相対座標(回転前)データに格納
            rwr = simple_col(data, 'right_wrist')
            lwr = simple_col(data, 'left_wrist')
            rwr = pd.DataFrame(REtoRW3_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'right_elbow')
            lwr = pd.DataFrame(LEtoLW3_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'left_elbow')
            
            # 4. Hip(8, 11)
            # 基準軸uvw MHip中心
            # v4: MidHip→Neck軸
            # w4: LeftHip→RightHip軸とu4の外積
            # u4: v4軸とw4軸の外積
            MHtoRH4_1 = np.array(simple_col(data, 'right_waist')- mid_waist)
            MHtoLH4_1 = np.array(simple_col(data, 'left_waist') - mid_waist)
            MHtoRH4_1_copy = np.copy(MHtoRH4_1)
            MHtoLH4_1_copy = np.copy(MHtoLH4_1)
            v4 = v1
            LHtoRH4 = np.array(vec(data, 'left_waist', 'right_waist'))
            len_LHtoRH4 = np.linalg.norm(LHtoRH4, ord=2, axis=1).reshape(len(data), 1)
            LHtoRH4_norm = LHtoRH4 / len_LHtoRH4
            w4 = np.cross(LHtoRH4_norm, v4, axis=1)
            u4 = np.cross(v4, w4, axis=1)
            uvw4 = np.array([u4, v4, w4]).transpose(1, 2, 0)
            
            j = 1
            start_list = uvw4[0]
            for i in range(len(data)):
                matrix_factor = uvw4[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw4[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw4[i - j + k + 2] = start_list
            
            # 回転行列
            RotMatrix4 = np.linalg.solve(uvw4, XYZ)
            # 相対座標計算(回転後)
            MHtoRH4_2 = np.matmul(RotMatrix4, MHtoRH4_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            MHtoLH4_2 = np.matmul(RotMatrix4, MHtoLH4_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) -20<β<+20
            RH_beta4 = np.arctan2(-MHtoRH4_2[:, 0], MHtoRH4_2[:, 2])
            LH_beta4 = np.arctan2(-MHtoLH4_2[:, 0], -MHtoLH4_2[:, 2])
            # 回転角γ(w軸回り) -10<γ<+10
            RH_gamma4 = np.arctan2(MHtoRH4_2[:, 1], MHtoRH4_2[:, 2])
            LH_gamma4 = np.arctan2(MHtoLH4_2[:, 1], -MHtoLH4_2[:, 2])
            # 可動域内にあるか判定
            RH_ToF4 = (RH_beta4>(-20*np.pi/180)) & (RH_beta4<(20*np.pi/180)) & (RH_gamma4>(-10*np.pi/180)) & (RH_gamma4<(10*np.pi/180))
            LH_ToF4 = (LH_beta4>(-20*np.pi/180)) & (LH_beta4<(20*np.pi/180)) & (LH_gamma4>(-10*np.pi/180)) & (LH_gamma4<(10*np.pi/180))
            # 可動域にない座標点をnanで置き換え
            MHtoRH4_1[~RH_ToF4] = np.nan
            MHtoLH4_1[~LH_ToF4] = np.nan
            # pandasで線形補間
            MHtoRH4_1_df = pd.DataFrame(MHtoRH4_1).interpolate(axis=0)
            MHtoLH4_1_df = pd.DataFrame(MHtoLH4_1).interpolate(axis=0)
            # 補正後の出力
            MHtoRH4_1_corrected = MHtoRH4_1_df.to_numpy()
            MHtoLH4_1_corrected = MHtoLH4_1_df.to_numpy()
            # nanを置き換え
            MHtoRH4_1_corrected = np.nan_to_num(MHtoRH4_1_corrected, nan=MHtoRH4_1_copy)
            MHtoLH4_1_corrected = np.nan_to_num(MHtoLH4_1_corrected, nan=MHtoLH4_1_copy)
            # 相対座標(回転前)データに格納
            rwa = simple_col(data, 'right_waist')
            lwa = simple_col(data, 'left_waist')
            rwa = pd.DataFrame(MHtoRH4_1_corrected, columns = ['x', 'y', 'z']) + mid_waist
            lwa = pd.DataFrame(MHtoLH4_1_corrected, columns = ['x', 'y', 'z']) + mid_waist
            
            # 5. Knee(9, 12)
            # 基準軸uvw RLHip中心
            # v5: MidHip→Neck軸
            # w5_1, w5_2: MHip→RLHipとv5軸の外積
            # u5_1, u5_2: v軸とw1, w2軸の外積
            RHtoRK5_1 = np.array(vec(data, 'right_waist', 'right_knee'))
            LHtoLK5_1 = np.array(vec(data, 'left_waist', 'left_knee'))
            RHtoRK5_1_copy = np.copy(RHtoRK5_1)
            LHtoLK5_1_copy = np.copy(LHtoLK5_1)
            v5 = v1
            MHtoRH5 = np.array(simple_col(data, 'right_waist') - mid_waist)
            MHtoLH5 = np.array(simple_col(data, 'left_waist') - mid_waist)
            len_MHtoRH5 = np.linalg.norm(MHtoRH5, ord=2, axis=1).reshape(len(data), 1)
            len_MHtoLH5 = np.linalg.norm(MHtoLH5, ord=2, axis=1).reshape(len(data), 1)
            MHtoRH5_norm = MHtoRH5 / len_MHtoRH5
            MHtoLH5_norm = MHtoLH5 / len_MHtoLH5
            w5_1 = np.cross(MHtoRH5_norm, v5, axis=1)
            w5_2 = np.cross(MHtoLH5_norm, v5, axis=1)
            u5_1 = np.cross(v5, w5_1, axis=1)
            u5_2 = np.cross(v5, w5_2, axis=1)
            uvw5_1 = np.array([u5_1, v5, w5_1]).transpose(1, 2, 0)
            uvw5_2 = np.array([u5_2, v5, w5_2]).transpose(1, 2, 0)
            
            j = 1
            start_list = uvw5_1[0]
            for i in range(len(data)):
                matrix_factor = uvw5_1[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw5_1[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw5_1[i - j + k + 2] = start_list
                
            j = 1
            start_list = uvw5_2[0]
            for i in range(len(data)):
                matrix_factor = uvw5_2[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw5_2[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw5_2[i - j + k + 2] = start_list
            
            # 回転行列
            RotMatrix5_1 = np.linalg.solve(uvw5_1, XYZ)
            RotMatrix5_2 = np.linalg.solve(uvw5_2, XYZ)
            # 相対座標計算(回転後)
            RHtoRK5_2 = np.matmul(RotMatrix5_1, RHtoRK5_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            LHtoLK5_2 = np.matmul(RotMatrix5_2, LHtoLK5_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) 0<β<+120
            RK_beta5 = np.arctan2(-RHtoRK5_2[:, 0], RHtoRK5_2[:, 2])*(-1)
            LK_beta5 = np.arctan2(LHtoLK5_2[:, 0], -LHtoLK5_2[:, 2])
            '''
            Rk = pd.DataFrame(RK_beta5)
            Lk = pd.DataFrame(LK_beta5)
            KA = pd.concat([Rk,Lk], axis = 1)
            KA.to_csv('KA.csv')
            '''

            # 回転角γ(w軸回り) +-180
            # RK_gamma5 = np.arctan2(RHtoRK5_2[:, 1], RHtoRK5_2[:, 2])
            # LK_gamma5 = np.arctan2(LHtoLK5_2[:, 1], LHtoLK5_2[:, 2])
            # 可動域内にあるか判定
            RK_ToF5 = (RK_beta5>(0*np.pi/180)) & (RK_beta5<(120*np.pi/180))
            LK_ToF5 = (LK_beta5>(0*np.pi/180)) & (LK_beta5<(120*np.pi/180))
            # 可動域にない座標点をnanで置き換え
            RHtoRK5_1[~RK_ToF5] = np.nan
            LHtoLK5_1[~LK_ToF5] = np.nan
            # pandasで線形補間
            RHtoRK5_1_df = pd.DataFrame(RHtoRK5_1).interpolate(axis=0)
            LHtoLK5_1_df = pd.DataFrame(LHtoLK5_1).interpolate(axis=0)
            # 補正後の出力
            RHtoRK5_1_corrected = RHtoRK5_1_df.to_numpy()
            LHtoLK5_1_corrected = LHtoLK5_1_df.to_numpy()
            # nanを置き換え
            RHtoRK5_1_corrected = np.nan_to_num(RHtoRK5_1_corrected, nan=RHtoRK5_1_copy)
            LHtoLK5_1_corrected = np.nan_to_num(LHtoLK5_1_corrected, nan=LHtoLK5_1_copy)
            # 相対座標(回転前)データに格納
            rk = simple_col(data, 'right_knee')
            lk = simple_col(data, 'left_knee')
            rk = pd.DataFrame(RHtoRK5_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'right_waist')
            lk = pd.DataFrame(LHtoLK5_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'left_waist')
            
            # 6. foot(10, 13)
            # 基準軸uvw Elbow中心
            # u6_1, u6_2: RLHip→Knee
            # w6_1, w6_2: u6とMidHip→Neck軸との外積
            # v6_1, v6_2: w6とu6の外積
            RKtoRA6_1 = np.array(vec(data, 'right_knee', 'right_foot'))
            LKtoLA6_1 = np.array(vec(data, 'left_knee', 'left_foot'))
            RKtoRA6_1_copy = np.copy(RKtoRA6_1)
            LKtoLA6_1_copy = np.copy(LKtoLA6_1)
            RHtoRK6 = np.array(vec(data, 'right_waist', 'right_knee'))
            LHtoLK6 = np.array(vec(data, 'left_waist', 'left_knee'))
            len_RHtoRK6 = np.linalg.norm(RHtoRK6, ord=2, axis=1).reshape(len(data), 1)
            len_LHtoLK6 = np.linalg.norm(LHtoLK6, ord=2, axis=1).reshape(len(data), 1)
            u6_1 = RHtoRK6 / len_RHtoRK6
            u6_2 = LHtoLK6 / len_LHtoLK6
            w6_1 = np.cross(u6_1, v1, axis=1)
            w6_2 = np.cross(u6_2, v1, axis=1)
            v6_1 = np.cross(w6_1, u6_1, axis=1)
            v6_2 = np.cross(w6_2, u6_2, axis=1)
            uvw6_1 = np.array([u6_1, v6_1, w6_1]).transpose(1, 2, 0)
            uvw6_2 = np.array([u6_2, v6_2, w6_2]).transpose(1, 2, 0)
            
            j = 1
            start_list = uvw6_1[0]
            for i in range(len(data)):
                matrix_factor = uvw6_1[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw6_1[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw6_1[i - j + k + 2] = start_list
                
            j = 1
            start_list = uvw6_2[0]
            for i in range(len(data)):
                matrix_factor = uvw6_2[i]
                matrix_size = np.linalg.det(matrix_factor)
                if matrix_size == 0:
                    j += 1
                
                else:
                    last_list = matrix_factor
                    for k in range(j):
                        uvw6_2[i - j + k + 1] = ((j - k - 1) * start_list + (k + 1) * last_list)/j
                        start_list = matrix_factor
                    j = 1
            
            if not j == 1:
                for l in range(j - 1):
                    uvw6_2[i - j + k + 2] = start_list
            
            # 回転行列
            RotMatrix6_1 = np.linalg.solve(uvw6_1, XYZ)
            RotMatrix6_2 = np.linalg.solve(uvw6_2, XYZ)
            # 相対座標計算(回転後)
            RKtoRA6_2 = np.matmul(RotMatrix6_1, RKtoRA6_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            LKtoLA6_2 = np.matmul(RotMatrix6_2, LKtoLA6_1.reshape(len(data), 3, 1)).reshape(len(data), 3)
            # 回転角β(v軸回り) +-180
            # RA_beta6 = np.arctan2(-RKtoRA6_2[:, 0], RKtoRA6_2[:, 2])
            # LA_beta6 = np.arctan2(-LKtoLA6_2[:, 0], LKtoLA6_2[:, 2]) * (-1)
            # 回転角γ(w軸回り) 0<γ<+180
            RA_gamma6 = np.arctan2(RKtoRA6_2[:, 1], RKtoRA6_2[:, 2])
            LA_gamma6 = np.arctan2(LKtoLA6_2[:, 1], LKtoLA6_2[:, 2])        
            # 可動域内にあるか判定
            RA_ToF6 = (RA_gamma6>(-180*np.pi/180)) & (RA_gamma6<(0*np.pi/180))
            LA_ToF6 = (LA_gamma6>(-180*np.pi/180)) & (LA_gamma6<(0*np.pi/180))
            # 可動域にない座標点をnanで置き換え
            RKtoRA6_1[~RA_ToF6] = np.nan
            LKtoLA6_1[~LA_ToF6] = np.nan
            # pandasで線形補間
            RKtoRA6_1_df = pd.DataFrame(RKtoRA6_1).interpolate(axis=0)
            LKtoLA6_1_df = pd.DataFrame(LKtoLA6_1).interpolate(axis=0)
            # 補正後の出力
            RKtoRA6_1_corrected = RKtoRA6_1_df.to_numpy()
            LKtoLA6_1_corrected = LKtoLA6_1_df.to_numpy()
            # nanを置き換え
            RKtoRA6_1_corrected = np.nan_to_num(RKtoRA6_1_corrected, nan=RKtoRA6_1_copy)
            LKtoLA6_1_corrected = np.nan_to_num(LKtoLA6_1_corrected, nan=LKtoLA6_1_copy)
            # 相対座標(回転前)データに格納
            rf = simple_col(data, 'right_foot')
            lf = simple_col(data, 'left_foot')
            rf = pd.DataFrame(RKtoRA6_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'right_knee')
            lf = pd.DataFrame(LKtoLA6_1_corrected, columns = ['x', 'y', 'z']) + simple_col(data, 'left_knee')

            #head = data.loc[:, 'head']
            neck = data.loc[:, 'neck']
            data2 = -1*pd.concat([head, neck,
                                rs, re, rwr,
                                ls, le, lwr,
                                rwa, rk, rf,
                                lwa, lk, lf,
                                ], axis = 1)
            
            return data2

        data2 = Motion_Range(df)
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
            'left_foot'
        ]
        separator = ('_x','_y','_z')
        header_new = []
        for keypoint in header_keypoints:
            for sep in separator:
                header_new.append( keypoint + sep )
        data2.columns = header_new
        df_anglec = pd.concat([df_count, data2, other], axis = 1)
        # print(df_anglec)
        # os.makedirs(rf'range_correction/{n}', exist_ok = True)
        # os.makedirs(f'Data/Output/range_correction/', exist_ok = True)
        df_anglec.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta\{filename}_range_correction.csv', index = False)
        
        new_data = pd.DataFrame([[str(filename) , ' ' , str(filename) + '_length.csv']] , columns=['filename' , 'subject_id' , 'lengthfile'])
        jedge_data = length_data.iloc[:,0]
        if not (filename == jedge_data).any():
            length_data = length_data.append(new_data , ignore_index=True)

length_data = sort.sort(length_data, 'filename')

length_data.index.name = 'id'
length_data.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\base\LENGTH_WORKSET.csv', index=True)