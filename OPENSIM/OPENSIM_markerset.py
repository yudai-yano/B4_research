import pandas as pd
import os
import sys
import math
import glob
sys.path.append(rf'C:\Users\yota0\Desktop\yano\program\python\definition')
import make_trcfile

exception = [0 , 1 , -1]

#modelbasedデータ
modelbased_folder = rf"C:\Users\yota0\Documents\Yota\githubrepos\poseestimate_mediapipe\poseestimate_mediapipe\out\modelbased\old"

for modelbased_file in glob.glob(rf'{modelbased_folder}\*'):
    #filenameに実験の名前を代入
    filepath = os.path.split(os.path.basename(modelbased_file))[-1]
    filename = filepath.replace("_correct_modelbased.csv", "")
    #csvデータ以外を処理しないように
    if not filepath.endswith("_correct_modelbased.csv"):
        pass
    else:
        result = pd.DataFrame()
        motion = pd.DataFrame()
        #modelbaseデータの入力
        modelbased_data = pd.read_csv(modelbased_file, header=1)
        coordinate_data = modelbased_data.loc[:, 'NOSE_x':'RIGHT_FOOT_INDEX_z']
        modelbased_row = len(modelbased_data)
        #時間の単位を秒に変換
        time_list = []
        for frame in range(modelbased_row):
            time = frame * (1/30)
            time_correct = round(time, 2)
            time_list.append(time_correct)
            
        result['time'] = time_list
        motion['time'] = time_list

        #modelbasedデータの処理
        #マーカー基準点の設定→ここでは腰あたりに設定
        standerd_coordinate_x = (modelbased_data.loc[:, 'RIGHT_HIP_x'] + modelbased_data.loc[:, 'LEFT_HIP_x'])/2
        standerd_coordinate_y = (modelbased_data.loc[:, 'RIGHT_HIP_y'] + modelbased_data.loc[:, 'LEFT_HIP_y'])/2
        standerd_coordinate_z = (modelbased_data.loc[:, 'RIGHT_HIP_z'] + modelbased_data.loc[:, 'LEFT_HIP_z'])/2
        
        #各データをマーカー座標に置き換え
        column_names = coordinate_data.columns.tolist()
        for item in column_names:
            if "_x" in item:
                difference_length = (coordinate_data.loc[:,item] - standerd_coordinate_x)*1000

            if "_y" in item:
                difference_length = -(coordinate_data.loc[:,item] - standerd_coordinate_y)*1000

            if "_z" in item:
                difference_length = -(coordinate_data.loc[:,item] - standerd_coordinate_z)*1000
            
            result[item] = difference_length
            motion[item] = difference_length
        
        #最初の腰の位置を入力→変な値ならその次のフレームのデータ
        standerd_HIP_x = standerd_coordinate_x.loc[0]
        for i in range(10000):
            if standerd_HIP_x in exception:
                standerd_HIP_x = standerd_coordinate_x.loc[i]
            else:
                pass
        standerd_HIP_y = standerd_coordinate_y.loc[0]
        for i in range(10000):
            if standerd_HIP_y in exception:
                standerd_HIP_y = standerd_coordinate_y.loc[i]
            else:
                pass
        standerd_HIP_z = standerd_coordinate_z.loc[0]
        for i in range(10000):
            if standerd_HIP_z in exception:
                standerd_HIP_z = standerd_coordinate_z.loc[i]
            else:
                pass
        
        #動作データの基準の座標を入力→変な値ならその次のフレームのデータ
        motion_coordinate_x = (result.loc[0, 'LEFT_HEEL_x'] + result.loc[0, 'RIGHT_HEEL_x'])/2
        for i in range(10000):
            if motion_coordinate_x in exception:
                motion_coordinate_x = (result.loc[i, 'LEFT_HEEL_x'] + result.loc[i, 'RIGHT_HEEL_x'])/2
            else:
                pass
        motion_coordinate_y = (result.loc[0, 'LEFT_HEEL_y'] + result.loc[0, 'RIGHT_HEEL_y'])/2
        for i in range(10000):
            if motion_coordinate_y in exception:
                motion_coordinate_y = (result.loc[i, 'LEFT_HEEL_y'] + result.loc[i, 'RIGHT_HEEL_y'])/2
            else:
                pass
        motion_coordinate_z = (result.loc[0, 'LEFT_HEEL_z'] + result.loc[0, 'RIGHT_HEEL_z'])/2
        for i in range(10000):
            if motion_coordinate_z in exception:
                motion_coordinate_z = (result.loc[i, 'LEFT_HEEL_z'] + result.loc[i, 'RIGHT_HEEL_z'])/2
            else:
                pass
        
        motion_data_x = []
        motion_data_y = []
        motion_data_z = []
        
        #それぞれのフレームで腰の位置が基準と比べてどの程度変化したか
        for j in range(modelbased_row):
            motion_HIP_x = (standerd_coordinate_x[j] - standerd_HIP_x)
            motion_correct_HIP_x = motion_HIP_x * 1000 - motion_coordinate_x + 2000
            motion_data_x.append(motion_correct_HIP_x)
            
            motion_HIP_y = -(standerd_coordinate_y[j] - standerd_HIP_y)
            motion_correct_HIP_y = motion_HIP_y * 1000 - motion_coordinate_y
            motion_data_y.append(motion_correct_HIP_y)
            
            motion_HIP_z = -(standerd_coordinate_z[j] - standerd_HIP_z)
            motion_correct_HIP_z = motion_HIP_z * 1000 - motion_coordinate_z
            motion_data_z.append(motion_correct_HIP_z)
            
        result['motion_x'] = motion_data_x
        result['motion_y'] = motion_data_y
        result['motion_z'] = motion_data_z
            

        '''
        文字の方→こっちは出力できる
        a = data_frame[:user_input]
        数字の方→こっちは出力できない
        b = data_frame[:selected_value]

        print(selected_value)

        '''

        #OPENSIMのフォルダに被験者の名前のフォルダを検索→なければ作成
        folder_path = rf"D:\yano\OPENSIM\input_data\{filename}"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        #csvデータとtrcデータのpathを指定
        motion_csv = rf'C:\Users\yota0\Desktop\yano\program\python\OPENSIM\out\{filename}_markerset.csv'
        motion_trc = rf'D:\yano\OPENSIM\input_data\{filename}\{filename}_markerset.trc'
        #csvデータの作成
        result.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\OPENSIM\out\{filename}_length.csv', index=False)
        motion.to_csv(motion_csv,index=False)
        #trcファイルの作成
        make_trcfile.csv_to_trc(motion_csv, motion_trc)