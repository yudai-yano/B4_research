import pandas as pd
import os
import glob
import sys
sys.path.append(rf'C:\Users\yota0\Desktop\yano\program\python\definition')
import make_trcfile

#lengthデータ
length_folder = rf'C:\Users\yota0\Desktop\yano\program\python\OPENSIM\out'

for length_file in glob.glob(rf'{length_folder}\*'):
    #filenameに実験の名前を代入
    filepath = os.path.split(os.path.basename(length_file))[-1]
    filename = filepath.replace("_length.csv", "")
    #length.csvデータ以外を処理しないように
    if not filepath.endswith("_length.csv"):
        pass
    else:
        #length.csvデータを入力
        length_data = pd.read_csv(length_file, header=0)
        body_data = length_data.loc[:, 'NOSE_x':'RIGHT_FOOT_INDEX_z']

        result = pd.DataFrame()
        result['time'] = length_data['time']
        
        #最初から動作がどの程度変化したかの値を入力
        motion_length_coordinate_x = length_data['motion_x']
        motion_length_coordinate_y = length_data['motion_y']
        motion_length_coordinate_z = length_data['motion_z']

        #各座標ごとに動作時の座標に変換
        column_names = body_data.columns.tolist()
        for column in column_names:
            if '_x' in column:
                coordinate_data = length_data[column] + motion_length_coordinate_x
                
            if '_y' in column:
                coordinate_data = length_data[column] + motion_length_coordinate_y
                
            if '_z' in column:
                coordinate_data = length_data[column] + motion_length_coordinate_z
                
            result[column] = coordinate_data
            
        csv_file = rf'C:\Users\yota0\Desktop\yano\program\python\OPENSIM\out\{filename}_motion.csv'
        trc_file = rf'D:\yano\OPENSIM\input_data\{filename}\{filename}_motion.trc'
        #csvデータを出力
        result.to_csv(csv_file, index=False)
        #trcデータを出力
        motion_file = make_trcfile.csv_to_trc(csv_file, trc_file)
