import pandas as pd
import os
import glob
import numpy as np

folderpath = rf'C:\\Users\\yota0\\Documents\\Yota\\githubrepos\\poseestimate_mediapipe\\poseestimate_mediapipe\\out\\modelbased'

for file_name in glob.glob(rf'{folderpath}\*'):
    filepath = os.path.split(os.path.basename(file_name))[-1]
    
    if not filepath.endswith(".csv"):
        pass
    else:
        filename = filepath.replace("_correct_modelbased.csv", "")
        df = pd.DataFrame()
        csv_frame = pd.read_csv(f'{file_name}', header=1)
        data_frame = csv_frame.loc[:, 'LEFT_WRIST_x':'RIGHT_WRIST_z']
        add_frame = csv_frame.loc[:, 'LEFT_HIP_x':'RIGHT_KNEE_z']
        data_frame = pd.concat([data_frame,add_frame], axis=1)
        
        print(filename)
        load = (input("what is load"))
        
        #バーの中心を求める
        bar_middle_x = (data_frame['LEFT_WRIST_x'] + data_frame['RIGHT_WRIST_x'])/2
        bar_middle_y = (data_frame['LEFT_WRIST_y'] + data_frame['RIGHT_WRIST_y'])/2
        bar_middle_z = (data_frame['LEFT_WRIST_z'] + data_frame['RIGHT_WRIST_z'])/2
        
        #太ももの位置を推定
        left_thigh_x = (data_frame['LEFT_HIP_x'] + data_frame['LEFT_KNEE_x'])/2
        left_thigh_y = (data_frame['LEFT_HIP_y'] + data_frame['LEFT_KNEE_y'])/2
        left_thigh_z = (data_frame['LEFT_HIP_z'] + data_frame['LEFT_KNEE_z'])/2
        right_thigh_x = (data_frame['RIGHT_HIP_x'] + data_frame['RIGHT_KNEE_x'])/2
        right_thigh_y = (data_frame['RIGHT_HIP_y'] + data_frame['RIGHT_KNEE_y'])/2
        right_thigh_z = (data_frame['RIGHT_HIP_z'] + data_frame['RIGHT_KNEE_z'])/2
        
        #手首から太ももの距離→バーから太ももの推定
        left_range_x = abs(data_frame['LEFT_WRIST_x'] - left_thigh_x)
        left_range_y = abs(data_frame['LEFT_WRIST_y'] - left_thigh_y)
        left_range_z = abs(data_frame['LEFT_WRIST_z'] - left_thigh_z)
        right_range_x = abs(data_frame['RIGHT_WRIST_x'] - right_thigh_x)
        right_range_y = abs(data_frame['RIGHT_WRIST_y'] - right_thigh_y)
        right_range_z = abs(data_frame['RIGHT_WRIST_z'] - right_thigh_z)
        
        #xz軸上の距離と全体の距離
        left_range_xz = np.sqrt(left_range_x**2 + left_range_z**2)
        right_range_xz = np.sqrt(right_range_x**2 + right_range_z**2)
        left_range = np.sqrt(left_range_x**2 + left_range_y**2 + left_range_z**2)
        right_range = np.sqrt(right_range_x**2 + right_range_y**2 + right_range_z**2)
        
        #バーから太ももまでの角度
        left_thigh_angle = np.arccos(left_range_y/left_range)
        right_thigh_angle = np.arccos(right_range_y/right_range)
        
        left_real_load = [float(load) * np.cos(y) /2 for y in left_thigh_angle]
        right_real_load = [float(load) * np.cos(y) /2 for y in right_thigh_angle]
        
        df['left_thigh_angle'] = left_thigh_angle
        df['right_thigh_angle'] = right_thigh_angle
        df['left_thigh_load'] = left_real_load
        df['right_thigh_load'] = right_real_load
        
        df.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\out\{filename}_thigh_to_load.csv', index=True)
        