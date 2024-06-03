import pandas as pd
import os
import glob
from collections import namedtuple

folderpath = rf'C:\\Users\\yota0\\Documents\\Yota\\githubrepos\\poseestimate_mediapipe\\poseestimate_mediapipe\\out\\modelbased'

for file_name in glob.glob(rf'{folderpath}\*'):
    filepath = os.path.split(os.path.basename(file_name))[-1]
    
    if not filepath.endswith(".csv"):
        pass
    else:
        filename = filepath.replace("_correct_modelbased.csv", "")
        foot_kakudo = pd.DataFrame()
        csv_frame = pd.read_csv(f'{file_name}', header=1)
        data_frame = csv_frame.loc[:, 'LEFT_HIP_x':'RIGHT_FOOT_INDEX_z']
        
        print(filename)
        
        #column_names = list(data_frame.columns)
        
        left_V1_xy = 