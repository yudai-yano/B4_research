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
        quadriceps_muscle = pd.DataFrame()
        biceps_femoris_muscle = pd.DataFrame()
        csv_frame = pd.read_csv(f'{file_name}', header=1)
        data_frame = csv_frame.loc[:, 'LEFT_SHOULDER_x':'RIGHT_SHOULDER_z']
        
        print(filename)
        
        column_names = list(data_frame.columns)
        
        for title in column_names:
            csv_data = data_frame[title]
            last_number = len(csv_data)
            model4 = []
            model2 = []
            model4.append(0)
            model2.append(0)
            for i in range(1,last_number):
                i_data = csv_data[i]
                i1_data = csv_data[i-1]
                
                
                V = (i_data - i1_data)*30
                
                if "_y" in title:
                    if V < 0:
                        model4.append(0)
                        model2.append(-V)
                    else:
                        model4.append(V)
                        model2.append(0)
                else:
                    model2.append(V)
                    model4.append(V)

            Mytuple4 = tuple(model4)
            Mytuple2 = tuple(model2)
            quadriceps_muscle[title] = Mytuple4
            biceps_femoris_muscle[title] = Mytuple2
            
        quadriceps_muscle.to_csv(rf'C:\Users\yota0\Desktop\yano\out\{filename}_quadriceps_muscle.csv', index=True)
        biceps_femoris_muscle.to_csv(rf'C:\Users\yota0\Desktop\yano\out\{filename}_biceps_femoris_muscle.csv', index=True)
        