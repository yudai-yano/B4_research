import pandas as pd
import os
import glob
from collections import namedtuple
import numpy as np
import Vector_Direction_Distinction as vd
import angle

folderpath = rf'C:\Users\yota0\Documents\Yota\githubrepos\poseestimate_mediapipe\poseestimate_mediapipe\out\modelbased'

mass = 65.3
load = 61

for file_name in glob.glob(rf'{folderpath}\*'):
    filepath = os.path.split(os.path.basename(file_name))[-1]
    
    if not filepath.endswith(".csv"):
        pass
    else:
        filename = filepath.replace("_correct_modelbased.csv", "")
        foot_kakudo = pd.DataFrame()
        foot_kakudo2 = pd.DataFrame()
        foot_kakudo3 = pd.DataFrame()
        new_columns = ['left_e1_xy','left_e2_xy','left_e3_xy','left_f1_xy','left_f2_xy','left_f3_xy','left_e1_zy','left_e2_zy','left_e3_zy','left_f1_zy','left_f2_zy','left_f3_zy']
        new_columns2 = ['right_e1','right_e2','right_e3','right_f1','right_f2','right_f3']
        my_list_x = []
        my_list_y = []
        my_list_z = []
        plate_press_x = []
        plate_press_y = []
        plate_press_z = []
        press_power_xy = []
        press_power_zy = []
        csv_frame = pd.read_csv(rf'{file_name}', header=1)
        data_frame = csv_frame.loc[:, 'LEFT_HIP_x':'RIGHT_FOOT_INDEX_z']
        
        #print(filename)
        
        #column_names = list(data_frame.columns)
        
        #尻から足のかかとまでのベクトル、D1
        left_foot_x1 = data_frame['LEFT_HIP_x'] - data_frame['LEFT_HEEL_x']
        left_foot_y1 = data_frame['LEFT_HIP_y'] - data_frame['LEFT_HEEL_y']
        left_foot_z1 = data_frame['LEFT_HIP_z'] - data_frame['LEFT_HEEL_z']
        right_foot_x1 = data_frame['RIGHT_HIP_x'] - data_frame['RIGHT_HEEL_x']
        right_foot_y1 = data_frame['RIGHT_HIP_y'] - data_frame['RIGHT_HEEL_y']
        right_foot_z1 = data_frame['RIGHT_HIP_z'] - data_frame['RIGHT_HEEL_z']
        
        #膝から足のかかとまでのベクトル、D2
        left_foot_x2 = data_frame['LEFT_KNEE_x'] - data_frame['LEFT_HEEL_x']
        left_foot_y2 = data_frame['LEFT_KNEE_y'] - data_frame['LEFT_HEEL_y']
        left_foot_z2 = data_frame['LEFT_KNEE_z'] - data_frame['LEFT_HEEL_z']
        right_foot_x2 = data_frame['RIGHT_KNEE_x'] - data_frame['RIGHT_HEEL_x']
        right_foot_y2 = data_frame['RIGHT_KNEE_y'] - data_frame['RIGHT_HEEL_y']
        right_foot_z2 = data_frame['RIGHT_KNEE_z'] - data_frame['RIGHT_HEEL_z']
        
        #尻から膝までのベクトル、D3
        left_foot_x3 = data_frame['LEFT_KNEE_x'] - data_frame['LEFT_HIP_x']
        left_foot_y3 = data_frame['LEFT_KNEE_y'] - data_frame['LEFT_HIP_y']
        left_foot_z3 = data_frame['LEFT_KNEE_z'] - data_frame['LEFT_HIP_z']
        right_foot_x3 = data_frame['RIGHT_KNEE_x'] - data_frame['RIGHT_HIP_x']
        right_foot_y3 = data_frame['RIGHT_KNEE_y'] - data_frame['RIGHT_HIP_y']
        right_foot_z3 = data_frame['RIGHT_KNEE_z'] - data_frame['RIGHT_HIP_z']
        
        foot_kakudo['left_foot_x1'] = left_foot_x1
        foot_kakudo['left_foot_y1'] = left_foot_y1
        foot_kakudo['left_foot_z1'] = left_foot_z1
        foot_kakudo['right_foot_x1'] = right_foot_x1
        foot_kakudo['right_foot_y1'] = right_foot_y1
        foot_kakudo['right_foot_z1'] = right_foot_z1
        foot_kakudo['left_foot_x2'] = left_foot_x2
        foot_kakudo['left_foot_y2'] = left_foot_y2
        foot_kakudo['left_foot_z2'] = left_foot_z2
        foot_kakudo['right_foot_x2'] = right_foot_x2
        foot_kakudo['right_foot_y2'] = right_foot_y2
        foot_kakudo['right_foot_z2'] = right_foot_z2
        foot_kakudo['left_foot_x3'] = left_foot_x3
        foot_kakudo['left_foot_y3'] = left_foot_y3
        foot_kakudo['left_foot_z3'] = left_foot_z3
        foot_kakudo['right_foot_x3'] = right_foot_x3
        foot_kakudo['right_foot_y3'] = right_foot_y3
        foot_kakudo['right_foot_z3'] = right_foot_z3
        
        
        #トルクから先端の力を求める
        #file_name2 = rf'C:\Users\yota0\Desktop\kamiya\program_copy\out\moment\kikuchi_70per_1_Knee_moment.csv'
        #file_name3 = rf'C:\Users\yota0\Desktop\kamiya\program_copy\out\moment\kikuchi_70per_1_Waist_moment.csv'
        #Knee_moment = pd.read_csv(rf'{file_name2}',header=0)
        #Waist_moment = pd.read_csv(rf'{file_name3}',header=0)
        
        #床反力
        #Data2以降これ取ってないから直してね
        file_name2 = rf'D:\nittai_force_plate\11_09\福万啓介\fukuman_70per_2_2022_1109_130610.csv'
        csv_frame2 = pd.read_csv(rf'{file_name2}', header=0, encoding='shift-jis')
        data_frame2 = csv_frame2.loc[:, 'Fx(FP1)[N]':'Fz(FP1)[N]']
        
        plate1_press_x = data_frame2['Fy(FP1)[N]']
        plate1_press_y = data_frame2['Fz(FP1)[N]'] - (mass + load) * 9.8
        plate1_press_z = data_frame2['Fx(FP1)[N]']
        
        
        row1 = len(left_foot_x1)
        #force_piateの方が先に起動されてるならそのフレーム数を入れる、なければ0
        j = 0
        #modelbasedの方が先に起動されてるならそのフレーム数を入れる、なければ0
        k = 1614
        #変更NG
        l = 1
        row = len(plate1_press_x)
        row2 = int(j  * (1000 / 30))
        
        if k == 0:
            pass
        
        else:
            for i in range(k):
                plate_press_x.append(0)
                plate_press_y.append(0)
                plate_press_z.append(0)
            
        for i in range(row2 , row):
            if l + k > row1:
                pass
            
            elif i == row - 1:
                my_list_x.append(plate1_press_x[i])
                my_list_y.append(plate1_press_y[i])
                my_list_z.append(plate1_press_z[i])
                plate_press_x.append(sum(my_list_x)/len(my_list_x))
                plate_press_y.append(sum(my_list_y)/len(my_list_y))
                plate_press_z.append(sum(my_list_z)/len(my_list_z))
                l += 1
                j += 1
                
            elif i < int(1000*(j + 1) / 30):
                my_list_x.append(plate1_press_x[i])
                my_list_y.append(plate1_press_y[i])
                my_list_z.append(plate1_press_z[i])
                
            else:
                my_list_x.append(plate1_press_x[i])
                my_list_y.append(plate1_press_y[i])
                my_list_z.append(plate1_press_z[i])
                plate_press_x.append(sum(my_list_x)/len(my_list_x))
                plate_press_y.append(sum(my_list_y)/len(my_list_y))
                plate_press_z.append(sum(my_list_z)/len(my_list_z))
                l += 1
                j += 1
                my_list_x = []
                my_list_y = []
                my_list_z = []
        
        if (l + k) < (row1 + 1):
            for i in range(l + k , row1 + 1):
                plate_press_x.append(0)
                plate_press_y.append(0)
                plate_press_z.append(0)
        
        press_kakudo_xy = np.arctan2(plate_press_y , plate_press_x)
        press_kakudo_zy = np.arctan2(plate_press_y , plate_press_z)
        
        for i in range(row1):
            press_x_real = plate_press_x[i]
            press_y_real = plate_press_y[i]
            press_z_real = plate_press_z[i]
            real_press_squared_xy = press_x_real**2 + press_y_real**2
            real_press_real_xy = np.sqrt(real_press_squared_xy)
            press_power_xy.append(real_press_real_xy)
            real_press_squared_zy = press_z_real**2 + press_y_real**2
            real_press_real_zy = np.sqrt(real_press_squared_xy)
            press_power_zy.append(real_press_real_xy)
        #床反力終了
        
        #D1の角度
        left_D1_kakudo_xy = np.arctan2(foot_kakudo['left_foot_y1'].tolist(),foot_kakudo['left_foot_x1'].tolist())
        left_D1_kakudo_zy = np.arctan2(foot_kakudo['left_foot_y1'].tolist(),foot_kakudo['left_foot_z1'].tolist())
        right_D1_kakudo_xy = np.arctan2(foot_kakudo['right_foot_y1'].tolist(),foot_kakudo['right_foot_x1'].tolist())
        right_D1_kakudo_zy = np.arctan2(foot_kakudo['right_foot_y1'].tolist(),foot_kakudo['right_foot_z1'].tolist())
        
        #D2の角度
        left_D2_kakudo_xy = np.arctan2(foot_kakudo['left_foot_y2'].tolist(),foot_kakudo['left_foot_x2'].tolist())
        left_D2_kakudo_zy = np.arctan2(foot_kakudo['left_foot_y2'].tolist(),foot_kakudo['left_foot_z2'].tolist())
        right_D2_kakudo_xy = np.arctan2(foot_kakudo['right_foot_y2'].tolist(),foot_kakudo['right_foot_x2'].tolist())
        right_D2_kakudo_zy = np.arctan2(foot_kakudo['right_foot_y2'].tolist(),foot_kakudo['right_foot_z2'].tolist())
        
        #D3の角度
        left_D3_kakudo_xy = np.arctan2(foot_kakudo['left_foot_y3'].tolist(),foot_kakudo['left_foot_x3'].tolist())
        left_D3_kakudo_zy = np.arctan2(foot_kakudo['left_foot_y3'].tolist(),foot_kakudo['left_foot_z3'].tolist())
        right_D3_kakudo_xy = np.arctan2(foot_kakudo['right_foot_y3'].tolist(),foot_kakudo['right_foot_x3'].tolist())
        right_D3_kakudo_zy = np.arctan2(foot_kakudo['right_foot_y3'].tolist(),foot_kakudo['right_foot_z3'].tolist())
        
        
        
        foot_kakudo['press_kakudo_xy'] = press_kakudo_xy
        foot_kakudo['press_kakudo_zy'] = press_kakudo_zy
        foot_kakudo['press_power_xy'] = press_power_xy
        foot_kakudo['press_power_zy'] = press_power_zy
        foot_kakudo['left_D1_kakudo_xy'] = left_D1_kakudo_xy
        foot_kakudo['left_D1_kakudo_zy'] = left_D1_kakudo_zy
        foot_kakudo['right_D1_kakudo_xy'] = right_D1_kakudo_xy
        foot_kakudo['right_D1_kakudo_zy'] = right_D1_kakudo_zy
        foot_kakudo['left_D2_kakudo_xy'] = left_D2_kakudo_xy
        foot_kakudo['left_D2_kakudo_zy'] = left_D2_kakudo_zy
        foot_kakudo['right_D2_kakudo_xy'] = right_D2_kakudo_xy
        foot_kakudo['right_D2_kakudo_zy'] = right_D2_kakudo_zy
        foot_kakudo['left_D3_kakudo_xy'] = left_D3_kakudo_xy
        foot_kakudo['left_D3_kakudo_zy'] = left_D3_kakudo_zy
        foot_kakudo['right_D3_kakudo_xy'] = right_D3_kakudo_xy
        foot_kakudo['right_D3_kakudo_zy'] = right_D3_kakudo_zy
        
        left_e1 = []
        left_e2 = []
        left_e3 = []
        left_f1 = []
        left_f2 = []
        left_f3 = []
        
        left_D4_kakudo_xy = angle.opposition(left_D1_kakudo_xy)
        left_D5_kakudo_xy = angle.opposition(left_D2_kakudo_xy)
        left_D6_kakudo_xy = angle.opposition(left_D3_kakudo_xy)
        left_D4_kakudo_zy = angle.opposition(left_D1_kakudo_zy)
        left_D5_kakudo_zy = angle.opposition(left_D2_kakudo_zy)
        left_D6_kakudo_zy = angle.opposition(left_D3_kakudo_zy)
        
        angle1_xy = [-x for x in left_D6_kakudo_xy]
        angle2_xy = [-x + y for x , y in zip(left_D2_kakudo_xy , left_D6_kakudo_xy)]
        
        
        for i in range(len(press_kakudo_xy)):
            
            my_list = [left_D1_kakudo_xy[i],left_D2_kakudo_xy[i],left_D3_kakudo_xy[i],left_D4_kakudo_xy[i],left_D5_kakudo_xy[i],left_D6_kakudo_xy[i]]
            my_list2 = [left_D1_kakudo_zy[i],left_D2_kakudo_zy[i],left_D3_kakudo_zy[i],left_D4_kakudo_zy[i],left_D5_kakudo_zy[i],left_D6_kakudo_zy[i]]
            my_list3 = [1,2,3,4,5,6]
            
            sequential_order = vd.Vector_Direction_Distinction(my_list,'角度',my_list3,'D')
            sequential_order2 = vd.Vector_Direction_Distinction(my_list2,'角度',my_list3,'D')
            
            if sequential_order.iloc[1,0] < press_kakudo_xy[i] <= sequential_order.iloc[0,0]:
                wariai = (press_kakudo_xy[i] - sequential_order.iloc[1,0])/(sequential_order.iloc[0,0] - sequential_order.iloc[1,0])
                first = vd.muscle_strength(sequential_order.iloc[0,1])
                second = vd.muscle_strength(sequential_order.iloc[1,1])
                first = np.array(first)
                second = np.array(second)
                MY_list = (first*wariai + second*(1 - wariai))
            
            elif sequential_order.iloc[2,0] < press_kakudo_xy[i] <= sequential_order.iloc[1,0]:
                wariai = (press_kakudo_xy[i] - sequential_order.iloc[2,0])/(sequential_order.iloc[1,0] - sequential_order.iloc[2,0])
                first = vd.muscle_strength(sequential_order.iloc[1,1])
                second = vd.muscle_strength(sequential_order.iloc[2,1])
                first = np.array(first)
                second = np.array(second)
                MY_list = (first*wariai + second*(1 - wariai))
            
            elif sequential_order.iloc[3,0] < press_kakudo_xy[i] <= sequential_order.iloc[2,0]:
                wariai = (press_kakudo_xy[i] - sequential_order.iloc[3,0])/(sequential_order.iloc[2,0] - sequential_order.iloc[3,0])
                first = vd.muscle_strength(sequential_order.iloc[2,1])
                second = vd.muscle_strength(sequential_order.iloc[3,1])
                first = np.array(first)
                second = np.array(second)
                MY_list = (first*wariai + second*(1 - wariai))
            
            elif sequential_order.iloc[4,0] < press_kakudo_xy[i] <= sequential_order.iloc[3,0]:
                wariai = (press_kakudo_xy[i] - sequential_order.iloc[4,0])/(sequential_order.iloc[3,0] - sequential_order.iloc[4,0])
                first = vd.muscle_strength(sequential_order.iloc[3,1])
                second = vd.muscle_strength(sequential_order.iloc[4,1])
                first = np.array(first)
                second = np.array(second)
                MY_list = (first*wariai + second*(1 - wariai))
            
            elif sequential_order.iloc[5,0] < press_kakudo_xy[i] <= sequential_order.iloc[4,0]:
                wariai = (press_kakudo_xy[i] - sequential_order.iloc[5,0])/(sequential_order.iloc[4,0] - sequential_order.iloc[5,0])
                first = vd.muscle_strength(sequential_order.iloc[4,1])
                second = vd.muscle_strength(sequential_order.iloc[5,1])
                first = np.array(first)
                second = np.array(second)
                MY_list = (first*wariai + second*(1 - wariai))
            
            #ここ解決してね
            else:
                if press_kakudo_xy[i] > 0:
                    wariai = (press_kakudo_xy[i] - sequential_order.iloc[0,0])/(sequential_order.iloc[5,0] - sequential_order.iloc[0,0] + 2*np.pi)
                    first = vd.muscle_strength(sequential_order.iloc[5,1])
                    second = vd.muscle_strength(sequential_order.iloc[0,1])
                    first = np.array(first)
                    second = np.array(second)
                    MY_list = (first*wariai + second*(1 - wariai))
                    
                if press_kakudo_xy[i] < 0:
                    wariai = (press_kakudo_xy[i] + 2*np.pi - sequential_order.iloc[0,0])/(sequential_order.iloc[5,0] - sequential_order.iloc[0,0] + 2*np.pi)
                    first = vd.muscle_strength(sequential_order.iloc[5,1])
                    second = vd.muscle_strength(sequential_order.iloc[0,1])
                    first = np.array(first)
                    second = np.array(second)
                    MY_list = (first*wariai + second*(1 - wariai))
                    
            if sequential_order2.iloc[1,0] < press_kakudo_zy[i] <= sequential_order2.iloc[0,0]:
                wariai = (press_kakudo_zy[i] - sequential_order2.iloc[1,0])/(sequential_order2.iloc[0,0] - sequential_order2.iloc[1,0])
                first2 = vd.muscle_strength(sequential_order2.iloc[0,1])
                second2 = vd.muscle_strength(sequential_order2.iloc[1,1])
                first2 = np.array(first2)
                second2 = np.array(second2)
                MY_list2 = (first2*wariai + second2*(1 - wariai))
            
            elif sequential_order2.iloc[2,0] < press_kakudo_zy[i] <= sequential_order2.iloc[1,0]:
                wariai = (press_kakudo_zy[i] - sequential_order2.iloc[2,0])/(sequential_order2.iloc[1,0] - sequential_order2.iloc[2,0])
                first2 = vd.muscle_strength(sequential_order2.iloc[1,1])
                second2 = vd.muscle_strength(sequential_order2.iloc[2,1])
                first2 = np.array(first2)
                second2 = np.array(second2)
                MY_list2 = (first2*wariai + second2*(1 - wariai))
            
            elif sequential_order2.iloc[3,0] < press_kakudo_zy[i] <= sequential_order2.iloc[2,0]:
                wariai = (press_kakudo_zy[i] - sequential_order2.iloc[3,0])/(sequential_order2.iloc[2,0] - sequential_order2.iloc[3,0])
                first2 = vd.muscle_strength(sequential_order2.iloc[2,1])
                second2 = vd.muscle_strength(sequential_order2.iloc[3,1])
                first2 = np.array(first2)
                second2 = np.array(second2)
                MY_list2 = (first2*wariai + second2*(1 - wariai))
            
            elif sequential_order2.iloc[4,0] < press_kakudo_zy[i] <= sequential_order2.iloc[3,0]:
                wariai = (press_kakudo_zy[i] - sequential_order2.iloc[4,0])/(sequential_order2.iloc[3,0] - sequential_order2.iloc[4,0])
                first2 = vd.muscle_strength(sequential_order2.iloc[3,1])
                second2 = vd.muscle_strength(sequential_order2.iloc[4,1])
                first2 = np.array(first2)
                second2 = np.array(second2)
                MY_list2 = (first2*wariai + second2*(1 - wariai))
            
            elif sequential_order2.iloc[5,0] < press_kakudo_zy[i] <= sequential_order2.iloc[4,0]:
                wariai = (press_kakudo_zy[i] - sequential_order2.iloc[5,0])/(sequential_order2.iloc[4,0] - sequential_order2.iloc[5,0])
                first2 = vd.muscle_strength(sequential_order2.iloc[4,1])
                second2 = vd.muscle_strength(sequential_order2.iloc[5,1])
                first2 = np.array(first2)
                second2 = np.array(second2)
                MY_list2 = (first2*wariai + second2*(1 - wariai))
            
            #ここ解決してね
            else:
                if press_kakudo_zy[i] > 0:
                    wariai = (press_kakudo_zy[i] - sequential_order2.iloc[0,0])/(sequential_order2.iloc[5,0] - sequential_order2.iloc[0,0] + 2*np.pi)
                    first2 = vd.muscle_strength(sequential_order2.iloc[5,1])
                    second2 = vd.muscle_strength(sequential_order2.iloc[0,1])
                    first2 = np.array(first2)
                    second2 = np.array(second2)
                    MY_list2 = (first2*wariai + second2*(1 - wariai))
                    
                if press_kakudo_zy[i] < 0:
                    wariai = (press_kakudo_zy[i] + 2*np.pi - sequential_order2.iloc[0,0])/(sequential_order2.iloc[5,0] - sequential_order2.iloc[0,0] + 2*np.pi)
                    first2 = vd.muscle_strength(sequential_order2.iloc[5,1])
                    second2 = vd.muscle_strength(sequential_order2.iloc[0,1])
                    first2 = np.array(first2)
                    second2 = np.array(second2)
                    MY_list2 = (first2*wariai + second2*(1 - wariai))
                    
            list_coalescence = list(MY_list) + list(MY_list2)
            foot_kakudo2 = foot_kakudo2.append(pd.Series(list_coalescence),ignore_index=True)
        
        #要素は6個、なのに項目は12個、なんで？→下に追加されてる
        foot_kakudo.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\out\{filename}_kakudo_only.csv', index=True)
        #foot_kakudo2.columns = new_columns
        #foot_kakudo2.to_csv(rf'C:\Users\yota0\Desktop\yano\out\{filename}_Muscle_activity.csv', index=True)