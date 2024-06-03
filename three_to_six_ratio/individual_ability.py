import numpy as np
import pandas as pd
import os
import angle
import Vector_Direction_Distinction as vd

hexagon_file = rf'C:\Users\yota0\Desktop\yano\out\fukuman_70per_2_maxangle.csv'
kakudo_file = rf'C:\Users\yota0\Desktop\yano\out\fukuman_70per_2_kakudo_only.csv'
filepath = os.path.split(os.path.basename(hexagon_file))[-1]
filename = filepath.replace("_maxangle.csv", "")
hexagon_csv = pd.read_csv(rf'{hexagon_file}', header=0)
kakudo_csv = pd.read_csv(rf'{kakudo_file}' , header=0)
foot_kakudo2 = pd.DataFrame()
new_columns = ['left_e1_xy','left_e2_xy','left_e3_xy','left_f1_xy','left_f2_xy','left_f3_xy']
hexagon_angle_data = hexagon_csv.loc[:, '6':'11']

left_D1_kakudo_xy = hexagon_angle_data['6']
left_D2_kakudo_xy = hexagon_angle_data['7']
left_D3_kakudo_xy = hexagon_angle_data['8']
left_D4_kakudo_xy = hexagon_angle_data['9']
left_D5_kakudo_xy = hexagon_angle_data['10']
left_D6_kakudo_xy = hexagon_angle_data['11']


press_kakudo_xy = kakudo_csv.loc[:, 'press_kakudo_xy']

for i in range(len(press_kakudo_xy)):
    
    my_list = [left_D1_kakudo_xy[i],left_D2_kakudo_xy[i],left_D3_kakudo_xy[i],left_D4_kakudo_xy[i],left_D5_kakudo_xy[i],left_D6_kakudo_xy[i]]
    my_list2 = [1,2,3,4,5,6]
    
    sequential_order = vd.Vector_Direction_Distinction(my_list,'角度',my_list2,'D')
    
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
            
    foot_kakudo2 = foot_kakudo2.append(pd.Series(list(MY_list)),ignore_index=True)

#要素は6個、なのに項目は12個、なんで？→下に追加されてる
foot_kakudo2.columns = new_columns
foot_kakudo2.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\out\{filename}_Muscle_individual_activity.csv', index=True)