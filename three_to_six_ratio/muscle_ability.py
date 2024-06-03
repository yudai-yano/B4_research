import pandas as pd
import Vector_Direction_Distinction as vd
import numpy as np
import os


muscle_angle_csv = rf'C:\Users\yota0\Desktop\yano\out\fukuman_70per_2_kakudo_only.csv'
max_hexagon_csv = rf'C:\Users\yota0\Desktop\yano\out\fukuman_70per_2_maxangle.csv'
filepath = os.path.split(os.path.basename(muscle_angle_csv))[-1]
filename = filepath.replace("_kakudo_only.csv", "")
muscle_angle = pd.read_csv(rf'{muscle_angle_csv}', header=0)
max_hexagon = pd.read_csv(rf'{max_hexagon_csv}', header=0)

force_plate = muscle_angle.loc[:, 'press_kakudo_xy']
max_press = []
press_x = []
press_y = []
muscle_ratio = []
real_press = []

row = len(max_hexagon)
for i in range(row):
    hexagon_angle_1 = max_hexagon.iloc[i,6]
    hexagon_angle_2 = max_hexagon.iloc[i,7]
    hexagon_angle_3 = max_hexagon.iloc[i,8]
    hexagon_angle_4 = max_hexagon.iloc[i,9]
    hexagon_angle_5 = max_hexagon.iloc[i,10]
    hexagon_angle_6 = max_hexagon.iloc[i,11]
    
    max_hexagon_1 = max_hexagon.iloc[i,0] * 9.8
    max_hexagon_2 = max_hexagon.iloc[i,1] * 9.8
    max_hexagon_3 = max_hexagon.iloc[i,2] * 9.8
    max_hexagon_4 = max_hexagon.iloc[i,3] * 9.8
    max_hexagon_5 = max_hexagon.iloc[i,4] * 9.8
    max_hexagon_6 = max_hexagon.iloc[i,5] * 9.8
    
    force_plate_angle = force_plate[i]
    
    angle = [hexagon_angle_1 , hexagon_angle_2 , hexagon_angle_3 , hexagon_angle_4 , hexagon_angle_5 , hexagon_angle_6]
    load = [max_hexagon_1 , max_hexagon_2 , max_hexagon_3 , max_hexagon_4 , max_hexagon_5 , max_hexagon_6]
    
    hexagon = vd.Vector_Direction_Distinction(angle , '角度' , load , '大きさ')
    if hexagon.iloc[4,0] >= force_plate_angle > hexagon.iloc[5,0]:
        ratio = (force_plate_angle - hexagon.iloc[5,0])/(hexagon.iloc[4,0] - hexagon.iloc[5,0])
        x_press = ((1 - ratio) * hexagon.iloc[5,1] * np.cos(hexagon.iloc[5,0]) + ratio * hexagon.iloc[4,1] * np.cos(hexagon.iloc[4,0]))
        y_press = ((1 - ratio) * hexagon.iloc[5,1] * np.sin(hexagon.iloc[5,0]) + ratio * hexagon.iloc[4,1] * np.sin(hexagon.iloc[4,0]))
        press = np.sqrt(x_press ** 2 + y_press ** 2)
        
    if hexagon.iloc[3,0] >= force_plate_angle > hexagon.iloc[4,0]:
        ratio = (force_plate_angle - hexagon.iloc[4,0])/(hexagon.iloc[3,0] - hexagon.iloc[4,0])
        x_press = ((1 - ratio) * hexagon.iloc[4,1] * np.cos(hexagon.iloc[4,0]) + ratio * hexagon.iloc[3,1] * np.cos(hexagon.iloc[3,0]))
        y_press = ((1 - ratio) * hexagon.iloc[4,1] * np.sin(hexagon.iloc[4,0]) + ratio * hexagon.iloc[3,1] * np.sin(hexagon.iloc[3,0]))
        press = np.sqrt(x_press ** 2 + y_press ** 2)
        
    if hexagon.iloc[2,0] >= force_plate_angle > hexagon.iloc[3,0]:
        ratio = (force_plate_angle - hexagon.iloc[3,0])/(hexagon.iloc[2,0] - hexagon.iloc[3,0])
        x_press = ((1 - ratio) * hexagon.iloc[3,1] * np.cos(hexagon.iloc[3,0]) + ratio * hexagon.iloc[2,1] * np.cos(hexagon.iloc[2,0]))
        y_press = ((1 - ratio) * hexagon.iloc[3,1] * np.sin(hexagon.iloc[3,0]) + ratio * hexagon.iloc[2,1] * np.sin(hexagon.iloc[2,0]))
        press = np.sqrt(x_press ** 2 + y_press ** 2)
        
    if hexagon.iloc[1,0] >= force_plate_angle > hexagon.iloc[2,0]:
        ratio = (force_plate_angle - hexagon.iloc[2,0])/(hexagon.iloc[1,0] - hexagon.iloc[2,0])
        x_press = ((1 - ratio) * hexagon.iloc[2,1] * np.cos(hexagon.iloc[2,0]) + ratio * hexagon.iloc[1,1] * np.cos(hexagon.iloc[1,0]))
        y_press = ((1 - ratio) * hexagon.iloc[2,1] * np.sin(hexagon.iloc[2,0]) + ratio * hexagon.iloc[1,1] * np.sin(hexagon.iloc[1,0]))
        press = np.sqrt(x_press ** 2 + y_press ** 2)
        
    if hexagon.iloc[0,0] >= force_plate_angle > hexagon.iloc[1,0]:
        ratio = (force_plate_angle - hexagon.iloc[1,0])/(hexagon.iloc[0,0] - hexagon.iloc[1,0])
        x_press = ((1 - ratio) * hexagon.iloc[1,1] * np.cos(hexagon.iloc[1,0]) + ratio * hexagon.iloc[0,1] * np.cos(hexagon.iloc[0,0]))
        y_press = ((1 - ratio) * hexagon.iloc[1,1] * np.sin(hexagon.iloc[1,0]) + ratio * hexagon.iloc[0,1] * np.sin(hexagon.iloc[0,0]))
        press = np.sqrt(x_press ** 2 + y_press ** 2)
        
    else:
        if force_plate_angle > 0:
            ratio = (force_plate_angle - hexagon.iloc[0,0])/(hexagon.iloc[5,0] - hexagon.iloc[0,0] + 2 * np.pi)
            x_press = ((1 - ratio) * hexagon.iloc[0,1] * np.cos(hexagon.iloc[0,0]) + ratio * hexagon.iloc[5,1] * np.cos(hexagon.iloc[5,0]))
            y_press = ((1 - ratio) * hexagon.iloc[0,1] * np.sin(hexagon.iloc[0,0]) + ratio * hexagon.iloc[5,1] * np.sin(hexagon.iloc[5,0]))
            press = np.sqrt(x_press ** 2 + y_press ** 2)
            
        elif force_plate_angle <= 0:
            ratio = (force_plate_angle - hexagon.iloc[0,0] + 2 * np.pi)/(hexagon.iloc[5,0] - hexagon.iloc[0,0] + 2 * np.pi)
            x_press = ((1 - ratio) * hexagon.iloc[0,1] * np.cos(hexagon.iloc[0,0]) + ratio * hexagon.iloc[5,1] * np.cos(hexagon.iloc[5,0]))
            y_press = ((1 - ratio) * hexagon.iloc[0,1] * np.sin(hexagon.iloc[0,0]) + ratio * hexagon.iloc[5,1] * np.sin(hexagon.iloc[5,0]))
            press = np.sqrt(x_press ** 2 + y_press ** 2)
            
    max_press.append(press)
            
real_press = muscle_angle.loc[:, 'press_power_xy']

for i in range(row):
    r_press = real_press[i]
    m_press = max_press[i]
    ratio2 = r_press/m_press
    
    muscle_ratio.append(ratio2)
    
data = {
    'maximum muscular strength':max_press,
    'real muscular strength':real_press,
    'Muscle activity':muscle_ratio,
}

df = pd.DataFrame(data)
df.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\out\{filename}_muscle_activity_correct.csv')