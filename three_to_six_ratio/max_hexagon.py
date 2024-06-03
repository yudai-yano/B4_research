import pandas as pd
import angle
import max_muscular
import os

file_name = rf'C:\Users\yota0\Desktop\yano\out\fukuman_70per_2_kakudo_only.csv'
filepath = os.path.split(os.path.basename(file_name))[-1]
filename = filepath.replace("_kakudo_only.csv", "")
csv_frame = pd.read_csv(rf'{file_name}', header=0)
D1_angle = csv_frame.loc[:, 'left_D1_kakudo_xy']
D2_angle = csv_frame.loc[:, 'left_D2_kakudo_xy']
D3_angle = csv_frame.loc[:, 'left_D3_kakudo_xy']
D1_angle = D1_angle.tolist()
D2_angle = D2_angle.tolist()
D3_angle = D3_angle.tolist()
D4_angle = angle.opposition(D1_angle)
D5_angle = angle.opposition(D2_angle)
D6_angle = angle.opposition(D3_angle)

angle = pd.DataFrame({'D1_angle':D1_angle , 'D2_angle':D2_angle , 'D3_angle':D3_angle , 'D4_angle':D4_angle , 'D5_angle':D5_angle , 'D6_angle':D6_angle})
df = max_muscular.angle_maximum_muscular_strength(angle)

#列名を変更してね
df.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\out\{filename}_maxangle.csv')