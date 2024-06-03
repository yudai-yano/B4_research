from csv import writer
from parameter import AE_COM_RATIO_MALE
from parameter import AE_COM_RATIO_FEMALE
import parameter as pm
import csv
#import PySimpleGUI as sg
import datetime
import numpy as np
import sys
import glob
import os
import pandas as pd
import subject_data as sd

CoM_header = ['head', 'body', 'r_up_arm', 'l_up_arm', 'r_fore_arm', 'l_fore_arm',
              'r_hand', 'l_hand', 'r_thigh', 'l_thigh', 'r_crus', 'l_crus', 'r_foot', 'l_foot']
coordinate = ['x', 'y', 'z']

csv_header = [(coor + '_' + name)
              for name in CoM_header for coor in coordinate]

#頭頂点-首中心
#首中心-腰
#肩-ひじ
#ひじ-手首
#手の大きさ
#腰-膝
#膝-足首
#足の大きさ

# sekiguchi size
#body_measured_list = [256, 448, 325, 255, 180, 400, 365, 250, 248, 200]
# kamiyama size
#body_measured_list = [285, 480, 320, 260, 200, 485, 420, 270, 248, 200]
# tojo size
#body_measured_list = [270, 460, 290, 265, 180, 480, 400, 255, 248, 200]
# sugishita size
#body_measured_list = [250, 500, 290, 234, 181, 475, 375, 255, 248, 200]

# Subject information (in this case, folder name)
#$subject = 'deguchi_70per_1_correct_modelbased'

# 1RM（in this case, weight = 0 kg）
#rm_100 = 0
#rm_70 = rm_100 * 0.7

# body weight
#body_weight = 57.3

# Load weight
#Weight = rm_70

# body part length（measured）
#right_hand = 170
#left_hand = 170
#foot_size = 225
#torso = 520 # upper/lower torso= 0.5 torso

#subject = input("Enter subject name:")


filepath = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta'

for dir in glob.glob(rf'{filepath}\*'):
    folder = os.path.basename(dir)
    subject = folder.replace("_length.csv", "")
    
    #length_path = rf'{dir}\length.csv'
    if not dir.endswith("_length.csv"):
        pass
    else:
        #subject_dataから被験者のデータを取ってくる
        subject_name = sd.subject_name_input(subject)
        subject_data = sd.subject_data_input(subject_name)
        
        sex = subject_data.loc['sex']
        if sex == "F":
            gender = 'female'
        if sex == "M":
            gender = 'male'
        rm_100 = subject_data.loc['ONE_RM']
        rm_70 = rm_100 * 0.7
        body_weight = subject_data.loc['mass']
        Weight = round(rm_70 , 0)
        right_hand = subject_data.loc['r_hand']
        left_hand = subject_data.loc['l_hand']
        foot_size = subject_data.loc['shoes']
        torso = subject_data.loc['body']
        
        length = pd.read_csv(dir, encoding="shift jis")
        
        # weight = 60  # body weight
        
        handWeightParame = [
            Weight/2, Weight/2
        ]  # 1:left,0:right

        # 単位をmmに直す
        head_neck = length.iloc[0][0]*1000
        shoulder_waist = length.iloc[0][1]*1000
        shoulder_elbow =length.iloc[0][2]*1000
        elbow_wrist = length.iloc[0][3]*1000
        waist_knee = length.iloc[0][4]*1000
        knee_ankle = length.iloc[0][5]*1000

        hand_size = (right_hand + left_hand) / 2
        body_estimated_list = [head_neck+100,shoulder_waist,shoulder_elbow,elbow_wrist
                            ,hand_size,waist_knee,knee_ankle] # 手の大きさを変更
        body_measured_list = body_estimated_list + [foot_size, torso/2, torso/2]
        body_measured_list_index = pd.Series(['headtop_neck', 'shoulder_waist', 'shoulder_elbow', 'elbow_wrist', 'hand', 'waist_knee', 'knee_ankle', 'foot', 'upper_torso', 'lower_torso'])

        '''
        if n =='M1':
            body_measured_list = body_estimated_list + [250, 248, 200]
        elif n =='M2':
            body_measured_list = body_estimated_list + [255, 248, 200]
        elif n =='M3':
            body_measured_list = body_estimated_list + [262, 248, 200]
        elif n =='M4':
            body_measured_list = body_estimated_list + [268, 248, 200]
        elif n =='M5':
            body_measured_list = body_estimated_list + [265, 248, 200]
        elif n =='M6':
            body_measured_list = body_estimated_list + [262, 248, 200]
        elif n =='M7':
            body_measured_list = body_estimated_list + [261, 248, 200]
        elif n =='M8':
            body_measured_list = body_estimated_list + [264, 248, 200]
        elif n =='M9':
            body_measured_list = body_estimated_list + [254, 248, 200]
        elif n =='M10':
            body_measured_list = body_estimated_list + [271, 248, 200]
        elif n =='M11':
            body_measured_list = body_estimated_list + [245, 248, 200]
        elif n =='M12':
            body_measured_list = body_estimated_list + [248, 248, 200]
        elif n =='M13':
            body_measured_list = body_estimated_list + [247, 248, 200]
        elif n =='M14':
            body_measured_list = body_estimated_list + [257, 248, 200]
        elif n =='M15':
            body_measured_list = body_estimated_list + [244, 248, 200]
        elif n =='M16':
            body_measured_list = body_estimated_list + [260, 248, 200]
        elif n =='M17':
            body_measured_list = body_estimated_list + [283, 248, 200]
        elif n =='M18':
            body_measured_list = body_estimated_list + [253, 248, 200]
            
        elif n =='W1':
            body_measured_list = body_estimated_list + [250, 248, 200]
        elif n =='W2':
            body_measured_list = body_estimated_list + [240, 248, 200]
        elif n =='W3':
            body_measured_list = body_estimated_list + [248, 248, 200]
        elif n =='W4':
            body_measured_list = body_estimated_list + [248, 248, 200]
        elif n =='W5':
            body_measured_list = body_estimated_list + [252, 248, 200]
        elif n =='W6':
            body_measured_list = body_estimated_list + [268, 248, 200]
        elif n =='W7':
            body_measured_list = body_estimated_list + [252, 248, 200]
        elif n =='W8':
            body_measured_list = body_estimated_list + [236, 248, 200]
        elif n =='W9':
            body_measured_list = body_estimated_list + [244, 248, 200]
        elif n =='W10':
            body_measured_list = body_estimated_list + [228, 248, 200]
        elif n =='W11':
            body_measured_list = body_estimated_list + [235, 248, 200]
        elif n =='W12':
            body_measured_list = body_estimated_list + [244, 248, 200]
        elif n =='W13':
            body_measured_list = body_estimated_list + [242, 248, 200]
        elif n =='W14':
            body_measured_list = body_estimated_list + [234, 248, 200]
        elif n =='W15':
            body_measured_list = body_estimated_list + [228, 248, 200]
        elif n =='W16':
            body_measured_list = body_estimated_list + [248, 248, 200]
        elif n =='W17':
            body_measured_list = body_estimated_list + [235, 248, 200]
        elif n =='W18':
            body_measured_list = body_estimated_list + [234, 248, 200]'''
            
        body_measured_d = pd.DataFrame(body_measured_list)
        body_measured_d.insert(0, 'index',  body_measured_list_index)


        body_measured_d.to_csv(rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta\{subject}_length2.csv', index=False, header = None)

        """
        AE_PARTMASS_EST_A0_MALE 
            'head':
            'body':
            'upper_arm':
            'fore_arm':
            'hand':
            'thigh':
            'crus':
            'foot':
            'upper_torso': 
            'lower_torso': 
        """
        keypoints = [
            [0],
            [1, 8, 11],
            [2, 3],
            [5, 6],
            [3, 4],
            [6, 7],
            [4],
            [7],
            [8, 9],
            [11, 12],
            [9, 10],
            [12, 13],
            [10],
            [13]
        ]
        def calc_CoM_from_dataSeries(data_series, CoM_List):
            '''
            calc part CoM location

            not beautiful. But i cant come up with an elegant idea...
            in the future, it should be written numpy
            '''
            if gender =='male':
                for data in keypoints:
                    for i in range(len(data_series)):
                        if data_series[i] == "":
                            data_series[i] = 0
                    if len(data) == 1:
                        # hand, head and foot parts CoM location
                        add_data = [float(data_series[data[0]*3 + i+1]) for i in range(3)]
                        CoM_List.extend(add_data)
                    elif len(data) == 3:
                        # body part CoM location
                        vec1 = [float(data_series[data[0]*3 + i+1]) for i in range(3)]
                        vec2 = [float(data_series[data[1]*3 + i+1]) for i in range(3)]
                        vec3 = [float(data_series[data[2]*3 + i+1]) for i in range(3)]
                        combined = [((x+y) / 2) for (x, y) in zip(vec2, vec3)]
                        r = pm.AE_COM_RATIO_MALE["body"]
                        add_data = [x + r * (y - x) for (x, y) in zip(vec1, combined)]
                        CoM_List.extend(add_data)
                    elif len(data) == 2:
                        # upper arm, forearm, thign and crus parts CoM location
                        vec1 = [float(data_series[data[0]*3 + i+1]) for i in range(3)]
                        vec2 = [float(data_series[data[1]*3 + i+1]) for i in range(3)]
                        if (data == [2, 3] or data == [5, 6]):
                            r = pm.AE_COM_RATIO_MALE["upper_arm"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
                        elif (data == [3, 4] or data == [6, 7]):
                            r = pm.AE_COM_RATIO_MALE["fore_arm"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
                        elif (data == [8, 9] or data == [11, 12]):
                            r = pm.AE_COM_RATIO_MALE["thigh"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
                        elif (data == [9, 10] or data == [12, 13]):
                            r = pm.AE_COM_RATIO_MALE["crus"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
            elif gender =='female':
                for data in keypoints:
                    if len(data) == 1:
                        add_data = [float(data_series[data[0]*3 + i+1]) for i in range(3)]
                        CoM_List.extend(add_data)
                    elif len(data) == 3:
                        vec1 = [float(data_series[data[0]*3 + i+1]) for i in range(3)]
                        vec2 = [float(data_series[data[1]*3 + i+1]) for i in range(3)]
                        vec3 = [float(data_series[data[2]*3 + i+1]) for i in range(3)]
                        combined = [((x+y) / 2) for (x, y) in zip(vec2, vec3)]
                        r = pm.AE_COM_RATIO_FEMALE["body"]
                        add_data = [x + r * (y - x) for (x, y) in zip(vec1, combined)]
                        CoM_List.extend(add_data)
                    elif len(data) == 2:
                        vec1 = [float(data_series[data[0]*3 + i+1]) for i in range(3)]
                        vec2 = [float(data_series[data[1]*3 + i+1]) for i in range(3)]
                        if (data == [2, 3] or data == [5, 6]):
                            r = pm.AE_COM_RATIO_FEMALE["upper_arm"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
                        elif (data == [3, 4] or data == [6, 7]):
                            r = pm.AE_COM_RATIO_FEMALE["fore_arm"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
                        elif (data == [8, 9] or data == [11, 12]):
                            r = pm.AE_COM_RATIO_FEMALE["thigh"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)
                        elif (data == [9, 10] or data == [12, 13]):
                            r = pm.AE_COM_RATIO_FEMALE["crus"]
                            add_data = [x + r * (y - x) for (x, y) in zip(vec1, vec2)]
                            CoM_List.extend(add_data)

                
        def calc_partMass(body_measured_list, weight, AE_PartMass):
            """
            Ae's estimation formula in part mass
            partMass[kg] = a0 + a1 * bodypart_length[m] + a2 * full body weight[kg]
            0:head
            1:body
            2:upper arm
            3:fore arm
            4:hand
            5:thigh
            6:crus
            7:foot
            8:upper torso
            9:lower torso
            """
            if gender == 'male':
                a1 = list(pm.AE_PARTMASS_EST_A1_MALE.values())
                a2 = list(pm.AE_PARTMASS_EST_A2_MALE.values())
                for index, data in enumerate(pm.AE_PARTMASS_EST_A0_MALE.values()):
                    partMass = data + a1[index] * body_measured_list[index] / 1000.0 + a2[index] * weight
                    AE_PartMass.append(partMass)
                
            elif gender =='female':
                a1 = list(pm.AE_PARTMASS_EST_A1_FEMALE.values())
                a2 = list(pm.AE_PARTMASS_EST_A2_FEMALE.values())
                for index, data in enumerate(pm.AE_PARTMASS_EST_A0_FEMALE.values()):
                    partMass = data + a1[index] * body_measured_list[index] / 1000 + a2[index] * weight
                    AE_PartMass.append(partMass)

        def extend_partMass(AE_PartMass, handweight):
            """
            extend partmass
            assume that left body parts are same as right one
            and add handweight to both handweight
            """
            extend_PartMass = [AE_PartMass[0],
                            AE_PartMass[1],
                            AE_PartMass[2],
                            AE_PartMass[2],
                            AE_PartMass[3],
                            AE_PartMass[3],
                            AE_PartMass[4] + float(handweight[0]),
                            AE_PartMass[4] + float(handweight[1]),
                            AE_PartMass[5],
                            AE_PartMass[5],
                            AE_PartMass[6],
                            AE_PartMass[6],
                            AE_PartMass[7],
                            AE_PartMass[7]]
            return extend_PartMass


        def main():
            load_data = pd.read_csv(rf'{filepath}\{subject}_range_correction.csv')
            load_file = rf'{filepath}\{subject}_range_correction.csv'
            #ここが読み込まない
            data = rf'{load_file}'
    
            """ ratio = list(AE_COM_RATIO_MALE.values())
            keys = list(AE_COM_RATIO_MALE.keys())
            for index, length in enumerate(body_measured_list):
                AE_CoM.append(length * ratio[index]) """

            param = handWeightParame
            # os.makedirs(f'Data/Output/CoMdata/{n}', exist_ok = True)
            # os.makedirs(f'Data/Output/CoMdata/', exist_ok = True)
            csvname = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\body_measure\{subject}_CoMdata.csv'
            CoM_csv = open(csvname, 'w', newline='')
            writer = csv.writer(CoM_csv)
            writer.writerow(csv_header)

            # calculate body part CoM location
            with open(data,'r') as csv_stream:
                reader = csv.reader(csv_stream)
                for index, data_row in enumerate(reader):
                    if index > 0:
                        for_csv_array = []
                        calc_CoM_from_dataSeries(data_row, for_csv_array)
                        writer.writerow(for_csv_array)
            CoM_csv.close()

            # calculation partMass
            AE_PartMass = []
            calc_partMass(body_measured_list, body_weight, AE_PartMass)

            # convert partmass into fullbody
            AE_PartMass = extend_partMass(AE_PartMass, param)
            # os.makedirs(f'Data/Output/BodyCoM/', exist_ok = True)
            body_CoM = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\body_measure\{subject}_BodyCoM.csv'
            body_CoM_csv = open(body_CoM, 'w', newline='')
            writer = csv.writer(body_CoM_csv)
            writer.writerow(['x_CoM', 'y_CoM', 'z_CoM'])

            with open(csvname,'r') as CoM_Series:
                reader = csv.reader(CoM_Series)
                for index, data_row in enumerate(reader):
                    if index > 0:
                        x = [float(data_row[3 * i]) for i in range(14)]
                        y = [float(data_row[3 * i + 1]) for i in range(14)]
                        z = [float(data_row[3 * i + 2]) for i in range(14)]
                        x_dot = np.dot(x, AE_PartMass)
                        y_dot = np.dot(y, AE_PartMass)
                        z_dot = np.dot(z, AE_PartMass)
                        Weight = np.sum(AE_PartMass)
                        writer.writerow(
                            [x_dot / Weight, y_dot / Weight, z_dot / Weight])

            body_CoM_csv.close()
            
            # 部分質量
            # os.makedirs(f'Data/Output/Massdata/{n}', exist_ok = True)
            # os.makedirs(f'Data/Output/Massdata', exist_ok = True)
            csvname2 = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\body_measure\{subject}_Massdata.csv'
            Mass_csv = open(csvname2, 'w', newline='')
            writer = csv.writer(Mass_csv)
            writer.writerow(CoM_header)
            with open(csvname2) as Mass_Series:
                reader = csv.reader(Mass_Series)
                writer.writerow(AE_PartMass)
            Mass_csv.close()


        if __name__ == "__main__":
            main()
