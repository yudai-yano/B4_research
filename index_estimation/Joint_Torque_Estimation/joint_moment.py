import numpy as np
import pandas as pd
import csv
import math
import glob
import os
import kalman_def2
import subject_data as sd

pd.set_option('display.max_columns', None)
'''
alfhaposeの座標の置き方とmediapipeの座標の置き方が違う
yはおそらく同じで
alfhaのxがmediaのz(逆もしかり)
変えるのめんどくさいので、xにz、zにxを代入している
と思ったけど、そうでもないかも
神谷データのyが変
'''
# Subject information (in this case, folder name)
#subject = 'deguchi_70per_1_correct_modelbased'
#gender = 'female'

# 1RM（in this case, weight = 0 kg）
#rm_100 = 0
#rm_70 = rm_100 * 0.7

# body weight
#body_weight = 80.4

# Load weight
#Weight = rm_70

# force
#m = body_weight + Weight
#mg = m * 9.8

# framerate
fps = 30

#subject = input("Enter subject name:")

'''入力側のファイル'''
filepath = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\length_csvdeta'
filepath2 = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\body_measure'
'''出力側のファイル'''
filepath3 = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\result'
filepath4 = rf'C:\Users\yota0\Desktop\yano\program\python\index_estimation\out\moment'

for dir in glob.glob(rf'{filepath}\*'):
    folder = os.path.basename(dir)
    subject = folder.replace("_length.csv", "")
    
    #length_path = rf'{dir}\length.csv'
    if not dir.endswith("_length.csv"):
        pass
    else:
        subject_name = sd.subject_name_input(subject)
        subject_data = sd.subject_data_input(subject_name)
        
        sex = subject_data.loc['sex']
        if sex == "F":
            gender = 'female'
        if sex == "M":
            gender = 'male'
        rm_100 = subject_data.loc['ONE_RM']
        rm_70 = rm_100 * 0.8
        body_weight = subject_data.loc['mass']
        Weight = round(rm_70 , 0)
        m = body_weight + Weight
        mg = m * 9.8
    
        #腰→膝，膝→足首の長さ取得
        length_data = pd.read_csv(rf'{filepath}\{subject}_length.csv', encoding="shift jis")
        body_list = [length_data.iloc[0][4], length_data.iloc[0][5]]
        length_data2 = pd.read_csv(rf'{filepath}\{subject}_length2.csv', encoding="shift jis")

        # 足の大きさ
        #L1 = 0.26 
        # 足首から地面までの長さ[m]
        L2 = 0.1
        # 足の長さ[m]
        L3 = length_data2.iloc[6][1]/1000 - 0.05 -0.02

        # 中央差分
        def centerdiff(data,dt):
            cdiff=(data.shift(-1)-data.shift(1))/2/dt
            return cdiff.fillna(method='ffill').fillna(method='bfill')
        # 2階中心差分
        def centerdiff2(data,dt):
            cdiff=(data.shift(-1)+data.shift(1)-2*data)/dt/dt
            return cdiff.fillna(method='ffill').fillna(method='bfill')

        # range_correction(座標データ)
        path_name = rf'{filepath}\{subject}_range_correction.csv'
        os.makedirs(rf'{filepath}\{subject}_joint_moment', exist_ok = True)

        # 関節座標データ
        df = -1*pd.read_csv(f"{path_name}")
        df_1 = df.loc[:, 'right_foot_x':'right_foot_z']
        df_2 = df.loc[:, 'left_foot_x':'left_foot_z']
        df1 = []
        df1.append(df_1)
        df1.append(df_2)
        df_foot = pd.concat(df1, axis=1)
        # ヘッダー
        CoM_header = ['r_foot', 'l_foot']
        coordinate = ['x', 'y', 'z']
        csv_header = [(coor + '_' + name)
                    for name in CoM_header for coor in coordinate]
        # 右足，左足の座標，後で計算しやすくする
        df_foot.columns = csv_header

        # 全体重心位置データ
        df_bcom = -1*pd.read_csv(rf'{filepath2}\{subject}_BodyCoM.csv')

        # 重量
        part_mass = pd.read_csv(rf'{filepath2}\{subject}_Massdata.csv')

        # 部分重心
        part_com = -pd.read_csv(rf'{filepath2}\{subject}_CoMdata.csv')
        part_com_header = part_com.columns

        rx_loca =[]
        ry_loca = []
        rz_loca =[]
        lx_loca =[]
        ly_loca = []
        lz_loca =[]

        # 床反力のx座標は足首と変わらない→alfhaposeのx座標がmediapipeのz座標
        rx_loca.append(df.loc[:, 'right_foot_z'])
        lx_loca.append(df.loc[:, 'left_foot_z'])

        # 重心の奥行位置→なくても問題ない？
        #body_g = df_bcom['z_CoM']

        # 足首の位置
        rx_g = df_foot['x_r_foot']
        ry_g = df_foot['y_r_foot']
        rz_g = df_foot['z_r_foot']
        lx_g = df_foot['x_l_foot']
        ly_g = df_foot['y_l_foot']
        lz_g = df_foot['z_l_foot']

        for i in range(len(df_foot)):
            # 地面までの距離(y)
            ryg = ry_g[i]
            lyg = ly_g[i]
            # 既存の位置から引く上正
            ryg_new = ryg - L2
            lyg_new = lyg - L2
            # 床反力作用点垂直位置
            ry_loca.append(ryg_new)
            ly_loca.append(lyg_new)
            # bodyg = body_g[i]
            rzg = rz_g[i]
            lzg = lz_g[i]
            # 足首位置との関係から床反力作用点推定
            rzg_new = rzg + L3/2
            rz_loca.append(rzg_new)
            lzg_new = lzg + L3/2
            lz_loca.append(lzg_new)
            
            '''
            # 右足
            if rzg-0.05 <= bodyg <= rzg+L3:
                rzg_new = bodyg
                rz_loca.append(rzg_new)
            elif bodyg > rzg+L3:
                rzg_new = rzg+L3
                rz_loca.append(rzg_new)
            elif bodyg < rzg-0.05:
                rzg_new = rzg-0.05
                rz_loca.append(rzg_new)
            # 左足    
            if lzg-0.05 <= bodyg <= lzg+L3:
                lzg_new = bodyg
                lz_loca.append(lzg_new)
            elif bodyg > lzg+L3:
                lzg_new = lzg+L3
                lz_loca.append(lzg_new)
            elif bodyg < lzg-0.05:
                lzg_new = lzg-0.05
                lz_loca.append(lzg_new)
            '''
        # 床反力右足x位置
        df_locate1 =pd.concat(rx_loca,axis = 1)
        df_locate1.columns = ['x_r_g']
        # 床反力左足y位置
        df_ry =pd.DataFrame(ry_loca)
        df_ry2 = []
        df_ry2.append(df_ry)
        df_locate2 =pd.concat(df_ry2,axis = 1)
        df_locate2.columns = ['y_r_g']
        # 床反力右足z位置
        df_rz = pd.DataFrame(rz_loca)
        df_rz2 = []
        df_rz2.append(df_rz)
        df_locate3 =pd.concat(df_rz2,axis = 1)
        df_locate3.columns = ['z_r_g']
        # 床反力左足x位置
        df_locate4 =pd.concat(lx_loca,axis = 1)
        df_locate4.columns = ['x_l_g']
        # 床反力左足y位置
        df_ly = pd.DataFrame(ly_loca)
        df5 = []
        df5.append(df_ly)
        df_locate5 =pd.concat(df5,axis = 1)
        df_locate5.columns = ['y_l_g']
        # 床反力左足z位置
        df_lz = pd.DataFrame(lz_loca)
        df6 = []
        df6.append(df_lz)
        df_locate6 =pd.concat(df6,axis = 1)
        df_locate6.columns = ['z_l_g']

        # 床反力作用点
        dfl_concat = pd.concat([df_locate1, df_locate2, df_locate3, df_locate4, df_locate5, df_locate6], axis = 1)
        dfl_concat.to_csv(rf'{filepath3}\{subject}_force_locate.csv')
        print('床反力作用点')

        df = df
        # 全体重心位置データの加速度
        acg_x = []
        acg_y = []
        acg_z = []

        # 中央差分法合成重心の加速度求める
        velocity_g = centerdiff(df_bcom,1/fps)
        # カルマンフィルタ
        velocity_g_kalman = kalman_def2.filtered_kalman(velocity_g, 0.1)
        accel_g = centerdiff(velocity_g_kalman ,1/fps)
        # カルマンフィルタ
        accel_g = kalman_def2.filtered_kalman(accel_g, 0.1)

        accel_g.columns = ['accel_gx', 'accel_gy', 'accel_gz']
        accel_g.to_csv(rf'{filepath3}\{subject}_cog_acceleration.csv')
        velocity_g_kalman.columns = ['vel_gx', 'vel_gy', 'vel_gz']
        velocity_g_kalman.to_csv(rf'{filepath3}\{subject}_cog_velocity.csv')
        print('重心加速度')

        accel_g.columns = ['accel_gx', 'accel_gy', 'accel_gz']

        # 床反力用
        #df_foot.columns = ['x','y','z','x','y','z']
        dfl_concat.columns = ['x','y','z','x','y','z']
        df_bcom.columns = ['x','y','z']

        # ここから床反力推定
        ax = accel_g.loc[:, 'accel_gx']
        ay = accel_g.loc[:, 'accel_gy']
        az = accel_g.loc[:, 'accel_gz']

        # 床反力垂直方向用
        # 重心→床反力作用点ベクトル
        delta_all_r = dfl_concat.iloc[:,0:3] - df_bcom
        deltarx = delta_all_r.loc[:, 'x']
        deltary = delta_all_r.loc[:, 'y']
        deltarz = delta_all_r.loc[:, 'z']
        delta_all_l = dfl_concat.iloc[:,3:7] - df_bcom
        deltalx = delta_all_l.loc[:, 'x']
        deltaly = delta_all_l.loc[:, 'y']
        deltalz = delta_all_l.loc[:, 'z']
        # 床反力作用点-重心距離
        distance_r = (deltarx**2 + deltary**2 + deltarz**2)
        distance_rsq = np.sqrt(distance_r)
        distance_l = (deltalx**2 + deltaly**2 + deltalz**2)
        distance_lsq = np.sqrt((distance_l))

        # 床反力左右前後方向用
        # 膝の座標
        df_knee_r = df.loc[:, 'right_knee_x':'right_knee_z']
        df_knee_l = df.loc[:, 'left_knee_x':'left_knee_z']
        df_knee = pd.concat([df_knee_r,df_knee_l], axis = 1)

        # 重心と床反力作用点のなす角
        #df_bcom = -1*pd.read_csv(rf"C:/Users/sk122/OneDrive - 東京理科大学/4M/引継ぎ資料/実験データ/日体大/BodyCoM/{n}/{filename}")
        dfs = {}
        for j in range(2):   
            com_x = df_bcom.iloc[:,0]
            com_y = df_bcom.iloc[:,1]
            com_z = df_bcom.iloc[:,2]
            cop_x = dfl_concat.iloc[:,3*j]
            cop_y = dfl_concat.iloc[:,3*j+1]
            cop_z = dfl_concat.iloc[:,3*j+2]
            theta_xy = np.arctan2(com_x-cop_x,com_y-cop_y)
            theta_yz = np.arctan2(com_z-cop_z,com_y-cop_y)
            dfs[f'{j}'] = {'xy1':theta_xy, 'yz1':theta_yz}
            
        r_rad = pd.DataFrame.from_dict(dfs['0'],orient='index').T
        l_rad = pd.DataFrame.from_dict(dfs['1'],orient='index').T
        com_cop_rad = pd.concat([r_rad, l_rad], axis =1)
        com_cop_rad.columns = ['rkc_xy','rkc_yz','lkc_xy','lkc_yz']
        com_cop_rad.to_csv(rf'{filepath3}\{subject}_cop_angle.csv')
        print('床反力角度')

        # 膝と床反力作用点の角度 
        '''
        dfs = {}
        for j in range(2):   
            knee_x = df_knee.iloc[:,3*j]
            knee_y = df_knee.iloc[:,3*j+1]
            knee_z = df_knee.iloc[:,3*j+2]
            cop_x = dfl_concat.iloc[:,3*j]
            cop_y = dfl_concat.iloc[:,3*j+1]
            cop_z = dfl_concat.iloc[:,3*j+2]
            theta_xy = -np.arctan2(knee_x-cop_x,knee_y-cop_y)
            theta_yz = np.arctan2(knee_z-cop_z,knee_y-cop_y)
            dfs[f'{j}'] = {'xy1':theta_xy, 'yz1':theta_yz}
            
        r_rad = pd.DataFrame.from_dict(dfs['0'],orient='index').T
        l_rad = pd.DataFrame.from_dict(dfs['1'],orient='index').T
        knee_cop_rad = pd.concat([r_rad, l_rad], axis =1)
        knee_cop_rad.columns = ['rkc_xy','rkc_yz','lkc_xy','lkc_yz']
        knee_cop_rad.to_csv(f'C:/Users/sk122/OneDrive - 東京理科大学/4M/引継ぎ資料/実験データ/日体大/{n}/{filename}_cop_angle.csv')
        '''
        Fry = []
        Frz = []
        Fly = []
        Flz = []
        Frx = []
        Flx = []
        Fv = []
        # 床反力推定(角度)
        for i in range(len(accel_g)):
            fv = m*ay.iloc[i]+mg
            fry = (m*ay.iloc[i]+mg)*(1-distance_rsq.iloc[i]/(
                distance_rsq.iloc[i]+distance_lsq.iloc[i]))
            fly = (m*ay.iloc[i]+mg)*(1-distance_lsq.iloc[i]/(
                distance_rsq.iloc[i]+distance_lsq.iloc[i]))
            '''
            frx = m*ax.iloc[i]/2
            flx = m*ax.iloc[i]/2
            frz = m*az.iloc[i]/2
            flz = m*az.iloc[i]/2
            '''
            frx = fry*np.tan(com_cop_rad.iloc[i][0])
            frz = fry*np.tan(com_cop_rad.iloc[i][1])
            flx = fry*np.tan(com_cop_rad.iloc[i][2])
            flz = fry*np.tan(com_cop_rad.iloc[i][3])
            
            Fv.append(fv)
            Fry.append(fry)
            Frz.append(frz)
            Fly.append(fly)
            Flz.append(flz)
            Frx.append(frx)
            Flx.append(flx)
        FV = pd.DataFrame(Fv)
        FRY = pd.DataFrame(Fry)
        FRZ = pd.DataFrame(Frz)
        FLY = pd.DataFrame(Fly)
        FLZ = pd.DataFrame(Flz)
        FRX = pd.DataFrame(Frx)
        FLX = pd.DataFrame(Flx)

        gf = pd.concat([FRZ, FRY, FLZ, FLY, FRX, FLX, FV], axis = 1)
        gf.columns = ['frz', 'fry', 'flz', 'fly', 'frx', 'flx', 'fv']

        # gf.to_csv(f'Data/Output/{n}/grf.csv')
        print('床反力')


        # 足首モーメント
        R_moment = []
        L_moment = []
        RL_moment = []
        distanceffy = []
        distanceffz = []
        distancerry = []
        distancerrz = []

        for i in range(len(gf)):
            # 足首と床反力作用点の距離(矢状平面)
            distancefy = -(dfl_concat.iloc[i][1] - df_foot.iloc[i][1])
            distancefz = dfl_concat.iloc[i][2] - df_foot.iloc[i][2]
            distancery = -(dfl_concat.iloc[i][4] - df_foot.iloc[i][4])
            distancerz = dfl_concat.iloc[i][5] - df_foot.iloc[i][5]
            distanceffy.append(distancefy)
            distanceffz.append(distancefz)
            distancerry.append(distancery)
            distancerrz.append(distancerz)
            # 足首関節モーメント
            R_moment1 = (gf.iloc[i][1]*distancefz + gf.iloc[i][0]*distancefy)
            L_moment1 = (gf.iloc[i][3]*distancerz + gf.iloc[i][2]*distancery)
            RL_moment1 = R_moment1 + L_moment1
            R_moment.append(R_moment1)
            L_moment.append(L_moment1)
            RL_moment.append(RL_moment1)

        distanceffrr1 = pd.DataFrame(distanceffy)
        distanceffrr2 = pd.DataFrame(distanceffz)
        distanceffrr3 = pd.DataFrame(distancerry)
        distanceffrr4 = pd.DataFrame(distancerrz)
        distanceffrr = pd.concat([distanceffrr1,distanceffrr2,
                                    distanceffrr3,distanceffrr4], axis = 1)
        distanceffrr.columns = ['ry','rz','ly','lz']
        distanceffrr.to_csv(rf'{filepath3}\{subject}_distanceffrr.csv')
        R_moment = pd.DataFrame(R_moment)
        L_moment = pd.DataFrame(L_moment)
        ankle_moment = pd.concat([R_moment, L_moment], axis = 1)
        ankle_moment.columns = ["R_moment", "L_moment"]
        RL_moment = pd.DataFrame(RL_moment)
        RL_moment.columns = ["Ankle_moment"]
        ankle_moment.to_csv(rf'{filepath4}\{subject}_ankle_moment.csv')
        RL_moment.to_csv(rf'{filepath4}\{subject}_RL_ankle_moment.csv')
        print('足首モーメント')

        ###################
        #ここから膝・股関節モーメントのパラメータ
                        
        # 床反力
        gf = gf

        acg_xp = []
        acg_yp = []
        acg_zp = []

        #part_com_a = part_com.copy()
        #part_com_a = pd.DataFrame(columns = part_com_header)
        '''
        xyz = []
        # 2階中心差分法部分重心の加速度求める
        for i in  range(1, len(part_com)-1):
            #xyz = []
            for j in range(14):
                acgx = (part_com.iloc[i-1][3*j] + part_com.iloc[i+1][3*j] - 2*part_com.iloc[i][3*j])*fps*fps
                acgy = (part_com.iloc[i-1][3*j+1] + part_com.iloc[i+1][3*j+1] - 2*part_com.iloc[i][3*j+1])*fps*fps
                acgz = (part_com.iloc[i-1][3*j+2] + part_com.iloc[i+1][3*j+2] - 2*part_com.iloc[i][3*j+2])*fps*fps
                xyz.append(acgx)
                xyz.append(acgy)
                xyz.append(acgz)
            #part_com_a.iloc[i,:]= xyz
        part_com_accel = np.array(xyz)
        part_com_accel = part_com_accel.reshape(-1, 42)
        pca = pd.DataFrame(part_com_accel)
        '''
        # 中央差分法部分重心の加速度求める
        velocity_part_com = centerdiff(part_com,1/fps)
        velocity_part_com_kalman = kalman_def2.filtered_kalman(velocity_part_com)
        part_com_accel = centerdiff(velocity_part_com_kalman ,1/fps)
        pca = kalman_def2.filtered_kalman(part_com_accel)
        pca.columns = part_com_header
        print('部分重心加速度')

        # 膝，腰の関節反力
        g = 9.8
        knee_rz = []
        knee_ry = []
        thigh_rz = []
        thigh_ry = []
        knee_lz = []
        knee_ly = []
        thigh_lz = []
        thigh_ly = []

        for i in range(len(pca)):
            kr_z = gf.iloc[i][0] - part_mass.iloc[0][10] * pca.iloc[i][32]
            kr_y = gf.iloc[i][1] - part_mass.iloc[0][10] * pca.iloc[i][31] - part_mass.iloc[0][10] * g
            kl_z = gf.iloc[i][2] - part_mass.iloc[0][11] * pca.iloc[i][35]
            kl_y = gf.iloc[i][3] - part_mass.iloc[0][11] * pca.iloc[i][34] - part_mass.iloc[0][11] * g
            tr_z = kr_z - part_mass.iloc[0][8] * pca.iloc[i][26]
            tr_y = kr_y - part_mass.iloc[0][8] * pca.iloc[i][25] - part_mass.iloc[0][8] * g
            tl_z = kl_z - part_mass.iloc[0][9] * pca.iloc[i][29]
            tl_y = kl_y - part_mass.iloc[0][9] * pca.iloc[i][28] - part_mass.iloc[0][9] * g
            knee_rz.append(kr_z)
            knee_ry.append(kr_y)
            knee_lz.append(kl_z)
            knee_ly.append(kl_y)
            thigh_rz.append(tr_z)
            thigh_ry.append(tr_y)
            thigh_lz.append(tl_z)
            thigh_ly.append(tl_y)
        para1 = pd.DataFrame(knee_rz)
        para2 = pd.DataFrame(knee_ry)
        para3 = pd.DataFrame(knee_lz)
        para4 = pd.DataFrame(knee_ly)
        para5 = pd.DataFrame(thigh_rz)
        para6 = pd.DataFrame(thigh_ry)
        para7 = pd.DataFrame(thigh_lz)
        para8 = pd.DataFrame(thigh_ly)
        pforce = pd.concat([para1, para2, para3, para4, para5, para6, para7, para8], axis = 1)
        pforce.columns = ["knee_rz", "knee_ry", "knee_lz", "knee_ly", "thigh_rz", "thigh_ry", "thigh_lz", "thigh_ly"]
        #pforce.to_csv(f'Data/Output/{n}/knee_thforce.csv')
        print('関節反力')

        # 膝，腰の関節角度
        # 関節座標の読み込み
        df = df
        dfc = df.loc[:, 'head_x':'left_ear_z']
        # データ列の切り出し
        iterables = [
            ["head", "neck",
                "right_shoulder", "right_elbow", "right_wrist",
                "left_shoulder", "left_elbow", "left_wrist",
                "right_waist", "right_knee", "right_foot",
                "left_waist", "left_knee", "left_foot",
                "right_eye", "left_eye",
                "right_ear", "left_ear"
            ],
            ['x', 'y', 'z']]
        dfc.columns = pd.MultiIndex.from_product(iterables, names=["parts", "coordinates"])
        dfs = []

        # パーツ間毎に処理
        target_pair = [('right_waist', 'right_knee'),
                        ('right_knee', 'right_foot'),
                        ('left_waist', 'left_knee'),
                        ('left_knee', 'left_foot')]

        for start_part, end_part in target_pair:
            # 始点座標
            sp = dfc.loc[:, (start_part, slice(None))].droplevel(0, axis=1)
            # 終点座標
            ep = dfc.loc[:, (end_part, slice(None))].droplevel(0, axis=1)
            # ベクトルを求める
            vec = ep - sp
            # 座標変換
            y_vec = -vec.loc[:,'y']
            z_vec = vec.loc[:,'z']
            #theta_rad = np.arctan2(z_vec,y_vec)
            
            if start_part == 'right_waist' or start_part == 'left_waist':
                theta_rad = np.arctan2(z_vec,y_vec)
            elif start_part == 'right_knee' or start_part == 'left_knee':
                theta_rad = np.arctan2(-z_vec,y_vec)
            
            
            # 結果をデータフレームに格納
            tmp = pd.DataFrame({'theta': theta_rad})
            tmp.columns = pd.MultiIndex.from_product([[f'FROM {start_part} TO {end_part}'],
                                                        ["theta"]])
            dfs.append(tmp)
        # すべての結果を結合
        angle_df = pd.concat(dfs, axis=1)
        #angle_df.transpose().reset_index(level=1, drop=True).transpose()
        #levels = angle_df.columns.levels
        angle_df.columns = angle_df.columns.droplevel(1)
        angle_df.to_csv(rf'{filepath3}\{subject}_kakudo.csv')
        print('関節角度')

        # 角加速度
        angle_vel = centerdiff(angle_df, 1/fps)
        angle_vel_kalman = kalman_def2.filtered_kalman(angle_vel)
        angle_accel = centerdiff(angle_vel_kalman, 1/fps)
        angle_accel = kalman_def2.filtered_kalman(angle_accel)
        print('関節角加速度')


        '''
        angle_accel2 = []
        # 2階中心差分法で角加速度求める
        for i in  range(1,len(angle_df)-1):
            for j in range(4):
                theta_accel = (angle_df.iloc[i-1][j] + angle_df.iloc[i+1][j] - 2*angle_df.iloc[i][j])*fps*fps
                angle_accel2.append(theta_accel)
        '''            
        #angle_accel = np.array(angle_accel2)
        #angle_accel = angle_accel.reshape(-1, 4)
        #angle_accel = pd.DataFrame(angle_accel)
        angle_vel_kalman.columns = angle_df.columns
        angle_vel_kalman.to_csv(rf'{filepath3}\{subject}_angle_velocity.csv')
        angle_accel.columns = angle_df.columns
        angle_accel.to_csv(rf'{filepath3}\{subject}_angle_accel.csv')
        # ローパスフィルタ
        #angle_accel = lowpass_def.csv_filter(angle_accel,0.7,2.0,type ='lp')

                
        # 慣性モーメント推定
        # 男子
        AE_moment_MALE = [-2043.38, 5547.75, 10.6498,
                        -1174.66, 3048.1, 5.19169]

        # 女子
        AE_moment_FEMALE = [-1851.78, 4347.35, 17.6609,
                        -830.815, 2342.82, 4.75943]

        header = ['r_thigh', 'l_thigh', 'r_crus', 'l_crus']
        coordinate = ['y']
        csv_header = [(coor + '_' + name)
                        for name in header for coor in coordinate]

        thigh =[]
        shin = []
        if gender == 'male':
            th = (AE_moment_MALE[0] + AE_moment_MALE[1] * m + AE_moment_MALE[2] * body_list[0])/10000
            thigh.append(th)
            sh = (AE_moment_MALE[3] + AE_moment_MALE[4] * m + AE_moment_MALE[5] * body_list[1])/10000
            shin.append(sh)
            
        elif gender == 'female':
            th = (AE_moment_FEMALE[0] + AE_moment_FEMALE[1] * m + AE_moment_FEMALE[2] * body_list[0])/10000
            thigh.append(th)
            sh = (AE_moment_FEMALE[3] + AE_moment_FEMALE[4] * m + AE_moment_FEMALE[5] * body_list[1])/10000
            shin.append(sh)

        thigh = pd.DataFrame(thigh)
        shin = pd.DataFrame(shin)
        thigh.columns = ['thigh']
        shin.columns = ['shin']
        in_moment = pd.concat([thigh, shin], axis = 1)
        in_moment.to_csv(rf'{filepath4}\{subject}_in_moment.csv')
        print('慣性モーメント')

        # 部分重心と関節の長さについて
        df = df
        # 部分重心
        part_com = part_com 

        # ヘッダー
        CoM_header_2 = ['r_thigh', 'l_thigh', 'r_crus', 'l_crus', 'r_foot', 'l_foot']
        coordinate_2 = ['x', 'y', 'z']
        csv_header_2 = [(coor + '_' + name)
                        for name in CoM_header_2 for coor in coordinate_2]
        CoM_header_3 = ['r_waist', 'r_knee', 'r_foot', 'l_waist', 'l_knee', 'l_foot']
        coordinate_3 = ['x', 'y', 'z']
        csv_header_3 = [(coor + '_' + name)
                        for name in CoM_header_3 for coor in coordinate_3]

        # 1つ目，関節座標
        df_1 = df.loc[:, 'right_waist_x':'right_foot_z']
        df_2 = df.loc[:, 'left_waist_x':'left_foot_z']
        pc1 = []
        pc1.append(df_1)
        pc1.append(df_2)
        pca1 = pd.concat(pc1, axis=1)
        pca1.columns = csv_header_3
        df_3 = pca1.loc[:, 'x_r_waist':'z_r_waist']
        df_4 = pca1.loc[:, 'x_r_knee':'z_r_knee']
        df_5 = pca1.loc[:, 'x_r_foot':'z_r_foot']
        df_6 = pca1.loc[:, 'x_l_waist':'z_l_waist']
        df_7 = pca1.loc[:, 'x_l_knee':'z_l_knee']
        df_8 = pca1.loc[:, 'x_l_foot':'z_l_foot']
        # 並びを変える
        pca = pd.concat((df_3, df_6, df_4, df_7, df_5, df_8), axis=1)
        pca.columns = csv_header_2

        # 2つ目，部分質量位置
        pcb = part_com.loc[:, 'x_r_thigh':'z_l_foot']
        #pc2 = []
        #pc2.append(df2_1)
        #pcb = pd.concat(pc2, axis=1)
        #pcb.drop
        #Lvec = pca.sub(pcb, axis=0)
        #Lvec = pca[0:].subtract(pcb, fill_value=0)
        # 関節座標から部分質量位置を引く
        Lvec = pca - pcb

        Lvec_before = pd.concat((Lvec, df_3, df_6, df_4, df_7, df_5, df_8), axis=1)
        #Lvec_before.to_csv(f'{filename}_Lbefore.csv')

        rt1 = []
        srt1 = []
        lt1 = []
        slt1 = []
        rc1 = []
        src1 = []
        lc1 = []
        slc1 = []

        # yz平面での部分重心との距離
        for i in range(len(Lvec_before)):
            rxt = Lvec_before.iloc[i][0]
            ryt = Lvec_before.iloc[i][1]
            rzt = Lvec_before.iloc[i][2]
            lxt = Lvec_before.iloc[i][3]
            lyt = Lvec_before.iloc[i][4]
            lzt = Lvec_before.iloc[i][5]
            rxc = Lvec_before.iloc[i][6]
            ryc = Lvec_before.iloc[i][7]
            rzc = Lvec_before.iloc[i][8]
            lxc = Lvec_before.iloc[i][9]
            lyc = Lvec_before.iloc[i][10]
            lzc = Lvec_before.iloc[i][11]
            # 足首部分差は使わない→そもそも0
            #ryf = Lvec_before.iloc[i][13]
            #rzf = Lvec_before.iloc[i][14]
            #lyf = Lvec_before.iloc[i][16]
            #lzf = Lvec_before.iloc[i][17]
            rxw = Lvec_before.iloc[i][18]
            ryw = Lvec_before.iloc[i][19]
            rzw = Lvec_before.iloc[i][20]
            lxw = Lvec_before.iloc[i][21]
            lyw = Lvec_before.iloc[i][22]
            lzw = Lvec_before.iloc[i][23]
            rxk = Lvec_before.iloc[i][24]
            ryk = Lvec_before.iloc[i][25]
            rzk = Lvec_before.iloc[i][26]
            lxk = Lvec_before.iloc[i][27]
            lyk = Lvec_before.iloc[i][28]
            lzk = Lvec_before.iloc[i][29]
            rxf = Lvec_before.iloc[i][30]
            ryf = Lvec_before.iloc[i][31]
            rzf = Lvec_before.iloc[i][32]
            lxf = Lvec_before.iloc[i][33]
            lyf = Lvec_before.iloc[i][34]
            lzf = Lvec_before.iloc[i][35]
            # 右大腿部分中心と腰の距離
            rt = math.sqrt(pow(rxt,2)+pow(ryt,2)+pow(rzt,2))
            # 右大腿の長さ
            rw = math.sqrt(pow(rxw-rxk,2)+pow(ryw-ryk,2)+pow(rzw-rzk,2))
            # 右大腿部分中心と膝の距離
            srt = rw -rt
            lt = math.sqrt(pow(lxt,2)+pow(lyt,2)+pow(lzt,2))
            lw = math.sqrt(pow(lxw-lxk,2)+pow(lyw-lyk,2)+pow(lzw-lzk,2))
            slt = lw -lt
            rc = math.sqrt(pow(rxc,2)+pow(ryc,2)+pow(rzc,2))
            rk = math.sqrt(pow(rxk-rxf,2)+pow(ryk-ryf,2)+pow(rzk-rzf,2))
            src = rk -rc
            lc = math.sqrt(pow(lxc,2)+pow(lyc,2)+pow(lzc,2))
            lk = math.sqrt(pow(lxk-lxf,2)+pow(lyk-lyf,2)+pow(lzk-lzf,2))
            slc = lk -lc
            rt1.append(rt)
            srt1.append(srt)
            lt1.append(lt)
            slt1.append(slt)
            rc1.append(rc)
            src1.append(src)
            lc1.append(lc)
            slc1.append(slc)

        rt = pd.DataFrame(rt1)
        srt = pd.DataFrame(srt1)
        lt = pd.DataFrame(lt1)
        slt = pd.DataFrame(slt1)
        rc = pd.DataFrame(rc1)
        src = pd.DataFrame(src1)
        lc = pd.DataFrame(lc1)
        slc = pd.DataFrame(slc1)
        kyori = pd.concat([srt, rt, slt, lt, src, rc, slc, lc], axis =1)
        kyori.columns = ["r_waistg_knee", "r_waist_waistg",
                            "l_waistg_knee", "l_waist_waistg",
                            "r_kneeg_ankle", "r_knee_kneeg",
                            "l_kneeg_ankle", "l_knee_kneeg"]
        kyori.to_csv(rf'{filepath3}\{subject}_kyori.csv')
        print('距離')

        in_moment = in_moment
        #print('慣性モーメント')

        angle_accel = angle_accel
        #print('角加速度')
        angle_accel.to_csv(rf'{filepath3}\{subject}_angle_acceleration.csv')

        ankle_moment = ankle_moment
        #print('足首モーメント')
        # ankle_moment.to_csv(f'Data/Output/{Name}/ankle_moment.csv')

        gf = gf
        #print('床反力')
        gf.to_csv(rf'{filepath3}\{subject}_floor_reaction.csv')

        pforce = pforce
        #print('関節反力')
        pforce.to_csv(rf'{filepath3}\{subject}_joint_reaction.csv')

        angle_df = kalman_def2.filtered_kalman(angle_df)
        #print('角度')
        angle_df.to_csv(rf'{filepath3}\{subject}_angle.csv')

        kyori = kyori
        #print('距離')
        kyori.to_csv(rf'{filepath3}\{subject}_distance.csv')

        knee_moment_r = []
        knee_moment_l = []
        knee_moment = []
        waist_moment_r = []
        waist_moment_l = []
        waist_moment = []

        # 膝関節モーメント, 腰関節モーメント
        for i in range(len(pforce)):
            
            knee_mr = (ankle_moment.iloc[i][0]
                        -in_moment.iloc[0][1]*angle_accel.iloc[i][1]
                        +gf.iloc[i][0]*kyori.iloc[i][4] * abs(math.cos(angle_df.iloc[i][1]))
                        -gf.iloc[i][1]*kyori.iloc[i][4] * abs(math.sin(angle_df.iloc[i][1]))
                        +pforce.iloc[i][0]*kyori.iloc[i][5]*abs(math.cos(angle_df.iloc[i][1]))
                        -pforce.iloc[i][1]*kyori.iloc[i][5]*abs(math.sin(angle_df.iloc[i][1])))
            
            knee_ml = (ankle_moment.iloc[i][1]
                        -in_moment.iloc[0][1]*angle_accel.iloc[i][3]
                        +gf.iloc[i][2]*kyori.iloc[i][6] * abs(math.cos(float(angle_df.iloc[i][3])))
                        -gf.iloc[i][3]*kyori.iloc[i][6] * abs(math.sin(float(angle_df.iloc[i][3])))
                        +pforce.iloc[i][2]*kyori.iloc[i][7]* abs(math.cos(float(angle_df.iloc[i][3])))
                        -pforce.iloc[i][3]*kyori.iloc[i][7]* abs(math.sin(float(angle_df.iloc[i][3]))))
            
            knee_m = knee_mr + knee_ml
            knee_moment.append(knee_m)
            knee_moment_r.append(knee_mr)
            knee_moment_l.append(knee_ml)

            waist_mr = (knee_mr
                        -in_moment.iloc[0][0]*angle_accel.iloc[i][0]
                        +pforce.iloc[i][0]*kyori.iloc[i][0]*abs(math.cos(float(angle_df.iloc[i][0])))
                        -pforce.iloc[i][1]*kyori.iloc[i][0]*abs(math.sin(float(angle_df.iloc[i][0])))
                        +pforce.iloc[i][4]*kyori.iloc[i][1]*abs(math.cos(float(angle_df.iloc[i][0])))
                        -pforce.iloc[i][5]*kyori.iloc[i][1]*abs(math.sin(float(angle_df.iloc[i][0]))))
            waist_ml = (knee_ml
                        -in_moment.iloc[0][0]*angle_accel.iloc[i][2]
                        +pforce.iloc[i][2]*kyori.iloc[i][2]*abs(math.cos(float(angle_df.iloc[i][2])))
                        -pforce.iloc[i][3]*kyori.iloc[i][2]*abs(math.sin(float(angle_df.iloc[i][2])))
                        +pforce.iloc[i][6]*kyori.iloc[i][3]*abs(math.cos(float(angle_df.iloc[i][2])))
                        -pforce.iloc[i][7]*kyori.iloc[i][3]*abs(math.sin(float(angle_df.iloc[i][2]))))
            
            waist_m = waist_mr + waist_ml
            waist_moment.append(waist_m)
            waist_moment_r.append(waist_mr)
            waist_moment_l.append(waist_ml)

        knee_moment_r = pd.DataFrame(knee_moment_r)
        knee_moment_l = pd.DataFrame(knee_moment_l)
        waist_moment_r = pd.DataFrame(waist_moment_r)
        waist_moment_l = pd.DataFrame(waist_moment_l)

        knee_moment = pd.DataFrame(knee_moment)
        waist_moment = pd.DataFrame(waist_moment)

        joint_moment = pd.concat([knee_moment, waist_moment], axis = 1)
        joint_moment.columns = ["knee_moment", "waist_moment"]
        joint_moment_rl = pd.concat([knee_moment_r,knee_moment_l,
                                        waist_moment_r,waist_moment_l], axis = 1)
        # 関節トルク保存
        joint_moment_rl.columns = ['knee_moment_r','knee_moment_l',
                                        'waist_moment_r','waist_moment_l']
        joint_moment_rl.to_csv(rf'{filepath4}\{subject}_joint_moment_rl.csv')
        waist_moment.to_csv(rf'{filepath4}\{subject}_Waist_moment.csv')
        knee_moment.to_csv(rf'{filepath4}\{subject}_Knee_moment.csv')
