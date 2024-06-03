import numpy as np
import pandas as pd
import angle

def angle_maximum_muscular_strength(coordinate = pd.DataFrame):
    
    iliopsoas_muscle = 79
    vastus_muscle = 96.3
    rectus_femoris_muscle = 50.7
    gluteus_maximus_muscle = 291.3
    biceps_femoris_muscle = 28.3
    semimembranosus_muscle = 46
    
    df = coordinate
    row_count = df.shape[0]
    my_DataFrame = pd.DataFrame()
    
    for i in range(row_count):
        
        summit_1_max_x = vastus_muscle *np.cos(df.iloc[i,0]) + rectus_femoris_muscle *np.cos(df.iloc[i,5]) + gluteus_maximus_muscle *np.cos(df.iloc[i,1])
        summit_1_max_y = vastus_muscle *np.sin(df.iloc[i,0]) + rectus_femoris_muscle *np.sin(df.iloc[i,5]) + gluteus_maximus_muscle *np.sin(df.iloc[i,1])
        summit_2_max_x = vastus_muscle *np.cos(df.iloc[i,0]) + gluteus_maximus_muscle *np.cos(df.iloc[i,1]) + semimembranosus_muscle *np.cos(df.iloc[i,2])
        summit_2_max_y = vastus_muscle *np.sin(df.iloc[i,0]) + gluteus_maximus_muscle *np.sin(df.iloc[i,1]) + semimembranosus_muscle *np.sin(df.iloc[i,2])
        summit_3_max_x = gluteus_maximus_muscle *np.cos(df.iloc[i,1]) + semimembranosus_muscle *np.cos(df.iloc[i,2]) + biceps_femoris_muscle *np.cos(df.iloc[i,3])
        summit_3_max_y = gluteus_maximus_muscle *np.sin(df.iloc[i,1]) + semimembranosus_muscle *np.sin(df.iloc[i,2]) + biceps_femoris_muscle *np.sin(df.iloc[i,3])
        summit_4_max_x = semimembranosus_muscle *np.cos(df.iloc[i,2]) + biceps_femoris_muscle *np.cos(df.iloc[i,3]) + iliopsoas_muscle *np.cos(df.iloc[i,4])
        summit_4_max_y = semimembranosus_muscle *np.sin(df.iloc[i,2]) + biceps_femoris_muscle *np.sin(df.iloc[i,3]) + iliopsoas_muscle *np.sin(df.iloc[i,4])
        summit_5_max_x = biceps_femoris_muscle *np.cos(df.iloc[i,3]) + iliopsoas_muscle *np.cos(df.iloc[i,4]) + rectus_femoris_muscle *np.cos(df.iloc[i,5])
        summit_5_max_y = biceps_femoris_muscle *np.sin(df.iloc[i,3]) + iliopsoas_muscle *np.sin(df.iloc[i,4]) + rectus_femoris_muscle *np.sin(df.iloc[i,5])
        summit_6_max_x = iliopsoas_muscle *np.cos(df.iloc[i,4]) + rectus_femoris_muscle *np.cos(df.iloc[i,5]) + vastus_muscle *np.cos(df.iloc[i,0])
        summit_6_max_y = iliopsoas_muscle *np.sin(df.iloc[i,4]) + rectus_femoris_muscle *np.sin(df.iloc[i,5]) + vastus_muscle *np.sin(df.iloc[i,0])
        
        square_1 = summit_1_max_x**2 + summit_1_max_y**2
        summit_1_max = np.sqrt(square_1)
        square_2 = summit_2_max_x**2 + summit_2_max_y**2
        summit_2_max = np.sqrt(square_2)
        square_3 = summit_3_max_x**2 + summit_3_max_y**2
        summit_3_max = np.sqrt(square_3)
        square_4 = summit_4_max_x**2 + summit_4_max_y**2
        summit_4_max = np.sqrt(square_4)
        square_5 = summit_5_max_x**2 + summit_5_max_y**2
        summit_5_max = np.sqrt(square_5)
        square_6 = summit_6_max_x**2 + summit_6_max_y**2
        summit_6_max = np.sqrt(square_6)
        
        summit_1_angle = np.arctan2(summit_1_max_y,summit_1_max_x)
        summit_2_angle = np.arctan2(summit_2_max_y,summit_2_max_x)
        summit_3_angle = np.arctan2(summit_3_max_y,summit_3_max_x)
        summit_4_angle = np.arctan2(summit_4_max_y,summit_4_max_x)
        summit_5_angle = np.arctan2(summit_5_max_y,summit_5_max_x)
        summit_6_angle = np.arctan2(summit_6_max_y,summit_6_max_x)
        
        my_list = [summit_1_max , summit_2_max , summit_3_max , summit_4_max , summit_5_max , summit_6_max , summit_1_angle , summit_2_angle , summit_3_angle , summit_4_angle , summit_5_angle , summit_6_angle]
        my_DataFrame = my_DataFrame.append(pd.Series(my_list), ignore_index=True)
    
    return my_DataFrame