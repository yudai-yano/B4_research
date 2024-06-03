import pandas as pd

def Vector_Direction_Distinction(list:list,name,list2:list,name2):
    data = {name:list,name2:list2}
    df = pd.DataFrame(data)
    
    sorted_df = df.sort_values(by=name, ascending=False)
    
    sorted_df = sorted_df.reset_index(drop=True)
    
    return sorted_df

def muscle_strength(a:int):
    if a == 1:
        my_list = [0,1,1,1,0,0]
        
    if a == 2:
        my_list = [0,1,0,1,0,1]
        
    if a == 3:
        my_list = [0,0,0,1,1,1]
        
    if a == 4:
        my_list = [1,0,0,0,1,1]
        
    if a == 5:
        my_list = [1,0,1,0,1,0]
        
    if a == 6:
        my_list = [1,1,1,0,0,0]
        
    return my_list