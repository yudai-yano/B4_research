import pandas as pd

def sort(data , sort_index):
    df = pd.DataFrame(data)
    
    df = df.sort_values(by=str(sort_index))
    
    df = df.reset_index(drop=True)
    
    return(df)