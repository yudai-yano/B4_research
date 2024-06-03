import math
import numpy as np

def angle (line:list,line2:list):
    row = len(line)
    angle = []
    
    for i in range(row):
        if line2[i] >= 0:
            y = line[i]
            x = line2[i]
            θ = np.arctan2(y,x)
            
        elif line2[i] < 0:
            y = line[i]
            x = line2[i]
            a = np.arctan2(y,x)
            if a >= 0:
                θ = a - math.pi
                
            elif a < 0:
                θ = a + math.pi
        
        angle.append(θ)
        
    return angle

def angle_individual (value1:float,value2:float):
    if value2 >= 0:
        y = value1
        x = value2
        θ = np.arctan2(y,x)
            
    elif value2 < 0:
        y = value1
        x = value2
        a = np.arctan2(y,x)
        if a >= 0:
            θ = a - math.pi
                
        elif a < 0:
            θ = a + math.pi
        
    angle = θ
        
    return angle

def opposition (line:list):
    row = len(line)
    angle = []
    for i in range(row):
        value = line[i] 
        if value >= 0:
            θ = value - math.pi
            
        elif value < 0:
            θ = value + math.pi
        angle.append(θ)
        
    return angle

