import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import max_muscular
import angle


v=np.array([[1,2,3],[4,5,6],[7,8,9]])
w=np.array([[4,5,6],[3,2,1],[9,8,7]])
result = np.cross(v, w, axis=1)

print(result)