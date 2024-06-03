import glob
import cv2
import os
from natsort import natsorted

img_array = []
n = 'kikuchi_sayaka'

for index, file in enumerate(natsorted(glob.glob(rf"C:\Users\sk122\mlproj\res\deguchi_minoru\alphapose\70RM_1\vis\*"))):
    img = cv2.imread(file)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

size = (640, 480)
videoname = rf'C:\Users\sk122\mlproj\res\deguchi_minoru\alphapose\70RM_1st.mp4'
out = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, size)

for i in range(len(img_array)):


    out.write(img_array[i])
out.release()
