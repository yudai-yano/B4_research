from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from scipy import ndimage
import cv2
import glob
import os
from natsort import natsorted

subject_name = 'deguchi_minoru'
folder_path = rf"C:\Users\sk122\mlproj\res\{subject_name}\Images_70RM_*"
trim_coor = (166, 452) # [min_x:max_x]

os.makedirs(rf"res\{subject_name}\trim_image", exist_ok=True)
img_folders = glob.glob(folder_path)

for progress, folder in enumerate(img_folders):
    folder_name = os.path.splitext(os.path.basename(folder))[0]
    os.makedirs(rf'res\{subject_name}\trim_image\{folder_name}', exist_ok=True)
    image_path = glob.glob(rf'{folder}\picture_*')
    path_names = natsorted(image_path)
    for img in path_names:
        img_name = os.path.basename(img)

        original_img = cv2.imread(img)
        trim_img = original_img[:, trim_coor[0]:trim_coor[1]]
        cv2.imwrite(rf'res\{subject_name}\trim_image\{folder_name}\{img_name}', trim_img)
    print(f'{progress+1}/{len(img_folders)} finished.')

# files = rf"C:\Users\sk122\mlproj\res\{subject_name}\Images_70RM_1\picture_*"
# for i in glob.glob(files):
#     img = cv2.imread(i)
#     trim_img = img[:, trim_coor[0]:trim_coor[1]]
#     filename = os.path.splitext(os.path.basename(i))[0]
#     cv2.imwrite(rf'C:\Users\sk122\mlproj\res\{subject_name}\Images_70RM_1\trim_{filename}.jpg', trim_img)
    

# for image_file in natsorted(glob.glob(image_path)):
#     folder_name = os.path.splitext(os.path.basename(image_file))[0]
#     images = rf'{image_file}\picture_*'
#     for image in glob.glob(images):
#         file_name = os.path.splitext(os.path.basename(images))[0]
#         # 画像読み込み
#         img = cv2.imread(images)
#         print(type(img))
#         # img[top : bottom, left : right]
#         # 画像の切り出し、保存
#         trim_img = img[:, trim_coor[0]:trim_coor[1]]
#         cv2.imwrite(rf"C:\Users\sk122\mlproj\res\{subject_name}\trimmed_image\{folder_name}\{file_name}.jpg", trim_img)
#     #     print(f'{file_name} saved.')
#     print(f'{folder_name} finished.')
