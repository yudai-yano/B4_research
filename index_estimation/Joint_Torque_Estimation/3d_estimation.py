# Program for 3-dim skelton point estimation

# import configuration info
import config

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import os.path for file path manipulation
import os.path
# Import glob for fetching bagfile folder series
import glob
# Import datetime for calc time
import datetime
# Import csv for save pose data
import csv

# data processing
import pandas as pd
import sys
from sys import platform
import os
import time
import traceback
import itertools
from PIL import Image
from natsort import natsorted
from logging import StreamHandler, Formatter, INFO, getLogger
import json

# for AlphaPose
import subprocess

''' -------------------------------------- Settings -------------------------------------- '''

# massage when error occurs
class myException( Exception ):
    def __init__( self, e, message ):
 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split( exc_tb.tb_frame.f_code.co_filename )[ 1 ]
        raise type( e )( str( sys.exc_info()[1] ) + ' : [' + str( fname ) + '][' + str( exc_tb.tb_lineno ) + ']' + message )


def clipping_background():
    raise NotImplementedError

# image sizing
def width_clip(img_width, value):
    if value <= 0:
        return 0
    elif value >= img_width:
        return img_width - 1
    return value


def height_clip(img_height, value):
    if value < 0:
        return 0
    elif value >= img_height:
        return img_height - 1
    return value

def is_inshape(width: int, height: int, pixels) -> bool:
    bool_inshape = True
    if not 0 <= pixels[0] < width:
        bool_inshape = False
    if not 0 <= pixels[1] < height:
        bool_inshape = False
    return bool_inshape

def main(path):
    dir_name = os.path.split(os.path.dirname(path))[-1]
    result = rf'C:\Users\sk122\inheriting\test\res\{dir_name}'
    os.makedirs(result, exist_ok=True)
    file_name = str(os.path.splitext(os.path.basename(path))[0])

    # if bagfile path doesn't exist, output alert and stop this process.
    if not bagfile_folder_path:
        print("file doesn't exist.")
        print("input files into folder")
        exit()

    # if source file is not bagfile or bagfile doesnt exist,
    # close this program
    if os.path.splitext(bagfile_folder_path[0])[1] != ".bag":
        print(".bag file dont exist in apps/src/ folder.")
        print("Please input bagfile in that folder.")
        exit()

    # for csv header
    coord = ['x', 'y', 'z']
    header = ['joint' + str(i//3) + '_' + str(coord[i % 3])
              for i in range(cfg["bone_number"] * len(coord))]
    header.append('framecount')
    header.append('timedelta[ms]')
    header.append('UNIXtimestamp[ms]')
    header.append('count')
    header.append('datetime')

    # for output file
    # image output
    outputfolder = rf'{result}\images\{file_name}'
    colorf = rf'{outputfolder}\color'
    bg_removef = rf"{outputfolder}\bg_removed"
    depthf = rf"{outputfolder}\depth"
    
    os.makedirs(rf"{outputfolder}", exist_ok=True)
    os.makedirs(rf"{colorf}", exist_ok = True)
    os.makedirs(rf"{bg_removef}", exist_ok = True)
    os.makedirs(rf"{depthf}", exist_ok = True)
    os.makedirs(rf'{result}\csvdata', exist_ok=True)
    
    # AlphaPose result (coor data)
    csvname = rf'{result}\csvdata\{file_name}.csv'

    # get now date and get csvname from datetime
    dt = datetime.datetime.now()

    # get file stream for csv file
    with open(csvname, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        try:
    
            # write header to csv
            writer.writerow(header)

            try:
                # create pipeline
                pipeline = rs.pipeline()

                # Create a config object
                rs_cfg = rs.config()

                #! need change to enable user to switch processing bagfile
                #! change to apply to folder instead file
                # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
                
                rs.config.enable_device_from_file(
                    rs_cfg, path, repeat_playback=False)

                # Start streaming from file
                profile = pipeline.start(rs_cfg)

                # no real time processing setting
                playback = profile.get_device().as_playback()
                playback.set_real_time(False)

                # get intrinsics
                dpt_intr = rs.video_stream_profile(
                    profile.get_stream(rs.stream.depth)).get_intrinsics()
                clo_intr = rs.video_stream_profile(
                    profile.get_stream(rs.stream.color)).get_intrinsics()
                
                # depth filter preparing
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                thres_fil = rs.threshold_filter(cfg["thres_min"], cfg["thres_max"])

                # Get product line for setting a supporting resolution
                device = profile.get_device()
                device_product_line = str(device.get_info(rs.camera_info.product_line))
                print(f"product name : {device_product_line}")

                found_rgb = False
                for s in device.sensors:
                    if s.get_info(rs.camera_info.name) == 'RGB Camera':
                        found_rgb = True
                        break

                if not found_rgb:
                    print("This program need depth camera with color sensor.")
                    exit(0)

                # Create an align object
                # rs.align allows us to perform alignment of depth frames to others frames
                # The "align_to" is the stream type to which we plan to align depth frames.
                align_to = rs.stream.color
                align = rs.align(align_to)

                # counter for csv
                framecount = cfg["initial_count"]
                count = cfg["initial_count"]
                timestamp = cfg["initial_timestamp"]
                count = -1

                # calc background cutting value
                clip_distance_forw = cfg["fill_min"] / depth_scale
                clip_distance_back = cfg["fill_max"] / depth_scale

                #cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

                # Streaming loop
                while True:
                    # Get frameset of depth
                    frames = pipeline.wait_for_frames()

                    # Get frame meta data
                    framecount = frames.get_frame_number()
                    temp = timestamp
                    timestamp = frames.get_timestamp()
                    delta = timestamp - temp
                    backend_timestamp = frames.get_frame_metadata(
                        rs.frame_metadata_value.backend_timestamp
                    )
                    # get now date and get csvname from datetime
                    dt = datetime.datetime.now()

                    count += 1
                    print(file_name + ': ' + str(count))

                    # decimarion_filter
                    decimate = rs.decimation_filter()
                    decimate.set_option(rs.option.filter_magnitude, 1)
                    
                    # spatial_filter
                    spatial = rs.spatial_filter()
                    spatial.set_option(rs.option.filter_magnitude, 1)
                    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
                    spatial.set_option(rs.option.filter_smooth_delta, 50)
                    
                    # hole_filling_filter
                    hole_filling = rs.hole_filling_filter()
                    
                    # disparity
                    depth_to_disparity = rs.disparity_transform(True)
                    disparity_to_depth = rs.disparity_transform(False)
                    
                    # Align the depth frame to color frame
                    aligned_frames = align.process(frames)

                    # Get depth frame and color frame
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    # Validate that both frames are valid
                    if not color_frame or not depth_frame:
                        continue

                    # Filter
                    filter_frame = decimate.process(depth_frame)
                    filter_frame = depth_to_disparity.process(filter_frame)
                    filter_frame = spatial.process(filter_frame)
                    filter_frame = disparity_to_depth.process(filter_frame)
                    filter_frame = hole_filling.process(filter_frame)
                    depth_frame = filter_frame
                                
                    # Depth and color image 
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    bg_remove_color = color_image.copy()

                    #! insert LiDAR post processing
                    filted_frames = thres_fil.process(depth_frame)
                    filted_frames = filted_frames.as_depth_frame()
                    
                    # background remove
                    filted_depth_frame = filted_frames.get_data()
                    filted_depth_image = np.asanyarray(filted_depth_frame)

                    depth_image_3d = np.dstack((filted_depth_image, filted_depth_image, filted_depth_image))                    

                    bg_remove_color[
                        ((depth_image_3d <= clip_distance_forw) |
                        (depth_image_3d >= clip_distance_back))
                    ] = cfg["fill_color"]

                    # RGB composition converting (RGB to BGR)
                    bg_remove_color = cv2.cvtColor(bg_remove_color, cv2.COLOR_RGB2BGR)
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)
                    
                    color_im = 'color-image_' + str(count)
                    bg_remove_im = 'bg_removed-image_' + str(count)
                    depth_im = 'depth_' + str(count)

                    cv2.imwrite(rf"{colorf}\{color_im}.png", color_image)
                    cv2.imwrite(rf"{bg_removef}\{bg_remove_im}.png", bg_remove_color)
                    cv2.imwrite(rf"{depthf}\{depth_im}.png", bg_remove_color)

                    # make folder for alphapose
                    os.makedirs(rf'{result}\alphapose', exist_ok=True)
                    alphapose_res = rf'test\res\{dir_name}\alphapose\{file_name}'
                    os.makedirs(alphapose_res, exist_ok=True)

                    # 2-dim estimation by AlphaPose
                    # need improvement
                    # Although a background-removed image should be used, a color image is used here as it is.
                    inputimage = rf'test\res\{dir_name}\images\{file_name}\color'
                    subprocess.run([r"program\tes.cmd", inputimage, alphapose_res, color_im])   

                    # coordination data extraction
                    json_path = rf'{alphapose_res}\sep-json\{color_im}.json'
                    
                    if os.path.exists(json_path) == True:
                        json_open = open(json_path, 'r')
                        json_load = json.load(json_open)

                        # extract 18 points of x, y coor
                        body_joints = []
                        for j in range(len(json_load["people"])):
                            keypoints = json_load["people"][0]["pose_keypoints_2d"]
                            x = keypoints[::3]
                            y = keypoints[1::3]
                            x_int = [int(num) for num in x]
                            y_int = [int(num) for num in y]
                            body_joints.append(x_int)
                            body_joints.append(y_int)

                        body_joints_arr = np.array(body_joints)    
                        if body_joints_arr.shape != ((2, 18)):
                            body_joints = np.zeros((1, 18, 3))
                        else:
                            pass
                    # If alphapose wasn't applied, put 0 instead. 
                    else:
                        body_joints_arr = np.zeros((2, 18))
                        body_joints_arr = np.asarray(body_joints_arr, dtype=int)

                    # 3-dim estimation
                    # get image width and height
                    height, width, _ = color_image.shape

                    # convert x,y coor
                    joint_xy_pixels = [ [
                        np.round(body_joints_arr[0][j]),
                        np.round(body_joints_arr[1][j])
                        ] for j in range(18)]

                    # depth value generator in joint pixel
                    # cliped joints xy pixels generator (not in memory)
                    clip_xy_pixels = (
                        [width_clip(width, xy[0]), height_clip(height, xy[1])]
                        for xy in joint_xy_pixels
                    )

                    # depth value generator in joint pixel
                    depth_value = (
                        filted_frames.get_distance(
                            pixel[0], pixel[1]
                        ) for pixel in clip_xy_pixels
                    )
                    #depth_value = (filted_frames.get_distance(x, y))

                    # realsense 3D points in camera coordinates
                    points = (
                        rs.rs2_deproject_pixel_to_point(
                            clo_intr, joint_xy_pixels[i], depth
                        ) for i, depth in enumerate(depth_value)
                    )
                    
                    # if joint isn't in image area, convert point [-1,-1,-1]
                    points_iter = itertools.chain.from_iterable(
                        (point if is_inshape(width, height, joint_xy_pixels[i]
                                            ) else [-1, -1, -1]
                        ) for i, point in enumerate(points)
                    )
                    other = [framecount, delta, timestamp, count, dt]
                    points_iter = list(points_iter)
                    points_iter.extend(other)

                    # datalist_for_csv.append(list(points_iter))
                    writer.writerow(points_iter)

                    # # show depth data
                    
                    #depth_colormap = cv2.applyColorMap(
                    #cv2.convertScaleAbs(filted_depth_image, alpha=1), cv2.COLORMAP_JET)
                    #cv2.imshow("Camera Stream", images)
                    #key = cv2.waitKey(1)
                    # depth_colormap_frame = rs.colorizer().colorize(filted_frames)
                    # depth_colormap_image = np.asanyarray(depth_colormap_frame.get_data())

                    # depth_colormap_frame2 = rs.colorizer().colorize(filted_frames)
                    # depth_colormap_image2 = np.asanyarray(depth_colormap_frame2.get_data())

                
                    
            except Exception as ex:
                print(f"Exception occured: \'{ex}\'")
                
                traceback.print_exc()
            
            except(RuntimeError):
                pass    

            finally:
                pipeline.stop()
                print('All finished.')
                pass
    
        except:
            print("Unexpected error:", str(sys.exc_info()[0]))
            print("Unexpected error:", str(sys.exc_info()[1]))
            print("Unexpected error:", str(sys.exc_info()[2]))
        finally:
            pass


# load config data
cfg = config.dic_config
# load bagfile folder
bagfile_folder_path = glob.glob(cfg["source_folder"])
path_names = natsorted(bagfile_folder_path)


if __name__ == '__main__':
    files = natsorted(bagfile_folder_path)
    #file_names = [os.path.split(i)[1] for i in path_names]
    for i in range(len(files)):
        path = files[i]
        print(path)
        main(path)