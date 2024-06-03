# First import library
import pyrealsense2 as rs
# Import OpenCV for easy image rendering
import cv2
# Import numpy for array processing
import numpy as np
# Import path processing for file path manipulation
import pathlib
import os

# Import config and control config
import configparser
import traceback


def main():

    config_ini = configparser.ConfigParser()
    config_ini.read('program/config.ini', encoding='utf-8')

    cwd = os.getcwd()
    parentpath = pathlib.Path(cwd)
    bagfolder = pathlib.Path(config_ini["MOVIETOOL"]["bagfiles_path"])
    bagfilelist = list(bagfolder.glob("*.bag"))

    out_path = list(parentpath.glob(config_ini["MOVIETOOL"]["movie_out_path"]))

    for bagfile_path in bagfilelist:
        mov_path = str(out_path[0]) + f'/{bagfile_path.stem}.avi'

        if os.path.exists(mov_path):
            continue

        try:

            # create pipeline
            pipeline = rs.pipeline()

            # Create a config object
            rs_cfg = rs.config()

            # Tell config that we will use a recorded device from file to be
            # used by the pipeline through playback.
            rs.config.enable_device_from_file(
                rs_cfg, str(bagfile_path), repeat_playback=False)

            # Start streaming from file
            profile = pipeline.start(rs_cfg)

            color_intrinsics = rs.video_stream_profile(
                profile.get_stream(rs.stream.color)).get_intrinsics()

            frame_width = str(color_intrinsics).split(' ')[1].split('x')[0]
            frame_height = str(color_intrinsics).split(' ')[1].split('x')[1]

            # no real time processing setting
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)

            # Get product line for setting a supporting resolution
            device = profile.get_device()
            device_product_line = str(
                device.get_info(rs.camera_info.product_line))
            print(f"product name : {device_product_line}")

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break

            if not found_rgb:
                print("This program need depth camera with color sensor.")
                raise Exception('RGB frame not found Exception')

            # Create an align object
            # rs.align allows us to perform alignment of
            # depth frames to others frames.
            # The "align_to" is the stream type to which
            # we plan to align depth frames.
            align_to = rs.stream.color
            align = rs.align(align_to)

            cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

            # VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                mov_path, fourcc, 30.0, (int(frame_width), int(frame_height)))

            # Streaming loop
            while True:
                # Get frameset of depth
                frames = pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get depth frame and color frame
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not color_frame or not depth_frame:
                    print("each frame doesnt exist!!")
                    continue

                # Colorize depth frame to jet colormap
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Render images:
                #   depth align to color on left
                #   depth on right
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((color_image, depth_colormap))

                # save for video
                writer.write(color_image)

                cv2.imshow("Camera Stream", images)
                key = cv2.waitKey(1)

                # if pressed escape exit program
                if key == 27 or key == ord('q'):
                    cv2.destroyAllWindows()
                    break

            writer.release()

        except RuntimeError as ex:
            print(f"Frame ended. : \'{ex}\'")
            writer.release()

        except Exception as ex:
            print(f"Exception occured: \'{ex}\' {ex.with_traceback}")
            traceback.print_exc()
            writer.release()


if __name__ == '__main__':
    main()
