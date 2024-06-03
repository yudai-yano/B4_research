import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os

subject_name = 'deguchi_minoru'
bagfiles = glob.glob(rf'C:\Users\sk122\mlproj\data\22_11_09\bagfile\{subject_name}\*.bag')
num_bags = len(bagfiles)

for progress, bagfile in enumerate(bagfiles):
    filename = os.path.splitext(os.path.basename(bagfile))[0]
    dirname = os.path.split(os.path.dirname(bagfile))[-1]
    os.makedirs(rf"C:\Users\sk122\mlproj\res\{dirname}", exist_ok=True)
    mp4file = rf'C:\Users\sk122\mlproj\res\{dirname}\{filename}'+'.mp4'
    if os.path.isfile(mp4file):
        continue

    config = rs.config()
    config.enable_device_from_file(bagfile)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    device = profile.get_device()
    playback = device.as_playback()

    for stream in profile.get_streams():
        vprof = stream.as_video_stream_profile()
        if  vprof.format() == rs.format.rgb8:
            frame_rate = vprof.fps()
            size = (vprof.width(), vprof.height())

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(mp4file, fmt, frame_rate, size) # ライター作成

    print('{}/{}: '.format(progress+1, num_bags),bagfile, size, frame_rate)

    try:
        cur = -1
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())       
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            writer.write(color_image)

            next = playback.get_position()
            if next < cur:
                break
            cur = next

    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()

    video_path = mp4file
    cap = cv2.VideoCapture(video_path)


    # 1フレームごとに切り出し
    num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        os.makedirs(rf"C:\Users\sk122\mlproj\res\{dirname}\Images_{filename}", exist_ok=True)
        if ret == True:
            cv2.imwrite(rf"C:\Users\sk122\mlproj\res\{dirname}\Images_{filename}\picture_{num}"+".jpg",frame)
            print(f"save picture{num}"+".jpg")
            num += 1
        else:
            break
    print('All images are saved.')
    cap.release()

    dir = rf'C:\Users\sk122\mlproj\res\{dirname}\Images_{filename}'
    count_file = 0
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)

        if os.path.isfile(file_path):
            count_file += 1

    print(f'num: {count_file}')
