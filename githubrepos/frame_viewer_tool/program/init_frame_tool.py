import cv2
import pathlib
import configparser

from monitor import Frame_monitor


def main():

    # get config path
    path = list(pathlib.Path(__file__).parent.glob('*.ini'))[0]

    # gen configparser object
    config_ini = configparser.ConfigParser()
    config_ini.read(path)

    # get movie folder path
    frametool_config = config_ini['FRAMETOOL']
    mov_folder = frametool_config.get('mov_folder_path')

    # get mov folder path object
    mov_folder_pathobj = pathlib.Path(mov_folder)
    movie_path_list = list(mov_folder_pathobj.iterdir())

    check_movie_found(movie_path_list)

    fileindex = input_file_number(movie_path_list)

    if fileindex == -1:
        exit()

    print(str(movie_path_list[fileindex]))

    cap = cv2.VideoCapture(str(movie_path_list[fileindex]))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'size: ({width}x{height}), fps: {fps}')

    frame_sum_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_monitor = Frame_monitor(frame_sum_number)
    listener = frame_monitor.listener
    listener.start()

    try:
        while (cap.isOpened()):
            listener.wait()

            if frame_monitor.update_frame:
                # set video frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_monitor.frame)

                # get video frame
                _ , frame = cap.read()

                # draw frame number to image
                cv2.putText(
                    frame,
                    text=f'{int(frame_monitor.frame)}',
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_4
                    )

                # show frame
                cv2.imshow('sample', frame)

                # update flag
                frame_monitor.update_frame = False

                # output text
                # TODO #3 framemonitorに移植
                frame_monitor._user_manual()

            if frame_monitor.loopend_flag:
                print('End this program')
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f'call exception : {e}')
    finally:
        cv2.destroyAllWindows()
        frame_monitor._output_savedata()
        listener.stop()
        cap.release()


def check_movie_found(movie_path_list):
    if len(movie_path_list) == 0:
        print('movie file doesnt exist.')
        print('end this program.')
        exit()


def input_file_number(path_list) -> int:

    # show number and file name.
    print('choose movie number from under selection')
    for i, path in enumerate(path_list):
        print(f'number : {i}, movie name : {path.name}')

    # select number
    try:
        print('Input number here (Press your keyboard and Enter if you end press) : ')
        select_num = int(input())

        if not (0 <= select_num < len(path_list)):
            raise ValueError

        return select_num

    except ValueError:
        print(
            f'invalid input. you must input number from 0 to {len(path_list) - 1}.')
        exit()


if __name__ == '__main__':
    main()
