import csv

def convert_csv_to_trc(csv_file_path, trc_file_path):
    # CSVファイルを読み込み、データをリストに格納する
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    # TRCファイルを作成し、データを書き込む
    with open(trc_file_path, 'w', encoding='utf-8') as trc_file:
        # ヘッダ行を書き込む
        trc_file.write('PathFileType\t4\t(X/Y/Z)\t' + trc_file_path + '\n')
        trc_file.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')

        # データの情報を書き込む
        num_frames = len(data) - 1
        num_markers = (len(data[0])- 1) // 3 
        trc_file.write('60\t60\t' + str(num_frames) + '\t' + str(num_markers) + '\tmm\t60\t1\t' + str(num_frames) + '\n')

        # 列ラベルを書き込む
        trc_file.write('Frame#\tTime\t')
        for i in range(1, num_markers + 1):
            trc_file.write('Marker' + str(i) + '\t\t\t')
        trc_file.write('\n')
        
        trc_file.write('\t\t')
        for i in range(1, num_markers + 1):
            trc_file.write('X' + str(i) + '\t' + 'Y' + str(i) + '\t' + 'Y' + str(i) + '\t')
        trc_file.write('\n')
        trc_file.write('\n')

        # データを書き込む
        for i in range(1, len(data)):
            frame_data = data[i]
            frame_num = str(i)
            frame_time = frame_data[0]
            trc_file.write(frame_num + '\t' + frame_time + '\t')
            for j in range(1, len(frame_data), 3):
                x = frame_data[j]
                y = frame_data[j + 1]
                z = frame_data[j + 2]
                trc_file.write(x + '\t' + y + '\t' + z + '\t')
            trc_file.write('\n')

    print('TRCファイルの変換が完了しました。')

# 使用例
csv_file_path = rf'C:\Users\yota0\Desktop\検証用\testfile.csv'  # 入力のCSVファイルパス
trc_file_path = rf'C:\Users\yota0\Desktop\検証用\testfile.trc'  # 出力のTRCファイルパス

convert_csv_to_trc(csv_file_path, trc_file_path)