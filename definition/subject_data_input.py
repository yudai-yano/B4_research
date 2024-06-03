import pandas as pd

#ファイルにある名前からsubject_dataの名前に変換
def subject_name_input(file_name):
    #LENGTHデータの読み込み
    length_csv = rf'C:\Users\yota0\Desktop\yano\program\python\base_data\LENGTH_WORKSET.csv'
    length_data = pd.read_csv(rf'{length_csv}')
    #subject_dataの読み込み
    subject_csv = rf'C:\Users\yota0\Documents\Yota\githubrepos\poseestimate_mediapipe\poseestimate_mediapipe\config\SUBJECTS_DATA.csv'
    subject_data = pd.read_csv(rf'{subject_csv}')
    
    #それぞれのidと名前を読み込み
    length_name_data = length_data.loc[:,'filename']
    length_id_data = length_data.loc[:,"subject_id"]
    #いらなくなった
    #subject_id_data = subject_data.loc[:,"segment_id"]
    subject_name_data = subject_data.loc[:,"name"]
    
    #length_dataのインデックスの名称変換
    input_row = length_name_data.shape[0]
    for i in range(input_row):
        length_id_data.rename(index={i: length_name_data[i]} , inplace=True)
        
    #ファイルの名前からsubject_idに
    input_id = length_id_data[file_name]
    
    if input_id == None:
        print("id is not supported. Please enter id")
        
        Comment = "finish"
        return(Comment)
    
    else:
        #subject_dataのインデックスの名称変換
        row = subject_name_data.shape[0]
        #ここは少し微妙
        subject_name_data.index = range(1 , row + 1)
        
            
        #subject_idから被験者の名前に
        output_name = subject_name_data.loc[input_id]
        
        return(output_name)

def subject_data_input(name):
    #各部位の長さファイルの読み込み
    file_path = rf'C:\Users\yota0\Documents\Yota\githubrepos\poseestimate_mediapipe\poseestimate_mediapipe\config\SUBJECTS_DATA.csv'
    csv_frame = pd.read_csv(rf'{file_path}', header=0)
    name_data = csv_frame.loc[:, 'name']
    data_frame = csv_frame.loc[:, 'age':'l_thigh_circum']


    #名前と各部位のデータを連動
    select_dict = {}
    name_list = []
    row = name_data.shape[0]
    for i in range(row):
        data_frame.rename(index={i: name_data.iloc[i]} , inplace=True)
        select_dict.update({name_data.iloc[i] :i})
        name_list.append(name_data.iloc[i])
        
    subject_data = data_frame.loc[name]
    
    return subject_data