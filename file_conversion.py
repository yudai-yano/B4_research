from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
import glob
import shutil

'''元のフォルダの選択'''
root = Tk()
root.withdraw()
folderpath = askdirectory()

'''保存先の指定'''
Save_Folder = rf'F:\processing_data\bagfile'

'''フォルダかファイルかの処理'''
def check_file_or_folder(path):
    if os.path.isfile(path):
        #ファイルの際の処理
        copy_file(path , Save_Folder)
    elif os.path.isdir(path):
        #フォルダの際の処理
        Folder_Processing(path)
    else:
        print(f"{path} は存在しないか、ファイルでもフォルダーでもありません。")

'''フォルダ時の処理'''
def Folder_Processing(folder):
    for name in glob.glob(rf'{folder}\*'):
        check_file_or_folder(name)
        
'''ファイル時の処理'''
def copy_file(source_path, destination_folder):
    try:
        # ファイルをコピー
        shutil.copy(source_path, destination_folder)
        #print(f"ファイルがコピーされました。")
    except FileNotFoundError:
        print(f"指定されたファイルが見つかりません。")
    except PermissionError:
        print(f"コピー先フォルダーへのアクセスが許可されていません。")

if folderpath:
    # ユーザーがフォルダーを選択した場合の処理
    Folder_Processing(folderpath)
else:
    # ユーザーが何も選択せずにダイアログを閉じた場合の処理
    print("フォルダーが選択されませんでした。")