import pandas as pd
import numpy as np

# サンプルデータフレームの作成
data = {'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}

df = pd.DataFrame(data)

# 挿入したいリスト
new_row_data = [10, 11, 12]

# リストをデータフレームに挿入
df.loc[len(df.index)] = new_row_data

# 結果の表示
print(df)
'''
# サンプルデータフレームの作成
data = {'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}

df = pd.DataFrame(data)

# 削除したい列の指定
column_to_remove = 2

# 列の削除と上詰め
df = df.drop(column_to_remove, axis=0)
df = df.loc[:, ~df.isnull().all()]

# 結果の表示
print(df)

first = "ai"
second = "test"
therd = first + "_" + second
test = pd.DataFrame()
a = [1,2,3]
test[first + "_" + second] = a
print(test)


# サンプルの多次元配列
a = [1,0,3]
b = [1,0,3]
a = np.array(a)
b = np.array(b)
my_array = np.divide(a,b)
my_array = np.divide(my_array,b)
print(my_array)
# 代入する値
replacement_value = 0  # 任意の値に置き換えてください

# ブールインデックスングを使用してNoneの場所を特定し、代入
a_list = []
mask = np.isnan(my_array)
for i in range(len(mask)):
    if mask[i] == True:
        a_list.append(i)
my_array[mask] = replacement_value

print(mask)
print(a_list)

# サンプルのリスト
my_list = [1, 2, None, 4, None, 6]

# 空欄に代入する数字
replacement_value = 3

# Noneの位置を特定し、代入
while None in my_list:
    index = my_list.index(None)
    my_list[index] = replacement_value

print(my_list)



def test_class(data_frame):
    df = pd.DataFrame(data_frame)
    row = len(df)
    columns = df.columns.tolist()
    result = pd.DataFrame()
    
    for i in range(row):
        result[f'x{i}'] = df.iloc[:,i]
    
    return result

df = {'名前': ['太郎', '花子', '次郎'],
        '年齢': [25, 30, 22],
        '都市': ['東京', '大阪', '名古屋']}
a = test_class(df)

a_row = len(a)
for i in range(a_row):
    f'x{i}' = a.iloc[:,i]

print(x1)
'''