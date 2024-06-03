# 3次元推定の流れ
## bagファイルから骨格座標を取り出す
#### 使用プログラム
- Programfiles/3destimation.py
- Programfiles/config.py

1. 'runalphapose.cmd'の"AlphaPosePath", "OutputDirPath"を自分のAlphaPoseのあるパスと出力したいフォルダパスに変える
2. 'config.py'中の"sourse_folder", "output_folder", "image-folder"を適切なものに変える. RealSenseの撮影条件に応じて"fps", "thres_min", "thres_max", "fill_min", "fill_max"を変更する(後述)
3. '3destimation.py'のmain関数中の"file_name"を骨格推定したいbagファイル名に変更する(拡張子は外して)
4. プログラム中のフォルダやファイルパスを適切なものに変える．
    参考：
    - line122: outputfolder
    - line127: csvdata
    - line129: csvname
    - line349: runalphapose.cmd
    - line351: json_open
基本はこれで実行すると3次元推定が行われる．出力は次のようになる．
- AlphaPoseのデータ: Data/Output/AlphaPoseData/{ファイル名}に格納．sep-json: json形式で18個のbody partsのx, y座標及び信頼度スコアが出力，vis: 骨格推定結果の画像，alphapose-result.json: AlphaPoseオリジナルの座標結果
※使用するデータはsep-jsonの方で，OpenPoseというソフトウェアの出力形式に合わせている
- 画像データ: Data/Output/image-folderに格納．bagファイルから切り取られた画像と背景除去された画像，および深度マップ画像が保存されている
- 3次元推定結果: Data/Output/csvdataに格納．x, y, z座標がメートル単位に直されて出力されている．正負については[RealSense D455(librealsense2)使ってみた #09' -座標系についての補足-](https://qiita.com/RoaaaA/items/f82d2e0b691d1d3ddf2a)に書いてある通り

#### 補足　背景除去について
撮影時に背景に被験者以外が入っても正確に推定を行うため，背景を除去し白で埋める作業を行っている．背景除去は有効範囲の距離を閾値として，それ以下や以上の範囲を除去するフィルタをかけることで実現．RealSenseの推奨撮影距離は0.6~6.0 m であるため，もともとは1~4 mに設定しているが，変更する場合は'config.py'の"thres_min"などを変更する．

#### 課題
やばいほど時間かかる．バッチファイル(cmdファイル)でAlphaPoseを動かすのがダサい．Pythonにちゃんと組み込みたい．

## データの平滑化や関節長さ取り出しなど
