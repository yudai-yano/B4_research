# 概要

Python version 3.9
project management tool is venv

使用する場合は,venvで仮想環境を作成し,activateしてから,
`pip install -r requirements.txt`でライブラリをインストールすること.

venvで仮想環境を作る場合は,
`py -3.9 -m venv yourvenvname`
yourvenvnameには好きな仮想環境の名前を入れてください.
Activateは
`yourvenvname/scripts/activate`
で行います.

gen_movie_from_bagfile.py: bagfile(デプスカメラから出力されるファイル)から動画を作成する.
init_frame_tool.py: 動画内のあるフレームを特定するためのザッピングツール.
monitor.py: init_frame_tool.py内で使用している自作クラス.

program: pythonスクリプトとipynbスクリプトが格納されている.
bagfiles: 処理に使用するbagfileを格納する.
out: 出力ファイル用等に使用する
