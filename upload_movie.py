# prompt: 動画ファイルをuploadしてfile_pathという変数に保存する
from google.colab import files

uploaded = files.upload()

file_path = list(uploaded.keys())[0]

print(f"Uploaded file path: {file_path}")


