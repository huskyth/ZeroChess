import os


def all_files(path):
    files = [os.path.join(path, file) for file in os.listdir(path)]

    # 遍历文件列表，输出文件名
    for file in files:
        print(file)
    return files

import chardet


def GetEncodingSheme(_filename):
    with open(_filename, 'rb') as file:
        buf = file.read()
    result = chardet.detect(buf)
    return result['encoding']

