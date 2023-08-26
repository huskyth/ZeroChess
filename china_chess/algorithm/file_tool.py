import os
from pickle import Pickler, Unpickler

import chardet


def all_files(path):
    files = [os.path.join(path, file) for file in os.listdir(path)]
    return files


def GetEncodingSheme(_filename):
    with open(_filename, 'rb') as file:
        buf = file.read()
    result = chardet.detect(buf)
    return result['encoding']


def write(file_path, data):
    with open(file_path, "wb+") as f:
        Pickler(f).dump(data)


def read(file_path):
    with open(file_path, "rb+") as f:
        return Unpickler(f).load()
