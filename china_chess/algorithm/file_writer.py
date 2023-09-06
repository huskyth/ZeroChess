import time
import os
from china_chess.constant import *

ticks = -1


def write_line(file_name, msg, title):
    global ticks
    if ticks == -1:
        ticks = str(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))  # 打印按指定格式排版的时间
    if not os.path.exists(LOGGER_PATH):
        os.mkdir(LOGGER_PATH)
    ticks = ticks.replace(':', '：')
    name = str(LOGGER_PATH / (ticks + "_" + file_name + "_os_ID_" + str(os.getpid()) + '.txt'))
    with open(name, 'a', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write(msg + '\n')
        f.write('**********************************************************' + '\n\n')


def write_line_by_name(dir, file_name, msg):
    if not os.path.exists(dir):
        os.mkdir(dir)

    with open(dir + os.sep + file_name + ".txt", 'a', encoding='utf8') as f:
        f.write(msg + '\n')
