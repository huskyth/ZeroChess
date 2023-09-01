import time
import os
from china_chess.constant import *

ticks = -1


def write_line(file_name, msg):
    global ticks
    if ticks == -1:
        ticks = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 打印按指定格式排版的时间

    with open(LOGGER_PATH / (ticks + "_" + file_name + "_os_ID:" + str(os.getpid()) + '.txt'), 'a',
              encoding='utf8') as f:
        f.write(msg + '\n')
        f.write('**********************************************************' + '\n\n')


def write_line_by_name(dir, file_name, msg):
    if not os.path.exists(dir):
        os.mkdir(dir)

    with open(dir + os.sep + file_name + ".txt", 'a', encoding='utf8') as f:
        f.write(msg + '\n')
