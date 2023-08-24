import os

from ChinaChess.constant import *
from ChinaChessBoard import *


def read_from_pgn(file_name):
    with open(file_name, encoding="GB2312") as file:
        flines = file.readlines()

    lines = []
    for line in flines:
        it = line.strip()  # TODO, fix it in linux

        if len(it) == 0:
            continue

        lines.append(it)

    lines = __get_headers(lines)
    lines, docs = __get_comments(lines)
    return __get_steps(lines)


def __get_headers(lines):
    index = 0
    for line in lines:

        if line[0] != "[":
            return lines[index:]

        if line[-1] != "]":
            raise Exception("Format Error on line %" % (index + 1))

        items = line[1:-1].split("\"")

        if len(items) < 3:
            raise Exception("Format Error on line %" % (index + 1))

        index += 1


def __get_comments(lines):
    if lines[0][0] != "{":
        return (lines, None)

    docs = lines[0][1:]

    # 处理一注释行的情况
    if docs[-1] == "}":
        return (lines[1:], docs[:-1].strip())

    # 处理多行注释的情况
    index = 1

    for line in lines[1:]:
        if line[-1] == "}":
            docs = docs + "\n" + line[:-1]
            return (lines[index + 1:], docs.strip())

        docs = docs + "\n" + line
        index += 1

        # 代码能运行到这里，就是出了异常了
    raise Exception("Comments not closed")


def __get_token(token_mode, lines):
    pass


def __get_steps(lines, next_step=1):
    all_step = []
    temp = ["*", "1-0", "0-1", "1/2-1/2"]
    for line in lines:
        if line[0] not in '0123456789':
            continue
        if line in temp:
            all_step.append(line)
            return all_step
        items = line.split(".")
        steps = items[1].strip().split(" ")
        all_step.append(steps)

    return all_step


def from_chinese_to_english_char(chinese_name, cur_player):
    name_2_english_char = {
        '相': 'r_x',
        '卒': 'b_z',
        '象': 'b_x',
        '兵': 'r_z',
        '帅': 'r_j',
        '将': 'b_j',
        '仕': 'r_s',
        '士': 'b_s',

    }
    temp = {'炮': '_p', '马': '_m', '车': '_c'}
    if chinese_name in temp:
        return cur_player + temp[chinese_name]
    return name_2_english_char[chinese_name]


chinese_number_2_int = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
}
not_see_number_2_int = {
    '１': 1, '２': 2, '３': 3, '４': 4, '５': 5, '６': 6, '７': 7, '８': 8, '９': 9
}

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def find_position_of_a_chinese(chinese_name, chinese_position, board, current_player):
    all_name = from_chinese_to_english_char(chinese_name, current_player)
    col = chinese_number_2_int[chinese_position] if chinese_position in chinese_number_2_int else not_see_number_2_int[
        chinese_position]
    for i in range(len(board)):
        if board[i][col].all_name == all_name:
            return i, col


def parse(string, board, current_player):
    chinese_name, chinese_position, opt, next_position_chinese = string[0], string[1], string[2], string[3]
    next_position_col = chinese_number_2_int[chinese_position] if chinese_position in chinese_number_2_int else \
        not_see_number_2_int[chinese_position]

    row, col = find_position_of_a_chinese(chinese_name, chinese_position, board, current_player)
    end_col = next_position_col
    end_row = -1
    if chinese_name in ['相', '象']:
        end_row = row - 2 if '进' == opt else row + 2
    if chinese_name in ['车','炮','兵','卒','将','帅']:
        if '进' == opt:

        elif '退' == opt:

        else:

        row = row + 2 if '进' == opt else row - 2



if __name__ == '__main__':
    labels = create_uci_labels()
    label_to_index = {}
    for i, l in enumerate(labels):
        label_to_index[l] = i
    y = read_from_pgn(r"C:\Users\Administrator\Desktop\ZeroChess\ChinaChess\dataset\0a45c9b6.pgn")
    b = ChinaChessBoard()
    episode = []
    for yi in y:
