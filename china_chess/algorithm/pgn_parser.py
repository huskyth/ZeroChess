import os

from china_chess.algorithm.china_chess_board import ChinaChessBoard
from reader_pgn import *
from china_chess.constant import *
from file_tool import *

import numpy as np


class PGNParser:

    @staticmethod
    def _chinese_map_number(string):
        return (CHINESE_NUMBER_TO_INT[string] if string in CHINESE_NUMBER_TO_INT else
                NOT_SEE_NUMBER_TO_INT[string])

    @staticmethod
    def _to_array_index(value):
        return 9 - value

    @staticmethod
    def _to_algorithm_index(value):
        return 9 - value

    @staticmethod
    def find_array_index_position_of_a_chess(chess_name, chess_current_position, board, current_player):
        all_name = chess_from_chinese_to_english_char(chess_name, current_player)
        current_col = PGNParser._chinese_map_number(chess_current_position)
        current_col_idx = PGNParser._to_array_index(current_col)
        for i in range(len(board)):
            if board[i][current_col_idx] and board[i][current_col_idx].all_name == all_name:
                return i, current_col_idx

    @staticmethod
    def find_array_index_position_of_a_chess_with_back_or_before(chess_name, chess_current_position, board,
                                                                 current_player):
        all_name = chess_from_chinese_to_english_char(chess_current_position, current_player)
        result = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] and board[i][j].all_name == all_name:
                    result.append((i, j))
        if chess_name == '前':
            return result[0]
        else:
            return result[1]

    @staticmethod
    def find_array_index_position_of_a_chessZ(chess_name, chess_current_position, finded_position, opt, board,
                                              current_player):
        all_name = chess_from_chinese_to_english_char(chess_current_position, current_player)
        step = finded_position
        finded_position_idx = PGNParser._to_array_index(finded_position)
        result = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] and board[i][j].all_name == all_name:
                    if '平' == opt:
                        if j - 1 == finded_position_idx or j + 1 == finded_position_idx:
                            result.append((i, j))
                    else:
                        result.append((i, j))

        print(result)
        temp = {}
        for x in result:
            if x[1] not in temp:
                temp[x[1]] = [x]
            else:
                temp[x[1]].append(x)
        for y in temp:
            if len(temp[y]) == 2:
                if chess_name == '前':
                    return temp[y][0]
                else:
                    return temp[y][1]

    @staticmethod
    def find_array_index_position_of_a_chessX(chess_name, chess_current_position, opt, board,
                                              current_player):
        all_name = chess_from_chinese_to_english_char(chess_name, current_player)
        current_col = PGNParser._chinese_map_number(chess_current_position)
        current_col_idx = PGNParser._to_array_index(current_col)
        for i in range(len(board)):
            if board[i][current_col_idx] and board[i][current_col_idx].all_name == all_name:
                if opt == '进' and i == 5: continue
                return i, current_col_idx

    @staticmethod
    def find_array_index_position_of_a_chessS(chess_name, chess_current_position, opt, board,
                                              current_player):
        all_name = chess_from_chinese_to_english_char(chess_name, current_player)
        current_col = PGNParser._chinese_map_number(chess_current_position)
        current_col_idx = PGNParser._to_array_index(current_col)
        for i in range(len(board)):
            if board[i][current_col_idx] and board[i][current_col_idx].all_name == all_name:
                if opt == '进' and i == 7: continue
                return i, current_col_idx

    @staticmethod
    def build_pi(idx):
        temp = [0] * ALL_SELECTION
        temp[idx] = 1
        return temp

    @staticmethod
    def parse(string, b, current_player):
        board = b.chessboard_map
        chess_name, chess_current_position, opt, chess_next_position = string[0], string[1], string[
            2], PGNParser._chinese_map_number(string[3])
        print(string)
        if chess_name == '后' or chess_name == '前':

            if chess_current_position in ['兵', '卒']:
                row_idx, col_idx = PGNParser.find_array_index_position_of_a_chessZ(chess_name,
                                                                                   chess_current_position,
                                                                                   chess_next_position,
                                                                                   opt,
                                                                                   board,
                                                                                   current_player)

            else:
                row_idx, col_idx = PGNParser.find_array_index_position_of_a_chess_with_back_or_before(chess_name,
                                                                                                      chess_current_position,
                                                                                                      board,
                                                                                                      current_player)
        elif chess_name in ['相', '象']:
            row_idx, col_idx = PGNParser.find_array_index_position_of_a_chessX(chess_name,
                                                                               chess_current_position, opt,
                                                                               board, current_player)
        elif chess_name in ['仕', '士']:
            row_idx, col_idx = PGNParser.find_array_index_position_of_a_chessS(chess_name,
                                                                               chess_current_position, opt,
                                                                               board, current_player)
        else:

            row_idx, col_idx = PGNParser.find_array_index_position_of_a_chess(chess_name, chess_current_position, board,
                                                                              current_player)

        end_col_idx = PGNParser._to_array_index(chess_next_position)
        end_row_idx = row_idx
        if chess_name in ['相', '象']:
            end_row_idx = row_idx - 2 if '进' == opt else row_idx + 2
        if chess_name in ['车', '炮', '兵', '卒', '将', '帅']:
            det = chess_next_position
            if '进' == opt:
                end_row_idx = row_idx - det
                end_col_idx = col_idx
            elif '退' == opt:
                end_row_idx = row_idx + det
                end_col_idx = col_idx
            else:
                pass
        if chess_name in ['前', '后']:
            if chess_current_position in ['车', '炮', '兵', '卒']:
                det = chess_next_position
                if '进' == opt:
                    end_row_idx = row_idx - det
                    end_col_idx = col_idx
                elif '退' == opt:
                    end_row_idx = row_idx + det
                    end_col_idx = col_idx
                else:
                    pass
            if chess_current_position in ['马']:
                det = 3 - abs(col_idx - end_col_idx)
                end_row_idx = row_idx - det if '进' == opt else row_idx + det
        if chess_name in ['士', '仕']:
            end_row_idx = row_idx - 1 if '进' == opt else row_idx + 1
        if chess_name in ['马']:
            det = 3 - abs(col_idx - end_col_idx)
            end_row_idx = row_idx - det if '进' == opt else row_idx + det

        print(string, row_idx, col_idx)
        move_string = LETTERS[col_idx] + NUMBERS[PGNParser._to_algorithm_index(row_idx)] + LETTERS[end_col_idx] + \
                      NUMBERS[PGNParser._to_algorithm_index(end_row_idx)]
        print(string, move_string)
        idx = LABELS_TO_INDEX[move_string]
        return row_idx, col_idx, end_row_idx, end_col_idx, PGNParser.build_pi(idx)

    @staticmethod
    def parse_a_file(path):
        print(GetEncodingSheme(path), path)
        y = PGNReader.read_from_pgn(path)
        b = ChinaChessBoard(None)
        data = []
        result_value = None
        for yi in y:
            if yi in PGNReader.RESULT_STRING_LIST:
                result_value = yi
                break
            if yi[1] in PGNReader.RESULT_STRING_LIST:
                result_value = yi[1]
                break
        if result_value == '1-0':
            winner = RED_STRING
        elif result_value == '0-1':
            winner = BLACK_STRING
        else:
            winner = ''

        for yi in y:
            if yi in PGNReader.RESULT_STRING_LIST:
                print(yi)
                b.print_visible_string()
                break
            row, col, end_row, end_col, build_pi = PGNParser.parse(yi[0], b, RED_STRING)
            if not winner:
                value = 0
            else:
                value = 1 if winner == RED_STRING else -1
            temp = (b.to_integer_map(), build_pi, value, RED_STRING)
            data.append(temp)
            b.move_chess(row, col, end_row, end_col)

            b.print_visible_string()
            b.flip_up_down_and_left_right()
            if yi[1] in PGNReader.RESULT_STRING_LIST:
                print(yi[1])
                b.print_visible_string()
                break
            row, col, end_row, end_col, build_pi = PGNParser.parse(yi[1], b, BLACK_STRING)
            if not winner:
                value = 0
            else:
                value = 1 if winner == BLACK_STRING else -1
            temp = (b.to_integer_map(), build_pi, value, BLACK_STRING)
            data.append(temp)
            b.move_chess(row, col, end_row, end_col)
            b.flip_up_down_and_left_right()
        write(TRAIN_DATASET_PATH / (path.split(os.sep)[-1].split('.')[0] + '.examples'), data)


if __name__ == '__main__':
    files = all_files(DATASET_PATH)
    for file in files:
        PGNParser.parse_a_file(file)
