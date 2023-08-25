from china_chess.algorithm.china_chess_board import ChinaChessBoard
from reader_pgn import *
from china_chess.constant import *


class PGNParser:

    @staticmethod
    def _chinese_map_number(string):
        return (CHINESE_NUMBER_TO_INT[string] if string in CHINESE_NUMBER_TO_INT else
                NOT_SEE_NUMBER_TO_INT[string])

    @staticmethod
    def find_position_of_a_chinese(chinese_name, chinese_position, board, current_player):
        all_name = from_chinese_to_english_char(chinese_name, current_player)
        col = 9 - PGNParser._chinese_map_number(chinese_position)
        for i in range(len(board)):
            if board[i][col] and board[i][col].all_name == all_name:
                return 9 - i, col

    @staticmethod
    def build_pi(idx):
        temp = [0] * ALL_SELECTION
        temp[idx] = 1
        return temp

    @staticmethod
    def parse(string, board, current_player):
        chinese_name, chinese_position, opt, next_position_chinese = string[0], string[1], string[2], string[3]
        next_position_col = 9 - PGNParser._chinese_map_number(next_position_chinese)

        row, col = PGNParser.find_position_of_a_chinese(chinese_name, chinese_position, board, current_player)
        end_col = next_position_col
        end_row = -1
        if chinese_name in ['相', '象']:
            end_row = row + 2 if '进' == opt else row - 2
        if chinese_name in ['车', '炮', '兵', '卒', '将', '帅']:
            det = PGNParser._chinese_map_number(next_position_chinese)
            if '进' == opt:
                end_row = row - det
            elif '退' == opt:
                end_row = row + det
            else:
                end_row = row
        if chinese_name in ['士', '仕']:
            end_row = row - 1 if '进' == opt else row + 1
        if chinese_name in ['马']:
            det = 3 - abs(col - end_col)
            end_row = row - det if '进' == opt else row + det
        move_string = LETTERS[col] + NUMBERS[row] + LETTERS[end_col] + NUMBERS[end_row]
        print(move_string)
        idx = LABELS_TO_INDEX[move_string]
        return row, col, end_row, end_col, PGNParser.build_pi(idx)


if __name__ == '__main__':
    y = PGNReader.read_from_pgn(DATASET_PATH / "0a45c9b6.pgn")
    b = ChinaChessBoard(None)
    episode = []
    current_player = 'r'
    for yi in y:
        row, col, end_row, end_col, build_pi = PGNParser.parse('车一平二', b.chessboard_map, 'r')
