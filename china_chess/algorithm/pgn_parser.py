from china_chess.algorithm.china_chess_board import ChinaChessBoard
from reader_pgn import *
from china_chess.constant import *


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
                    result.append((i,j))
        if chess_name == '前':
            return result[0]
        else:
            return result[1]

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

        if chess_name == '后' or chess_name == '前':
            row_idx, col_idx = PGNParser.find_array_index_position_of_a_chess_with_back_or_before(chess_name,
                                                                                                  chess_current_position,
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
        if chess_name in ['士', '仕']:
            end_row_idx = row_idx - 1 if '进' == opt else row_idx + 1
        if chess_name in ['马']:
            det = 3 - abs(col_idx - end_col_idx)
            end_row_idx = row_idx - det if '进' == opt else row_idx + det
        if string == '炮２进５':
            print()
        print(string)
        move_string = LETTERS[col_idx] + NUMBERS[PGNParser._to_algorithm_index(row_idx)] + LETTERS[end_col_idx] + \
                      NUMBERS[PGNParser._to_algorithm_index(end_row_idx)]
        print(string, move_string)
        idx = LABELS_TO_INDEX[move_string]
        return row_idx, col_idx, end_row_idx, end_col_idx, PGNParser.build_pi(idx)


if __name__ == '__main__':
    y = PGNReader.read_from_pgn(DATASET_PATH / "0ceef374.pgn")
    b = ChinaChessBoard(None)
    episode = []
    current_player = 'r'
    for yi in y:
        if yi in PGNReader.RESULT_STRING_LIST:
            print(yi)
            b.print_visible_string()
            break
        row, col, end_row, end_col, build_pi = PGNParser.parse(yi[0], b, 'r')
        b.move_chess(row, col, end_row, end_col)
        b.flip_up_down_and_left_right()
        row, col, end_row, end_col, build_pi = PGNParser.parse(yi[1], b, 'b')
        b.move_chess(row, col, end_row, end_col)
        b.flip_up_down_and_left_right()
