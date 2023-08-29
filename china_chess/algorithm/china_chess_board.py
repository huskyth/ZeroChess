from china_chess.policy_main_player_and_machine import *
import numpy as np
from china_chess.constant import *
import copy


class ChinaChessBoard(ChessBoard):

    def __init__(self, screen):
        super().__init__(screen)

    def get_legal_moves(self, current_player):
        chess_list = self.get_chess()
        chess_list = [c for c in chess_list if c.team == current_player]
        legal_moves = []
        for i in range(len(chess_list)):
            dot_list = self.get_put_down_postion(chess_list[i])
            for dot in dot_list:
                legal_moves.append((chess_list[i].row, chess_list[i].col, *dot))
        return legal_moves

    def algorithm_idx_to_row_column(self, idx, player):
        string = LABELS[idx]
        if player == 1:
            col = LETTERS_TO_IND[string[0]]
            row = 9 - NUMBERS_TO_IND[string[1]]
            new_col = LETTERS_TO_IND[string[2]]
            new_row = 9 - NUMBERS_TO_IND[string[3]]
        else:
            col = 8 - LETTERS_TO_IND[string[0]]
            row = NUMBERS_TO_IND[string[1]]
            new_col = 8 - LETTERS_TO_IND[string[2]]
            new_row = NUMBERS_TO_IND[string[3]]

        return row, col, new_row, new_col

    def row_column_to_algorithm_idx(self, row, col, new_row, new_col):
        string = LETTERS[col] + NUMBERS[9 - row] + LETTERS[new_col] + NUMBERS[9 - new_row]
        return LABELS_TO_INDEX[string]

    def flip_up_down_and_left_right(self):
        for i in range(len(self.chessboard_map) // 2):
            for j in range(len(self.chessboard_map[0])):
                self.chessboard_map[i][j], self.chessboard_map[9 - i][8 - j] = self.chessboard_map[9 - i][8 - j], \
                                                                               self.chessboard_map[i][j]

    def to_integer_map(self):
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                temp = self.chessboard_map[i][j]
                value = 0 if not temp else ABBREVIATION_TO_VALUE[temp.all_name]
                result[i][j] = value
        return np.array(result)

    def to_chess_map(self, board):
        for i in range(10):
            for j in range(9):
                if not board[i][j]:
                    self.chessboard_map[i][j] = None
                else:
                    self.chessboard_map[i][j] = Chess(None, VALUE_TO_ABBREVIATION[board[i][j]], i, j)

    def visual_current_map(self):
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        for i in range(10):
            for j in range(9):
                if self.chessboard_map[i][j]:
                    result[i][j] = self.chessboard_map[i][j].all_name
                else:
                    result[i][j] = '---'
        return result

    def move_chess(self, old_row, old_col, new_row, new_col):
        self.chessboard_map[new_row][new_col] = self.chessboard_map[old_row][old_col]
        self.chessboard_map[new_row][new_col].update_position(new_row, new_col)
        self.chessboard_map[old_row][old_col] = None


if __name__ == '__main__':
    # x = ChinaChessBoard(None)
    # x.print_visible_string()
    # y = x.to_integer_map()
    # print(y)
    # x.flip_up_down_and_left_right()
    # x.print_visible_string()
    s = (1, 2, 3)
    s += (5,)
    print(s)
