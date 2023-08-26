import numpy as np

from othello.pytorch.NNet import NNetWrapper
from china_chess.constant import *
import copy


class PolicyAdapter:

    def __init__(self):
        self.net = NNetWrapper()
        self.net.load_checkpoint(folder=MODEL_PATH)

    def _to_integer_map(self, chessboard_map):
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                temp = chessboard_map[i][j]
                value = ABBREVIATION_TO_VALUE[temp]
                result[i][j] = value
        return np.array(result)

    def _flip_up_down_and_left_right(self, chessboard_map):
        chessboard_map = copy.deepcopy(chessboard_map)
        for i in range(len(chessboard_map) // 2):
            for j in range(len(chessboard_map[0])):
                chessboard_map[i][j], chessboard_map[9 - i][8 - j] = chessboard_map[9 - i][8 - j], \
                                                                     chessboard_map[i][j]
        return chessboard_map

    def _transfer_game_board(self, game_board, current_player):
        assert current_player == 'b'
        game_board = self._flip_up_down_and_left_right(game_board)
        return self._to_integer_map(game_board)

    def _parse_move(self, string):
        return int(string[1]), 8 - LETTERS_TO_IND[string[0]], int(string[3]), 8 - LETTERS_TO_IND[string[2]],

    def _parse_to_move_idx_from_game_ind(self, row, column, end_row, end_column):
        return LETTERS[8 - column] + NUMBERS[row] + LETTERS[8 - end_column] + NUMBERS[end_row]

    def get_next_policy(self, original_game_board_with_chess, current_player, can_move_list):
        if not can_move_list:
            return None, None, None, None
        original_game_board_with_chess_temp = [[j for j in range(len(original_game_board_with_chess[0]))] for i in
                                               range(len(original_game_board_with_chess))]
        for i in range(len(original_game_board_with_chess)):
            for j in range(len(original_game_board_with_chess[0])):
                if original_game_board_with_chess[i][j]:
                    original_game_board_with_chess_temp[i][j] = original_game_board_with_chess[i][j].all_name
                else:
                    original_game_board_with_chess_temp[i][j] = ''
        original_game_board_with_chess = original_game_board_with_chess_temp
        board = self._transfer_game_board(original_game_board_with_chess, current_player)
        pi, v = self.net.predict(board)
        max_move_string, max_move_p = None, -float('inf')
        for can in can_move_list:
            move_temp = self._parse_to_move_idx_from_game_ind(*can)
            idx = LABELS_TO_INDEX[move_temp]
            if max_move_p < pi[idx]:
                max_move_string = move_temp
                max_move_p = pi[idx]
        if max_move_string is None:
            print()
        return self._parse_move(max_move_string)
