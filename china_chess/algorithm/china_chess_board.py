from china_chess.policy_main_player_and_machine import *
import numpy as np
from china_chess.constant import *


class ChinaChessBoard(ChessBoard):

    def __init__(self, screen):
        super().__init__(screen)

    def get_legal_moves(self, current_player):
        chess_list = self.get_chess()
        chess_list = [c for c in chess_list if c.team == current_player]
        legal_moves = []
        for i in range(len(chess_list)):
            dot_list = self.get_put_down_postion(chess_list[i])
            legal_moves.append(dot_list)
        return legal_moves

    def flip_up_down_and_left_right(self):
        for i in range(len(self.chessboard_map) // 2):
            for j in range(len(self.chessboard_map[0])):
                self.chessboard_map[i][j], self.chessboard_map[9 - i][8 - j] = self.chessboard_map[9 - i][8 - j], \
                                                                               self.chessboard_map[i][j]
        print()

    def to_integer_map(self):
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                temp = self.chessboard_map[i][j]
                value = 0 if not temp else ABBREVIATION_TO_VALUE[temp.all_name]
                result[i][j] = value
        return np.array(result)


if __name__ == '__main__':
    # x = ChinaChessBoard(None)
    # x.print_visible_string()
    # y = x.to_integer_map()
    # print(y)
    # x.flip_up_down_and_left_right()
    # x.print_visible_string()
    s = (1,2,3)
    s += (5,)
    print(s)
