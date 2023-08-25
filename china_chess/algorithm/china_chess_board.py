from china_chess.policy_main import *


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


if __name__ == '__main__':
    x = ChinaChessBoard(None).get_chess()
    print(x)
