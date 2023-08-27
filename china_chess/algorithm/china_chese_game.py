from Game import Game
from china_chess.algorithm.china_chess_board import *
import numpy as np
from china_chess.constant import *


class ChinaChessGame(Game):

    def __init__(self):
        self.width, self.height = 9, 10

    def stringRepresentation(self, board):
        return board.tostring()

    def getInitBoard(self):
        # return initial board (numpy board)
        b = ChinaChessBoard(None)
        return b.to_integer_map()

    def getBoardSize(self):
        return self.width, self.height

    def getActionSize(self):
        return ALL_SELECTION

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = ChinaChessBoard(None)
        b.to_chess_map(board)
        b.move_chess(*b.algorithm_idx_to_row_column(action, player))
        b.flip_up_down_and_left_right()
        return b.to_integer_map(), -player

    def getValidMoves(self, board, player):
        # TODO://
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        b = ChinaChessBoard(None)
        b.to_chess_map(board)
        player = 'r' if player == 1 else 'b'
        legalMoves = b.get_legal_moves(player)

        if len(legalMoves) == 0:
            raise Exception("No legal moves")

        for x, y in legalMoves:
            valids[self.width * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = ChinaChessBoard(None)
        b.to_chess_map(board)
        player = 'r' if player == 1 else 'b'
        return b.judge_win(player)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return board
        else:
            _flip_up_down_and_left_right
        return player * board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.n ** 2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def _flip_up_down_and_left_right(self, board):
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        for i in range(len(board) // 2):
            for j in range(len(board[0])):
                result[i][j], result[9 - i][8 - j] = board[9 - i][8 - j], board[i][j]

        return result
