from Game import Game
from china_chess_board import *
import numpy as np
from china_chess.constant import *


class ChinaCheseGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    def __init__(self):
        self.width, self.height = 9, 10

    def getInitBoard(self):
        # return initial board (numpy board)
        b = ChinaChessBoard()
        return np.array(b.pieces)

    def getBoardSize(self):
        return self.width, self.height

    def getActionSize(self):
        return ALL_SELECTION

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # TODO://
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        b = ChinaChessBoard()
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)

        if len(legalMoves) == 0:
            raise Exception("No legal moves")

        for x, y in legalMoves:
            valids[self.width * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = ChinaChessBoard()
        b.pieces = np.copy(board)
        return b.is_game_end(player)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
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
