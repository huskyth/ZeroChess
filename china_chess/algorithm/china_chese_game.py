from Game import Game
from china_chess.algorithm.china_chess_board import *
import numpy as np
from china_chess.constant import *


class ChinaChessGame(Game):

    def __init__(self):
        self.width, self.height = 9, 10
        self.round_time = 0

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

    def _acc_logic(self, is_eat):
        if is_eat:
            self.continue_not_eat_chess_number = 0
        else:
            self.continue_not_eat_chess_number += is_eat

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        '''
        TODO:可能有问题
        '''
        b = ChinaChessBoard(None)
        b.to_chess_map(board)
        need_transfer = False
        if self._position_r_j(b.chessboard_map) and player == 1:
            need_transfer = True
            b = ChinaChessBoard(None)
            b.to_chess_map(board * -1)
        is_eat = b.move_chess(*b.algorithm_idx_to_row_column(action))
        if need_transfer:
            return b.to_integer_map() * -1, -player, is_eat
        return b.to_integer_map(), -player, is_eat

    def _position_r_j(self, chessboard_map):
        for x in range(10):
            for y in range(9):
                if chessboard_map[x][y] and chessboard_map[x][y].all_name == 'r_j':
                    return x <= 4

    def _debug_filter(self, board):
        if -1 in board[1] and -1 in board[2] and 5 in board[1]:
            return True
        return False

    def getValidMoves(self, board, player):
        '''
        TODO:可能会有问题
        '''
        b = ChinaChessBoard(None)
        b.to_chess_map(board)
        if self._position_r_j(b.chessboard_map):
            assert player == 1
            board_temp = board * -1
            b = ChinaChessBoard(None)
            b.to_chess_map(board_temp)
            player = -1

        if self._debug_filter(b.to_integer_map()):
            print()
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        assert player in [1, -1]
        player = 'r' if player == 1 else 'b'
        legalMoves = b.get_legal_moves(player)

        reverse_player = 'r' if player == 'b' else 'b'

        if len(legalMoves) == 0:
            if b.judge_win(reverse_player):
                print('游戏结束')
            else:
                raise Exception("No legal moves")
        valid_copy_idx = []
        for move in legalMoves:
            temp = b.row_column_to_algorithm_idx(*move)
            filter_value = self._filter(b.to_integer_map(), move[2], move[3], player)
            if True:
                valids[temp] = 1
            valid_copy_idx.append(temp)
        if not any(valids):
            '''
            可能产生问题'''
            for t in valid_copy_idx:
                valids[t] = 1
        return np.array(valids)

    def _filter(self, board, row, col, current_player):
        assert current_player in ['r', 'b']
        current_player = 1 if current_player == 'r' else -1
        return board[row][col] * current_player < 0

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        # TODO://解决action = -1的问题待验证
        b = ChinaChessBoard(None)
        b.to_chess_map(board)
        is_transfer = 1
        if self._position_r_j(b.chessboard_map):
            board_temp = board * -1
            b = ChinaChessBoard(None)
            b.to_chess_map(board_temp)
            is_transfer = -1

        player = 'r' if player == 1 else 'b'
        reverse_player = 'r' if player == 'b' else 'b'
        if b.judge_win(player):
            return True, 1 * is_transfer
        elif b.judge_win(reverse_player):
            return True, -1 * is_transfer
        else:
            return False, 0

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

    def _flip_up_down_and_left_right(self, board):
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        for i in range(len(board) // 2):
            for j in range(len(board[0])):
                result[i][j], result[9 - i][8 - j] = board[9 - i][8 - j], board[i][j]

        return result
