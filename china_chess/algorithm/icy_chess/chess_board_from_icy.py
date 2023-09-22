from enum import IntEnum
import copy

from china_chess.algorithm.icy_chess.exception import CChessException
from china_chess.algorithm.icy_chess.move import Move
from china_chess.algorithm.icy_chess.piece import Piece, fench_to_species, Pos

import numpy as np

_text_board = [
    # u' 1  2   3   4   5   6   7   8   9',
    u'0 ┌─┬─┬─┬───┬─┬─┬─┐',
    u'  │  │  │  │＼│／│　│　│　│',
    u'1 ├─┼─┼─┼─※─┼─┼─┼─┤',
    u'  │　│　│　│／│＼│　│　│　│',
    u'2 ├─┼─┼─┼─┼─┼─┼─┼─┤',
    u'  │　│　│　│　│　│　│　│　│',
    u'3 ├─┼─┼─┼─┼─┼─┼─┼─┤',
    u'  │　│　│　│　│　│　│　│　│',
    u'4 ├─┴─┴─┴─┴─┴─┴─┴─┤',
    u'  │　                         　 │',
    u'5 ├─┬─┬─┬─┬─┬─┬─┬─┤',
    u'  │　│　│　│　│　│　│　│　│',
    u'6 ├─┼─┼─┼─┼─┼─┼─┼─┤',
    u'  │　│　│　│　│　│　│　│　│',
    u'7 ├─┼─┼─┼─┼─┼─┼─┼─┤',
    u'  │　│　│　│＼│／│　│　│　│',
    u'8 ├─┼─┼─┼─※─┼─┼─┼─┤',
    u'  │　│　│　│／│＼│　│　│　│',
    u'9 └─┴─┴─┴───┴─┴─┴─┘',
    u'  0   1   2   3   4   5   6   7   8'
    # u'  九 八  七  六  五  四  三  二  一'
]
_fench_txt_name_dict = {
    'K': u"帅",
    'k': u"将",
    'A': u"仕",
    'a': u"士",
    'B': u"相",
    'b': u"象",
    'N': u"马",
    'n': u"碼",
    'R': u"车",
    'r': u"砗",
    'C': u"炮",
    'c': u"砲",
    'P': u"兵",
    'p': u"卒"

}


# -----------------------------------------------------#

def _pos_to_text_board_pos(pos):
    return Pos(2 * pos.x + 2, (9 - pos.y) * 2)


def _fench_to_txt_name(fench):
    return _fench_txt_name_dict[fench]


class ChessSide(IntEnum):
    RED = 0
    BLACK = 1

    @staticmethod
    def next_side(side):
        return {ChessSide.RED: ChessSide.BLACK, ChessSide.BLACK: ChessSide.RED}[side]


class BaseChessBoard(object):
    def __init__(self, fen=None):
        self.clear()
        if fen: self.from_fen(fen)

    def clear(self):
        self._board = [[None for x in range(9)] for y in range(10)]
        self.move_side = ChessSide.RED

    def copy(self):
        return copy.deepcopy(self)

    def put_fench(self, fench, pos):
        if self._board[pos.y][pos.x] != None:
            return False

        self._board[pos.y][pos.x] = fench

        return True

    def get_fench(self, pos):
        return self._board[pos.y][pos.x]

    def get_piece(self, pos):
        fench = self._board[pos.y][pos.x]

        if not fench:
            return None

        return Piece.create(self, fench, pos)

    def is_valid_move_t(self, move_t):
        pos_from, pos_to = move_t
        return self.is_valid_move(pos_from, pos_to)

    def is_valid_move(self, pos_from, pos_to):

        '''
        只进行最基本的走子规则检查，不对每个子的规则进行检查，以加快文件加载之类的速度
        '''

        if not (0 <= pos_to.x <= 8): return False
        if not (0 <= pos_to.y <= 9): return False

        fench_from = self._board[pos_from.y][pos_from.x]
        if not fench_from:
            return False

        _, from_side = fench_to_species(fench_from)

        # move_side 不是None值才会进行走子颜色检查，这样处理某些特殊的存储格式时会处理比较迅速
        if self.move_side and (from_side != self.move_side):
            return False

        fench_to = self._board[pos_to.y][pos_to.x]
        if not fench_to:
            return True

        _, to_side = fench_to_species(fench_to)

        return (from_side != to_side)

    def _move_piece(self, pos_from, pos_to):

        fench = self._board[pos_from.y][pos_from.x]
        self._board[pos_to.y][pos_to.x] = fench
        self._board[pos_from.y][pos_from.x] = None

        return fench

    def move(self, pos_from, pos_to):
        pos_from.y = 9 - pos_from.y
        pos_to.y = 9 - pos_to.y
        if not self.is_valid_move(pos_from, pos_to):
            return None

        board = self.copy()
        fench = self.get_fench(pos_to)
        self._move_piece(pos_from, pos_to)

        return Move(board, pos_from, pos_to)

    def move_iccs(self, move_str):
        move_from, move_to = Move.from_iccs(move_str)
        return self.move(move_from, move_to)

    def move_chinese(self, move_str):
        move_from, move_to = Move.from_chinese(self, move_str)
        return self.move(move_from, move_to)

    def next_turn(self):
        if self.move_side == None:
            return None

        self.move_side = ChessSide.next_side(self.move_side)

        return self.move_side

    def from_fen(self, fen):

        num_set = set(('1', '2', '3', '4', '5', '6', '7', '8', '9'))
        ch_set = set(('k', 'a', 'b', 'n', 'r', 'c', 'p'))

        self.clear()

        if not fen or fen == '':
            return

        fen = fen.strip()

        x = 0
        y = 9

        for i in range(0, len(fen)):
            ch = fen[i]

            if ch == ' ':
                break
            elif ch == '/':
                y -= 1
                x = 0
                if y < 0: break
            elif ch in num_set:
                x += int(ch)
                if x > 8: x = 8
            elif ch.lower() in ch_set:
                if x <= 8:
                    self.put_fench(ch, Pos(x, y))
                    x += 1
            else:
                return False

        fens = fen.split()

        self.move_side = None
        if (len(fens) >= 2) and (fens[1] == 'b'):
            self.move_side = ChessSide.BLACK
        else:
            self.move_side = ChessSide.RED

        if len(fens) >= 6:
            self.round = int(fens[5])
        else:
            self.round = 1

        return True

    def count_x_line_in(self, y, x_from, x_to):
        return reduce(lambda count, fench: count + 1 if fench else count, self.x_line_in(y, x_from, x_to), 0)

    def count_y_line_in(self, x, y_from, y_to):
        return reduce(lambda count, fench: count + 1 if fench else count, self.y_line_in(x, y_from, y_to), 0)

    def x_line_in(self, y, x_from, x_to):
        step = 1 if x_to > x_from else -1
        return [self._board[y][x] for x in range(x_from + step, x_to, step)]

    def y_line_in(self, x, y_from, y_to):
        step = 1 if y_to > y_from else -1
        return [self._board[y][x] for y in range(y_from + step, y_to, step)]

    def to_fen(self):
        return self.to_short_fen() + ' - - 0 1'

    def to_short_fen(self):
        fen = ''
        count = 0
        for y in range(9, -1, -1):
            for x in range(9):
                fench = self._board[y][x]
                if fench:
                    if count is not 0:
                        fen += str(count)
                        count = 0
                    fen += fench
                else:
                    count += 1

            if count > 0:
                fen += str(count)
                count = 0

            if y > 0: fen += '/'

        if self.move_side is ChessSide.BLACK:
            fen += ' b'
        elif self.move_side is ChessSide.RED:
            fen += ' w'
        else:
            raise CChessException('Move Side Error' + str(self.move_side))

        return fen

    def dump_board(self):

        board_str = _text_board[:]

        y = 0
        for line in self._board:
            x = 0
            for ch in line:
                if ch:
                    pos = _pos_to_text_board_pos(Pos(x, y))
                    new_text = board_str[pos.y][:pos.x] + _fench_to_txt_name(ch) + board_str[pos.y][pos.x + 1:]
                    board_str[pos.y] = new_text
                x += 1
            y += 1

        return board_str

    def print_board(self):

        board_txt = self.dump_board()
        print()
        for line in board_txt:
            print(line)
        print()

    def get_board_arr(self):
        return np.asarray(self._board[::-1])
