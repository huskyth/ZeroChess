from china_chess.algorithm.china_chess_board import ChinaChessBoard
from china_chess.algorithm.icy_chess.game_board import GameBoard
from china_chess.constant import INTEGER_TO_STATE_STR, MAP_WIDTH, MAP_HEIGHT, STATE_STR_TO_INTEGER


class GameState:
    def __init__(self, enable_record_im=False):
        # self.state_str = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
        self.state_str = 'RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr'
        self.current_player = 'w'
        self.ys = '9876543210'[::-1]
        self.xs = 'abcdefghi'
        self.past_dic = {}
        self.max_repeat = 0
        self.last_move = ""
        self.move_number = 0
        self.enable_record_im = enable_record_im

    def is_check_catch(self):
        moveset = GameBoard.get_legal_moves(self.state_str, self.get_next_player())
        targetset = set([i[-2:] for i in moveset])

        wk, bk = self.get_king_pos()
        targetkingdic = {'b': wk, 'w': bk}
        targ_king = targetkingdic[self.get_next_player()]
        # TODO add long catch logic
        if targ_king in targetset:
            return True
        else:
            return False

    def get_king_pos(self):
        board = self.state_str.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')
        K, k = None, None
        for i in range(3):
            pos = board[i].find('K')
            if pos != -1:
                K = "{}{}".format(self.xs[pos], self.ys[i])
        for i in range(-1, -4, -1):
            pos = board[i].find('k')
            if pos != -1:
                k = "{}{}".format(self.xs[pos], self.ys[i])
        return K, k

    def game_end(self):
        if self.state_str.find('k') == -1:
            return True, 'w', "红获胜"
        elif self.state_str.find('K') == -1:
            return True, 'b', "黑获胜"
        wk, bk = self.get_king_pos()
        if self.max_repeat >= 3 and (self.last_move[-2:] != wk and self.last_move[-2:] != bk):
            return True, self.get_current_player(), "重复大于等于3"
        if self.max_repeat >= 4:
            return True, self.get_current_player(), "重复大于等于4"
        target_king_dic = {'b': wk, 'w': bk}
        move_set = GameBoard.get_legal_moves(self.state_str, self.get_current_player())

        target_set = set([i[-2:] for i in move_set])

        targ_king = target_king_dic[self.current_player]
        if targ_king in target_set:
            return True, self.current_player, "targ_king in target_set"
        return False, None, ""

    def get_current_player(self):
        return self.current_player

    def get_next_player(self):
        if self.current_player == 'w':
            return 'b'
        elif self.current_player == 'b':
            return 'w'

    def do_move(self, move):
        self.last_move = move
        self.state_str = GameBoard.sim_do_action(move, self.state_str)
        if self.current_player == 'w':
            self.current_player = 'b'
        elif self.current_player == 'b':
            self.current_player = 'w'
        self.past_dic.setdefault(self.state_str, [0, False, self.get_next_player()])  # times, longcatch/check
        self.past_dic[self.state_str][0] += 1
        self.max_repeat = self.past_dic[self.state_str][0]
        if self.enable_record_im:
            self.past_dic[self.state_str][1] = self.is_check_catch()
        self.move_number += 1

    def should_cutoff(self):
        verbose = False
        # the pastdic is empty when first move was made
        if self.move_number < 2:
            return False
        state_appear_num = self.past_dic[self.state_str][0]
        if state_appear_num > 1 and self.is_check_catch():
            if verbose:
                print("find something to cut off")
            return True
        else:
            return False

    def from_integer_to_state_str(self, board):
        temp = ''
        for line in board[::-1]:
            n = 0
            for cell in line:
                if cell in INTEGER_TO_STATE_STR:
                    if n != 0:
                        temp += str(n)
                        n = 0
                    temp += INTEGER_TO_STATE_STR[cell]
                else:
                    n += 1
            if n != 0:
                temp += str(n)
                n = 0

            temp += '/'
        self.state_str = temp[:-1]

    def str_to_integer_map(self):
        top_to_bottom = self.state_str.split('/')
        result = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)]
        cur_idx = 0

        def get_row_column(idx):
            return cur_idx // MAP_WIDTH, cur_idx - cur_idx // MAP_WIDTH * MAP_WIDTH

        for line in top_to_bottom:
            for cell in line:
                if cell in '0123456789':
                    n = int(cell)
                    for i in range(n):
                        row, column = get_row_column(cur_idx)
                        result[row][column] = 0
                        cur_idx += 1
                else:
                    row, column = get_row_column(cur_idx)
                    result[row][column] = STATE_STR_TO_INTEGER[cell]
                    cur_idx += 1
        return result

    def display(self, is_print_screen=False):
        b = ChinaChessBoard(None)
        b.to_chess_map(self.str_to_integer_map())

        return b.print_visible_string(is_print_screen=True)
