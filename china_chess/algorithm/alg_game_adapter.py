from Coach import Coach
from china_chess.algorithm.cchess.const_function import is_kill_move, flipped_uci_labels
from china_chess.algorithm.cchess.game_board import GameBoard
from china_chess.algorithm.icy_chess.game_state import GameState
from china_chess.algorithm.sl_net import NNetWrapper
from china_chess.algorithm.china_chese_game import *
from china_chess.algorithm.tensor_board_tool import MySummary


class PolicyAdapter:

    def __init__(self):
        self.summary = MySummary("test")

        self.net = NNetWrapper(self.summary)
        self.net.load_checkpoint()
        self.game = ChinaChessGame()
        self.gs = GameState()
        self.coach = Coach(exploration=False)

    def _parse_move(self, string):
        return 9 - int(string[1]), LETTERS_TO_IND[string[0]], 9 - int(string[3]), LETTERS_TO_IND[string[2]],

    def _parse_move_new(self, string, c_player):
        assert c_player in ['w', 'b']
        string = flipped_uci_labels(string)
        return int(string[1]), LETTERS_TO_IND[string[0]], int(string[3]), LETTERS_TO_IND[string[2]],

    def _parse_to_move_idx_from_game_ind(self, row, column, end_row, end_column):
        return LETTERS[8 - column] + NUMBERS[row] + LETTERS[8 - end_column] + NUMBERS[end_row]

    def action_by_mcst(self, c_player):
        move = self.mcts.get_move_probs(self.gs)
        self.gs.do_move(move)
        self.mcts.update_with_move(move, allow_legacy=True)
        return move

    def get_next_policy(self, c_player):
        action, probs, win_rate = self.coach.get_action(self.coach.game_board.state, temperature=1)
        last_state = self.coach.game_board.state
        print(self.coach.game_board.current_player, " now take a action : ", action, "[Step {}]".format(
            self.coach.game_board.round))
        self.coach.game_board.state = GameBoard.sim_do_action(action, self.coach.game_board.state)
        self.coach.game_board.round += 1
        self.coach.game_board.current_player = "w" if self.coach.game_board.current_player == "b" else "b"
        if is_kill_move(last_state, self.coach.game_board.state) == 0:
            self.coach.game_board.restrict_round += 1
        else:
            self.coach.game_board.restrict_round = 0

        return self._parse_move_new(action, c_player)
