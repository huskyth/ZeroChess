from china_chess.algorithm.icy_chess.game_state import GameState
from china_chess.algorithm.mcts_async import *
from othello.pytorch.NNet import NNetWrapper
from china_chess.algorithm.china_chese_game import *


class PolicyAdapter:

    def __init__(self):
        self.net = NNetWrapper()
        self.net.load_checkpoint(folder=SL_MODEL_PATH, filename="best_loss.pth.tar")
        self.game = ChinaChessGame()
        self.mcts = MCTS(policy_value_fn=policy_value_fn_queue_of_my_net, policy_loop_arg=True)
        self.gs = GameState()

    def _parse_move(self, string):
        return 9 - int(string[1]), LETTERS_TO_IND[string[0]], 9 - int(string[3]), LETTERS_TO_IND[string[2]],

    def _parse_to_move_idx_from_game_ind(self, row, column, end_row, end_column):
        return LETTERS[8 - column] + NUMBERS[row] + LETTERS[8 - end_column] + NUMBERS[end_row]

    def action_by_mcst(self, c_player):
        acts, act_probs = self.mcts.get_move_probs(self.gs, predict_workers=[prediction_worker(self.mcts)])
        move = acts[np.argmax(act_probs)]
        self.gs.do_move(move)
        self.mcts.update_with_move(move, allow_legacy=False)
        return move

    def get_next_policy(self, c_player):
        move = self.action_by_mcst(c_player)

        return self._parse_move(move)
