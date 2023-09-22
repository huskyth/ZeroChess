from china_chess.algorithm.icy_chess.game_state import GameState
from china_chess.algorithm.mcts_async import *
from othello.pytorch.NNet import NNetWrapper
from china_chess.algorithm.china_chese_game import *


class PolicyAdapter:

    def __init__(self):
        self.net = NNetWrapper()
        self.net.load_checkpoint(folder=SL_MODEL_PATH, filename="best_loss.pth.tar")
        self.game = ChinaChessGame()
        self.pmcts = MCTS(policy_value_fn=policy_value_fn_queue, policy_loop_arg=True)

    def _parse_move(self, string):
        return 9 - int(string[1]), LETTERS_TO_IND[string[0]], 9 - int(string[3]), LETTERS_TO_IND[string[2]],

    def _parse_to_move_idx_from_game_ind(self, row, column, end_row, end_column):
        return LETTERS[8 - column] + NUMBERS[row] + LETTERS[8 - end_column] + NUMBERS[end_row]

    def action_by_mcst(self, board):
        gs = GameState()
        gs.from_integer_to_state_str(board)
        return np.argmax(self.pmcts.get_move_probs(gs, predict_workers=[prediction_worker(self.pmcts)]))

    def get_next_policy(self, original_game_board_with_chess, c_player):
        ccb = ChinaChessBoard(None)
        ccb.deepcopy(original_game_board_with_chess)

        board = ccb.to_integer_map()
        c_player = 1 if c_player == 'r' else -1
        board = self.game.getCanonicalForm(board, c_player)

        idx = self.action_by_mcst(board)

        move = LABELS[idx]

        return self._parse_move(move)
