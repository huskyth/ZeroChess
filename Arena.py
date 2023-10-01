import logging

from tqdm import tqdm
import MCTS
from china_chess.algorithm.icy_chess.chess_board_from_icy import BaseChessBoard
from china_chess.algorithm.icy_chess.game_state import GameState
from china_chess.algorithm.mcts_async import prediction_worker

log = logging.getLogger(__name__)
from china_chess.algorithm.china_chess_board import *
from china_chess.algorithm.elo_helper import *


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.display = display
        self.elo_red = 0
        self.elo_black = 0

    def playGame(self):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        it = 0
        winner = None
        game_state = GameState()

        peace_round = 0
        remain_piece = countpiece(game_state.state_str)

        while not game_state.game_end()[0]:
            current_player = players[cur_player + 1]
            another_player = players[1 - cur_player]
            it += 1

            move = current_player.get_move_probs(game_state, predict_workers=[
                prediction_worker(current_player)])
            show_current_player = game_state.get_current_player()
            game_state.do_move(move)
            is_end, winner, info = game_state.game_end()
            current_player.update_with_move(move)
            another_player.update_with_move(move)
            cur_player *= -1

            remain_piece_round = countpiece(game_state.state_str)
            if remain_piece_round < remain_piece:
                remain_piece = remain_piece_round
                peace_round = 0
            else:
                peace_round += 1

            if it > 150 and peace_round > 60:
                temp = [x.strip() for x in game_state.display()]
                msg = str("\n".join(temp))
                write_line(file_name="terminal_in_arena", msg=msg, title="终结局面(和棋)")
                return None

            if is_end:
                temp = [x.strip() for x in game_state.display()]
                msg = str("\n".join(temp)) + "\n执行的行为是{}".format(move) + "\n执行该行为的玩家为{}".format(
                    show_current_player) + "\n当前玩家为{}".format(game_state.get_current_player())
                write_line(file_name="terminal_in_arena", msg=msg, title="终结局面：" + info)
                return winner
        return winner

    def playGames(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        self.player1, self.player2 = self.player2, self.player1
        draw_num = 0
        red_win = 0
        black_win = 0
        for _ in tqdm(range(num), desc="Arena.playGames"):
            game_result = self.playGame()
            assert game_result in ['w', 'b', None]
            if game_result == 'w':
                w = 1
                red_win += 1
            elif game_result == 'b':
                w = 0
                black_win += 1
            else:
                draw_num += 1
                w = 0.3

            self.elo_red, self.elo_black = compute_elo(self.elo_red, self.elo_black, w)

        # TODO://with error
        return self.elo_red, self.elo_black, draw_num, red_win, black_win
