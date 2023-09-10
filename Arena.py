import logging

from tqdm import tqdm
import ray
import MCTS
from china_chess.constant import MAX_NOT_EAR_NUMBER

log = logging.getLogger(__name__)
from china_chess.algorithm.china_chess_board import *


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
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
        self.game = game
        self.display = display

    @ray.remote
    def playGame(self, verbose=False, player1=None, player2=None):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [player2, None, player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        sum_of_is_eat = 0
        continue_list = []
        while not self.game.getGameEnded(board, curPlayer)[
            0] and sum_of_is_eat < MAX_NOT_EAR_NUMBER and not MCTS.MCTS.is_draw(continue_list):
            it += 1
            ChinaChessBoard.print_visible_string_from_integer_map(board,
                                                                  title='第{}次着'.format(

                                                                      it
                                                                  ), iter_number=-438)
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if len(continue_list) == 12:
                del continue_list[0]
            continue_list.append(action)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer, is_eat = self.game.getNextState(board, curPlayer, action)
            if is_eat:
                sum_of_is_eat = 0
            else:
                sum_of_is_eat += is_eat
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        if sum_of_is_eat >= MAX_NOT_EAR_NUMBER or MCTS.MCTS.is_draw(continue_list):
            return 0
        is_end, value = self.game.getGameEnded(board, curPlayer)
        assert is_end
        return curPlayer * value

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        y = [self.playGame.remote(self.player1, self.player2) for i in range(num)]
        ready_ids, remaining_ids = ray.wait(y)
        res_ready = ray.get(ready_ids)
        res_remaining = ray.get(remaining_ids)
        temp = res_ready + res_remaining
        assert len(temp) == num
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = temp[_]
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        y = [self.playGame.remote(self.player2, self.player1) for i in range(num)]
        ready_ids, remaining_ids = ray.wait(y)
        res_ready = ray.get(ready_ids)
        res_remaining = ray.get(remaining_ids)
        temp = res_ready + res_remaining
        assert len(temp) == num
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = temp[_]
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
