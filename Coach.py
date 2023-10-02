import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

from Arena import Arena

from china_chess.algorithm.mcts_async import *
from china_chess.algorithm.tensor_board_tool import MySummary
from china_chess.constant import LABELS, LABELS_TO_INDEX, countpiece
import gc
from concurrent.futures import ProcessPoolExecutor

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in bakeup_main.py.
    """

    def __init__(self, nnet, args):
        self.nnet = nnet
        self.pnet = self.nnet.__class__()  # the competitor network
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.summary = MySummary("elo")

    def execute_episode(self, iter_number, mcts):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        gs = GameState()
        train_examples = []
        episode_step = 0

        peace_round = 0
        remain_piece = countpiece(gs.state_str)
        while True:
            episode_step += 1
            temp = int(episode_step < self.args.tempThreshold)

            move = mcts.get_move_probs(gs, temp=temp)
            pi = [0] * len(LABELS)
            pi[LABELS_TO_INDEX[move]] = 1
            bb = BaseChessBoard(gs.state_str)
            state_str = bb.get_board_arr()
            net_x = boardarr2netinput(state_str, gs.get_current_player())
            train_examples.append([net_x, pi, None, gs.get_current_player()])
            current_player = gs.get_current_player()
            gs.do_move(move)
            is_end, winner, info = gs.game_end()
            mcts.update_with_move(move)

            remain_piece_round = countpiece(gs.state_str)
            if remain_piece_round < remain_piece:
                remain_piece = remain_piece_round
                peace_round = 0
            else:
                peace_round += 1

            temp = [x.strip() for x in gs.display()]
            msg = str("\n".join(temp)) + "\n执行的行为是{}".format(move) + "\n执行该行为的玩家为{}".format(
                current_player) + "\n当前玩家为{}".format(gs.get_current_player())
            write_line(file_name="process" + str(iter_number), msg=msg, title="过程：" + info)

            if episode_step > 150 and peace_round > 60:
                for t in range(len(train_examples)):
                    train_examples[t][2] = 0
                temp = [x.strip() for x in gs.display()]
                msg = str("\n".join(temp))
                write_line(file_name="terminal", msg=msg, title="终结局面(和棋)：" + info)
                return train_examples

            if is_end:
                temp = [x.strip() for x in gs.display()]
                msg = str("\n".join(temp)) + "\n执行的行为是{}".format(move) + "\n执行该行为的玩家为{}".format(
                    current_player) + "\n当前玩家为{}".format(gs.get_current_player())
                write_line(file_name="terminal", msg=msg, title="终结局面：" + info)
                for t in range(len(train_examples)):
                    if winner == gs.get_current_player():
                        train_examples[t][2] = 1
                    else:
                        train_examples[t][2] = -1
                return train_examples

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                with ProcessPoolExecutor(max_workers=self.args.numEps) as executor:
                    futures = []
                    for i in range(10):
                        mcts = MCTS(policy_loop_arg=True, net=self.nnet)
                        futures.append(executor.submit(self.execute_episode, i, mcts))

                for r in futures:
                    self.trainExamplesHistory.append(r.result())

                # save the iteration examples to the history 

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(policy_loop_arg=True, net=self.pnet,
                         name="p-mcts")
            self.nnet.train(trainExamples, i)
            nmcts = MCTS(policy_loop_arg=True, net=self.nnet,
                         name="n-mcts")

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(pmcts, nmcts)
            red_elo_current, black_elo_current, draws, red_win, black_win = arena.playGames(self.args.arenaCompare)
            self.summary.add_float(x=i, y=red_elo_current, title='Red Elo')
            self.summary.add_float(x=i, y=black_elo_current, title='Black Elo')
            self.summary.add_float(x=i, y=red_win / self.args.arenaCompare, title='Red Win Rate')
            log.info('DRAWS : %d' % (draws / self.args.arenaCompare))
            if red_win > black_win:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            gc.collect()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
