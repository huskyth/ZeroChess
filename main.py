import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
import logging

import coloredlogs

from Coach import Coach
from china_chess.algorithm.china_chese_game import *
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 50,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 150,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 40,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 30,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.8,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 50,

})


def main():
    log.info('Loading %s...', ChinaChessGame.__name__)
    g = ChinaChessGame()

    log.info('Loading %s...', nn.__name__)
    nnet = nn()

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
