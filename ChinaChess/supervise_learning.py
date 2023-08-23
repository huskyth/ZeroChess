import logging

import coloredlogs

from Coach import Coach
from othello.pytorch.NNet import NNetWrapper as nn
from othello.pytorch.OthelloNNet import OthelloNNet
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'batch_size': 128,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    g = ChineseChess()
    nnet = OthelloNNet(g, args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    for e in range(args.numEps):
        for batch_idx, batch in enumerate(training_dataset):
            state_true, v_true = batch
            pi, v = nnet(state)
            y = loss()
            y.backword()
            optimizer.step()
            ...


if __name__ == "__main__":
    main()
