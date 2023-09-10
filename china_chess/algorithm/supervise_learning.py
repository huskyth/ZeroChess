import numpy as np

from china_chess.algorithm.data_loader import LoadData
from othello.pytorch.NNet import NNetWrapper
from china_chese_game import *


def main():
    g = ChinaChessGame()
    nn = NNetWrapper()
    examples = LoadData().get_all_examples()
    for e in examples:
        x, y = g.getSymmetries(e[0], e[1])
        examples.append()
    nn.train(examples)


if __name__ == "__main__":
    main()
