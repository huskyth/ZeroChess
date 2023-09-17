import numpy as np
from matplotlib import pyplot as plt

from china_chess.algorithm.data_loader import LoadData

from china_chese_game import *
from china_chess.algorithm.sl_net import NNetWrapper


def main():
    g = ChinaChessGame()
    nn = NNetWrapper()
    examples = LoadData().get_all_examples()
    for i in range(len(examples)):
        e = examples[i]
        b, p = g.getSymmetries(e[0], e[1])[1]
        examples.append((b, p, e[2]))

    nn.load_checkpoint(SL_MODEL_PATH, "best_loss.pth.tar")
    nn.train(examples)


if __name__ == "__main__":
    main()
