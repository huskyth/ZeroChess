import numpy as np
from matplotlib import pyplot as plt

from china_chess.algorithm.data_loader import LoadData

from china_chese_game import *
from china_chess.algorithm.sl_net import NNetWrapper
from china_chess.algorithm.icy_chess.game_convert import *


def expand_data(board):

    data, player, v = board
    data = integer_to_state_str(data)
    temp = boardarr2netinput(data, player)
    return temp


def main():
    g = ChinaChessGame()
    nn = NNetWrapper()
    examples = LoadData().get_all_examples()
    for i in range(len(examples)):
        e = examples[i]
        b, p = g.getSymmetries(e[0], e[1])[1]
        examples.append((b, p, e[2]))

    for e_i in range(len(examples)):
        examples[e_i] = expand_data(examples[e_i])

    # nn.train(examples)


if __name__ == "__main__":
    main()
