from china_chess.algorithm.data_loader import LoadData
from othello.pytorch.NNet import NNetWrapper
from china_chese_game import *


def main():
    ChinaChessGame()
    nn = NNetWrapper()
    examples = LoadData().get_all_examples()
    nn.train(examples)


if __name__ == "__main__":
    main()
