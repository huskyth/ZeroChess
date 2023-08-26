from othello.pytorch.NNet import NNetWrapper
from china_chese_game import *


def main():
    g = ChinaCheseGame()
    nn = NNetWrapper(g)
    examples = Reader.read()
    nn.train(examples)


if __name__ == "__main__":
    main()
