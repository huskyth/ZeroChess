from torch.utils.data import dataloader

from china_chess.algorithm.data_loader import LoadData
from othello.pytorch.NNet import NNetWrapper
from china_chese_game import *

EPOCH = 5


def train():
    train_dataset = LoadData()
    train_loader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False
    )
    ccg = ChinaCheseGame()
    net = NNetWrapper(ccg)

    test = train_dataset.get_all_examples()
    net.train(test)


if __name__ == '__main__':
    train()
