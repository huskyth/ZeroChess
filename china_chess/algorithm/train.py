from china_chess.algorithm.data_loader import LoadData
from china_chess.constant import MODEL_PATH
from othello.pytorch.NNet import NNetWrapper


def train():
    train_dataset = LoadData()

    net = NNetWrapper()
    net.load_checkpoint(folder=MODEL_PATH)

    test = train_dataset.get_all_examples()
    net.train(test)


if __name__ == '__main__':
    train()
