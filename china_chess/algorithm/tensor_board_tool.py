import os

from tensorboard import summary
from china_chess.constant import SUMMARY_PATH
from torch.utils.tensorboard import SummaryWriter

log_dir = SUMMARY_PATH


class MySummary:

    def __init__(self, log_dir_name=None):
        log_path = str(log_dir / log_dir_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)

    def add_float(self, x, y, title):
        self.writer.add_scalar(title, y, x)

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    m = MySummary('test3')
    for i in range(10):
        m.add_float(i, i ** 2, "test_scalor")
    m.close()
